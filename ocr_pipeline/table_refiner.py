from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .engines.paddle_ocr_engine import PaddleOCRAdapter
from .engines.paddle_table_engine import PaddleTableRecognitionAdapter
from .engines.paddle_vl_engine import PaddleVLAdapter
from .pdf_utils import extract_pdf_page_words
from .preprocessing import enhance_table_lines
from .profile import profile_input
from .schemas import OCRBlock, OCRDocument, OCRPage
from .table_cropper import PDFTableCropper
from .utils.io import ensure_dir, save_json
from .utils.text import normalize_whitespace


HTML_ROW_RE = re.compile(r"<tr\b", re.I)
HTML_CELL_RE = re.compile(r"<t[dh]\b", re.I)
HEADER_HINT_RE = re.compile(r"(구분|공정|가동율|계획|실적|수량|금액|달성율|진척율|양품|불량)", re.I)


class TableRefiner:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._pdf_path: Path | None = None
        self._pdf_word_cache: dict[tuple[int, int, int], list[tuple[float, float, float, float, str]]] = {}

    def refine_document(self, input_path: str | Path, doc: OCRDocument, output_dir: str | Path) -> OCRDocument:
        input_path = Path(input_path)
        if input_path.suffix.lower() != ".pdf" or not doc.pages:
            return doc
        self._pdf_path = input_path

        engine = self._build_engine()
        if engine is None or not engine.available():
            return doc

        refine_dir = ensure_dir(Path(output_dir) / "table_refine")
        cropper = PDFTableCropper(input_path, refine_dir / "rendered_pages", dpi=self.config.table_refine.dpi)
        report: list[dict[str, Any]] = []

        for page in doc.pages:
            table_indices = [idx for idx, block in enumerate(page.blocks) if block.block_type == "table"]
            for table_ord, block_idx in enumerate(table_indices[: self.config.table_refine.max_tables_per_page], start=1):
                block = page.blocks[block_idx]
                crop_bbox = self._estimate_table_bbox(page, block_idx)
                if crop_bbox is None:
                    report.append(self._report_item(page.page_no, table_ord, "skipped", "no_bbox"))
                    continue

                crop_path = refine_dir / "crops" / f"page_{page.page_no:04d}_table_{table_ord:03d}.png"
                saved = cropper.crop_table(
                    page,
                    crop_bbox,
                    crop_path,
                    padding=self.config.table_refine.padding,
                    min_width=self.config.table_refine.min_crop_width,
                    min_height=self.config.table_refine.min_crop_height,
                )
                if saved is None:
                    report.append(self._report_item(page.page_no, table_ord, "skipped", "crop_too_small"))
                    continue

                # Enhance table lines in the crop for better structure detection
                if self.config.preprocess.strengthen_table_lines:
                    try:
                        import cv2
                        import numpy as np
                        gray = cv2.imread(str(saved), cv2.IMREAD_GRAYSCALE)
                        if gray is not None:
                            enhanced = enhance_table_lines(gray, line_kernel_scale=self.config.preprocess.line_kernel_scale)
                            cv2.imwrite(str(saved), enhanced)
                    except Exception:
                        pass

                work_dir = refine_dir / f"page_{page.page_no:04d}_table_{table_ord:03d}" / engine.name
                candidate = self._run_table_engine(engine, saved, work_dir)
                if candidate is None:
                    report.append(self._report_item(page.page_no, table_ord, "skipped", "no_candidate"))
                    continue

                old_score = self._score_table_block(block)
                new_score = self._score_table_block(candidate)
                replaced = new_score > old_score
                if replaced:
                    merged_extra = {
                        **block.extra,
                        "table_refined": True,
                        "table_refine_engine": engine.name,
                        "table_crop_path": str(saved),
                    }
                    page.blocks[block_idx] = replace(
                        block,
                        html=candidate.html or block.html,
                        markdown=candidate.markdown or block.markdown,
                        text=candidate.text or block.text,
                        confidence=max(block.confidence, candidate.confidence),
                        extra=merged_extra,
                    )

                report.append(
                    {
                        "page_no": page.page_no,
                        "table_index": table_ord,
                        "status": "replaced" if replaced else "kept",
                        "old_score": round(old_score, 3),
                        "new_score": round(new_score, 3),
                        "crop_path": str(saved),
                        "engine": engine.name,
                    }
                )

        save_json(refine_dir / "table_refine_report.json", report)
        doc.metadata["table_refine"] = {"enabled": True, "report_path": str(refine_dir / "table_refine_report.json")}
        return doc

    def _build_engine(self):
        if self.config.table_refine.engine == "table_recognition":
            return PaddleTableRecognitionAdapter(device=self.config.device)
        if self.config.table_refine.engine == "paddle_ocr":
            return PaddleOCRAdapter(device=self.config.device, language=self.config.language)
        return PaddleVLAdapter(
            device=self.config.device,
            merge_tables=False,
            relevel_titles=False,
            concatenate_pages=False,
        )

    def _run_table_engine(self, engine, image_path: Path, work_dir: Path) -> OCRBlock | None:
        profile = profile_input(image_path)
        try:
            refined = engine.process(image_path, work_dir, profile)
        except Exception as exc:
            return None
        candidates: list[OCRBlock] = []
        for page in refined.pages:
            for block in page.blocks:
                if block.block_type == "table" and (block.html or block.markdown):
                    candidates.append(block)
        if candidates:
            candidates.sort(key=self._score_table_block, reverse=True)
            return candidates[0]
        return None

    def _estimate_table_bbox(self, page: OCRPage, block_idx: int) -> list[float] | None:
        block = page.blocks[block_idx]
        bbox = list(block.bbox[:4]) if len(block.bbox) >= 4 else []
        if self._bbox_meaningful(page, bbox):
            return bbox

        top = 0.0
        bottom = float(page.height)
        left = 0.0
        right = float(page.width)
        prev_title_text = ""
        next_title_text = ""

        for idx in range(block_idx - 1, -1, -1):
            prev = page.blocks[idx]
            if prev.block_type == "title":
                prev_title_text = prev.text
                if self._bbox_meaningful(page, prev.bbox):
                    top = min(float(page.height), prev.bbox[3] + 8.0)
                else:
                    anchor_y = self._title_anchor_y(page, prev.text)
                    if anchor_y is not None:
                        top = min(float(page.height), anchor_y + 18.0)
                break

        for idx in range(block_idx + 1, len(page.blocks)):
            nxt = page.blocks[idx]
            if nxt.block_type == "title":
                next_title_text = nxt.text
                if self._bbox_meaningful(page, nxt.bbox):
                    bottom = max(top + 20.0, nxt.bbox[1] - 8.0)
                else:
                    anchor_y = self._title_anchor_y(page, nxt.text)
                    if anchor_y is not None:
                        bottom = max(top + 20.0, anchor_y - 12.0)
                break

        if not prev_title_text and not next_title_text:
            if self._pdf_path is not None:
                return None
            area_ratio = 1.0 if (page.width > 0 and page.height > 0) else 0.0
            if area_ratio >= 0.92:
                return None
        if bottom - top < self.config.table_refine.min_crop_height:
            return None
        return [left, top, right, bottom]

    def _bbox_meaningful(self, page: OCRPage, bbox: list[float]) -> bool:
        if len(bbox) < 4:
            return False
        x0, y0, x1, y1 = bbox[:4]
        width = max(0.0, x1 - x0)
        height = max(0.0, y1 - y0)
        if width < self.config.table_refine.min_crop_width or height < self.config.table_refine.min_crop_height:
            return False
        if page.width <= 0 or page.height <= 0:
            return False
        area_ratio = (width * height) / float(page.width * page.height)
        return area_ratio < 0.92

    def _title_anchor_y(self, page: OCRPage, title_text: str) -> float | None:
        if self._pdf_path is None or not title_text:
            return None
        words = self._page_words(page)
        normalized_title = normalize_whitespace(re.sub(r"^(?:\d+\.\s*|#+\s*)", "", title_text))
        if not normalized_title:
            return None
        for _, wy0, _, wy1, token in words:
            normalized_token = normalize_whitespace(token)
            if normalized_token and normalized_title in normalized_token:
                return (wy0 + wy1) / 2.0
        for _, wy0, _, wy1, token in words:
            normalized_token = normalize_whitespace(token)
            if (
                normalized_token
                and len(normalized_token) >= max(4, len(normalized_title) // 2)
                and normalized_token in normalized_title
            ):
                return (wy0 + wy1) / 2.0
        return None

    def _page_words(self, page: OCRPage) -> list[tuple[float, float, float, float, str]]:
        if self._pdf_path is None:
            return []
        key = (page.page_no, page.width, page.height)
        cached = self._pdf_word_cache.get(key)
        if cached is not None:
            return cached
        words = extract_pdf_page_words(self._pdf_path, page.page_no, scale_to=(page.width, page.height))
        self._pdf_word_cache[key] = words
        return words

    def _score_table_block(self, block: OCRBlock) -> float:
        html = block.html or ""
        markdown = block.markdown or ""
        text = normalize_whitespace(block.text)
        row_count = len(HTML_ROW_RE.findall(html))
        cell_count = len(HTML_CELL_RE.findall(html))
        header_hits = len(HEADER_HINT_RE.findall(html or markdown or text))
        non_empty_text = len(re.findall(r"[0-9A-Za-z가-힣]+", normalize_whitespace(re.sub(r"<[^>]+>", " ", html))))
        if not html and markdown:
            non_empty_text += len(re.findall(r"[0-9A-Za-z가-힣]+", markdown))
        score = 0.0
        score += row_count * 2.0
        score += cell_count * 0.6
        score += header_hits * 2.5
        score += non_empty_text * 0.2
        score += min(len(html), 4000) / 400.0
        if self.config.table_refine.prefer_larger_table:
            score += min(len(markdown), 2000) / 500.0

        # Structural integrity bonus: penalise empty cells and reward consistent column count
        struct_score = self._score_structure_integrity(html)
        score += struct_score

        return score

    def _score_structure_integrity(self, html: str) -> float:
        """Reward tables with fewer empty cells and consistent column count."""
        if not html:
            return 0.0
        rows = re.findall(r"<tr\b[^>]*>(.*?)</tr>", html, re.I | re.S)
        if len(rows) < 2:
            return 0.0

        total_cells = 0
        empty_cells = 0
        for row_html in rows:
            cells = re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", row_html, re.I | re.S)
            for cell_html in cells:
                total_cells += 1
                text = re.sub(r"<[^>]+>", " ", cell_html).strip()
                if not text:
                    empty_cells += 1

        if total_cells == 0:
            return 0.0

        # Penalise empty cells
        empty_ratio = empty_cells / total_cells
        empty_penalty = max(0.0, 5.0 * (1.0 - empty_ratio))

        # Reward consistent column count
        col_counts = []
        for row_html in rows:
            cells = re.findall(r"<t[dh]\b[^>]*>", row_html, re.I)
            cs_sum = 0
            for tag in cells:
                m = re.search(r"colspan\s*=\s*[\"']?(\d+)", tag, re.I)
                cs_sum += int(m.group(1)) if m else 1
            col_counts.append(cs_sum)

        if len(set(col_counts)) == 1:
            consistency_bonus = 3.0
        else:
            # Minor penalty for inconsistency but not zero (rowspan/colspan cause this)
            consistency_bonus = max(0.0, 1.0 - len(set(col_counts)))

        return empty_penalty + consistency_bonus

    def _report_item(self, page_no: int, table_index: int, status: str, reason: str) -> dict[str, Any]:
        return {
            "page_no": page_no,
            "table_index": table_index,
            "status": status,
            "reason": reason,
        }
