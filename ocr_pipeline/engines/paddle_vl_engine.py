from __future__ import annotations

import re
import uuid
from html import unescape
from pathlib import Path
from typing import Any

from ..pdf_utils import extract_pdf_page_words
from ..schemas import OCRBlock, OCRDocument, OCRPage, InputProfile
from ..table_rebuilder import extract_best_table_html, html_table_to_markdown, markdown_table_to_html, repair_html_table_spans
from ..utils.io import ensure_dir, load_json, save_json
from ..utils.text import normalize_whitespace
from .base import BaseOCREngine


DEFAULT_BBOX = [0.0, 0.0, 10.0, 10.0]
MD_TABLE_RE = re.compile(r"(?:^|\n)(\|.+\|\n\|[-:| ]+\|\n(?:\|.*\|\n?)*)", re.M)
HTML_TABLE_RE = re.compile(r"(<table\b.*?</table>)", re.I | re.S)
SECTION_TITLE_RE = re.compile(r"^(?:#{1,6}\s+|[■☑]\s*|\d+\.\s*)")
HTML_TR_RE = re.compile(r"<tr\b[^>]*>.*?</tr>", re.I | re.S)
HTML_CELL_RE = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.I | re.S)
HTML_TAG_RE = re.compile(r"<[^>]+>")
HTML_ROW_RE = re.compile(r"(<tr\b[^>]*>)(.*?)(</tr>)", re.I | re.S)
HTML_FIRST_CELL_RE = re.compile(r"^\s*<(t[dh])\b([^>]*)>(.*?)</t[dh]>", re.I | re.S)
NUMERIC_TOKEN_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?%?")
# Parenthesized/circled numbers (①②③…) used as footnote markers, not real data
_CIRCLED_NUM_RE = re.compile(r"[①-㊿②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]")
TABLE_SPLIT_HEADER_RE = re.compile(r"(구분|계획|실적|금액|비율|달성율|진척율|재고회전|영업일수|품목수|수량|합계)", re.I)
FIRST_CELL_SPLIT_RE = re.compile(r"^(구분|재고회전|관세|사업계획)", re.I)
ATTR_SPAN_RE = re.compile(r"""\b(?P<name>rowspan|colspan)\s*=\s*["']?(?P<value>\d+)["']?""", re.I)


class PaddleVLAdapter(BaseOCREngine):
    name = "paddle_vl"

    def __init__(
        self,
        *,
        device: str = "gpu",
        merge_tables: bool = True,
        relevel_titles: bool = True,
        concatenate_pages: bool = False,
        lora_path: str | None = None,
    ):
        self.device = device
        self.merge_tables = merge_tables
        self.relevel_titles = relevel_titles
        self.concatenate_pages = concatenate_pages
        self.lora_path = lora_path

    def available(self) -> bool:
        try:
            import paddle  # type: ignore
            from paddleocr import PaddleOCRVL  # type: ignore
            _ = paddle
            _ = PaddleOCRVL
            return True
        except Exception:
            return False

    def process(self, input_path: str | Path, work_dir: str | Path, profile: InputProfile) -> OCRDocument:
        try:
            import paddle  # type: ignore
            from paddleocr import PaddleOCRVL  # type: ignore
            _ = paddle
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PaddleOCR-VL 실행에 필요한 `paddle` 모듈이 없습니다. "
                "공식 문서에 따라 `paddlepaddle` 또는 `paddlepaddle-gpu`를 먼저 설치한 뒤 "
                "`paddleocr[doc-parser]`를 설치하세요."
            ) from exc

        input_path = Path(input_path)
        work_dir = ensure_dir(work_dir)
        raw_dir = ensure_dir(work_dir / "raw_paddle_vl")

        # Check if LoRA adapter should be applied via transformers
        if self.lora_path:
            import warnings
            warnings.warn(
                "LoRA inference via transformers is not yet integrated with PaddleOCR-VL pipeline. "
                "The PaddleOCR-VL pipeline runs with PaddlePaddle runtime; LoRA adapters require "
                "a separate transformers-based inference path. Falling back to base model.",
                stacklevel=2,
            )
            self.lora_path = None

        pipeline = PaddleOCRVL(device=self.device, format_block_content=True)
        output = pipeline.predict(input=str(input_path))
        pages_res = list(output)
        if input_path.suffix.lower() == ".pdf":
            pages_res = list(
                pipeline.restructure_pages(
                    pages_res,
                    merge_tables=self.merge_tables,
                    relevel_titles=self.relevel_titles,
                    concatenate_pages=self.concatenate_pages,
                )
            )

        pages: list[OCRPage] = []
        for idx, res in enumerate(pages_res, start=1):
            page_dir = ensure_dir(raw_dir / f"page_{idx:04d}")
            try:
                res.save_to_json(save_path=str(page_dir))
            except TypeError:
                res.save_to_json(str(page_dir))
            try:
                res.save_to_markdown(save_path=str(page_dir))
            except Exception:
                pass

            json_file = self._find_single_file(page_dir, ".json")
            md_file = self._find_single_file(page_dir, ".md")
            raw_obj = load_json(json_file) if json_file and json_file.exists() else {}
            markdown = md_file.read_text(encoding="utf-8") if md_file and md_file.exists() else ""
            page = self._parse_page(raw_obj, markdown, idx, profile)
            if input_path.suffix.lower() == ".pdf" and profile.has_text_layer:
                try:
                    pdf_words = extract_pdf_page_words(input_path, idx, scale_to=(page.width, page.height))
                    page.blocks = self._repair_table_blocks_from_pdf_text(page.blocks, pdf_words)
                except Exception:
                    pass
            page.raw_paths = {
                "json": str(json_file) if json_file else "",
                "markdown": str(md_file) if md_file else "",
            }
            pages.append(page)

        doc = OCRDocument(
            document_id=str(uuid.uuid4()),
            source_file=input_path.name,
            pdf_type=profile.pdf_type,
            page_count=len(pages),
            pages=pages,
            engine_used=self.name,
            metadata={"profile": profile.to_dict()},
        )
        save_json(work_dir / "document_profile.json", profile.to_dict())
        return doc

    def _find_single_file(self, directory: Path, suffix: str) -> Path | None:
        files = sorted(directory.glob(f"*{suffix}"))
        return files[0] if files else None

    def _parse_page(self, raw_obj: dict[str, Any], markdown: str, page_no: int, profile: InputProfile) -> OCRPage:
        width, height = self._extract_page_size(raw_obj)
        source_image = self._extract_source_image(raw_obj)
        page = OCRPage(page_no=page_no, width=width, height=height, source_image=source_image)

        blocks = self._extract_blocks(raw_obj, page_no)
        blocks = self._split_merged_table_blocks(blocks)
        blocks = self._fill_table_markdown(blocks)
        blocks = self._repair_html_table_structure(blocks)
        blocks = self._merge_or_replace_with_markdown(blocks, markdown, page_no, width, height)
        page.blocks = blocks
        page.extra["page_markdown"] = markdown
        page.extra["raw_top_level_keys"] = sorted(list(raw_obj.keys())) if isinstance(raw_obj, dict) else []
        return page

    def _extract_page_size(self, raw_obj: dict[str, Any]) -> tuple[int, int]:
        width = int(raw_obj.get("width", raw_obj.get("img_w", 0)) or 0)
        height = int(raw_obj.get("height", raw_obj.get("img_h", 0)) or 0)
        if not width or not height:
            image_info = raw_obj.get("input_img", {}) if isinstance(raw_obj, dict) else {}
            width = int(image_info.get("width", width) or width or 1)
            height = int(image_info.get("height", height) or height or 1)
        return width or 1, height or 1

    def _extract_source_image(self, raw_obj: dict[str, Any]) -> str:
        for key in ("input_path", "img_path", "source", "page_image"):
            value = raw_obj.get(key)
            if isinstance(value, str) and value:
                return value
        return ""

    def _extract_blocks(self, raw_obj: Any, page_no: int) -> list[OCRBlock]:
        blocks: list[OCRBlock] = []
        counter = 1

        def walk(obj: Any) -> None:
            nonlocal counter
            if isinstance(obj, dict):
                block_type = str(obj.get("block_type") or obj.get("block_label") or obj.get("type") or obj.get("label") or obj.get("category") or "").lower()
                text = self._extract_text(obj)
                bbox = self._normalize_bbox(obj.get("bbox") or obj.get("box") or obj.get("dt_polys") or obj.get("coordinate") or obj.get("block_bbox"))
                html = self._extract_local_html(obj)
                markdown = self._extract_markdown(obj)
                conf = self._extract_confidence(obj)
                has_content = bool(text or html or markdown)
                text_html_table = self._extract_table_html_from_text(text)
                is_table = (
                    block_type in {"table", "tab", "table_body"}
                    or bool(html)
                    or bool(markdown_table_to_html(markdown))
                    or bool(text_html_table)
                )
                bbox_meaningful = bbox != DEFAULT_BBOX
                interesting = has_content and (block_type or bbox_meaningful)
                if interesting:
                    if is_table:
                        final_type = "table"
                        if not html and text_html_table:
                            html = text_html_table
                        if not html and markdown:
                            html = markdown_table_to_html(markdown)
                    elif block_type in {"title", "header", "heading", "doc_title"} or self._looks_like_title(text):
                        final_type = "title"
                    elif block_type in {"formula", "equation"}:
                        final_type = "formula"
                    elif block_type in {"figure", "image", "chart", "header_image", "image_caption"}:
                        final_type = "figure"
                    else:
                        final_type = "text"
                    blocks.append(
                        OCRBlock(
                            block_id=f"p{page_no}_b{counter}",
                            block_type=final_type,
                            page_no=page_no,
                            bbox=bbox,
                            text=text if final_type != "table" else "",
                            html=html,
                            markdown=markdown,
                            confidence=conf,
                            source_engine=self.name,
                            extra={"raw_type": block_type},
                        )
                    )
                    counter += 1
                for value in obj.values():
                    walk(value)
            elif isinstance(obj, list):
                for item in obj:
                    walk(item)

        walk(raw_obj)
        dedup: list[OCRBlock] = []
        seen: set[tuple[str, str, str, tuple[int, ...]]] = set()
        for block in blocks:
            sig = (
                block.block_type,
                normalize_whitespace(block.text)[:200],
                normalize_whitespace(block.markdown)[:200],
                tuple(int(v) for v in block.bbox[:4]),
            )
            if sig in seen:
                continue
            seen.add(sig)
            dedup.append(block)
        return dedup

    def _extract_text(self, obj: dict[str, Any]) -> str:
        text_candidates: list[str] = []
        for key in (
            "text",
            "content",
            "block_content",
            "text_content",
            "rec_text",
            "ocr_text",
            "title",
            "plain_text",
            "markdown",
        ):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                text_candidates.append(value)
        if not text_candidates:
            texts = obj.get("texts") or obj.get("text_list") or obj.get("text_lines")
            if isinstance(texts, list):
                joined = "\n".join(str(x) for x in texts if str(x).strip())
                if joined.strip():
                    text_candidates.append(joined)
        if not text_candidates:
            return ""
        best = max(text_candidates, key=lambda s: len(normalize_whitespace(s)))
        if "|" in best and "\n" in best and markdown_table_to_html(best):
            return ""
        if "<table" in best.lower():
            return ""
        return normalize_whitespace(best)

    def _extract_local_html(self, obj: dict[str, Any]) -> str:
        for key in ("html", "table_html", "res_html", "block_content"):
            value = obj.get(key)
            if isinstance(value, str) and "<table" in value:
                return value
        return ""

    def _extract_markdown(self, obj: dict[str, Any]) -> str:
        for key in ("markdown", "md", "content_markdown"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _extract_confidence(self, obj: dict[str, Any]) -> float:
        for key in ("score", "confidence", "rec_score", "prob"):
            value = obj.get(key)
            if isinstance(value, (int, float)):
                return max(0.0, min(1.0, float(value)))
        return 0.80

    def _normalize_bbox(self, bbox: Any) -> list[float]:
        if isinstance(bbox, (list, tuple)):
            if len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
                vals = [float(v) for v in bbox]
                if vals[2] > vals[0] and vals[3] > vals[1]:
                    return vals
            if bbox and isinstance(bbox[0], (list, tuple)):
                xs = [float(pt[0]) for pt in bbox if len(pt) >= 2]
                ys = [float(pt[1]) for pt in bbox if len(pt) >= 2]
                if xs and ys and max(xs) > min(xs) and max(ys) > min(ys):
                    return [min(xs), min(ys), max(xs), max(ys)]
        return DEFAULT_BBOX.copy()

    def _merge_or_replace_with_markdown(self, blocks: list[OCRBlock], markdown: str, page_no: int, width: int, height: int) -> list[OCRBlock]:
        markdown = (markdown or "").strip()
        if not markdown:
            return blocks

        page_md_blocks = self._parse_markdown_fallback(markdown, page_no, width, height)
        usable_blocks = [b for b in blocks if self._has_meaningful_content(b)]
        if not usable_blocks:
            return page_md_blocks or blocks

        if page_md_blocks:
            json_has_table_with_bbox = any(
                b.block_type == "table" and (b.html or b.markdown) and b.bbox != DEFAULT_BBOX
                for b in usable_blocks
            )
            has_table = any(b.block_type == "table" and (b.html or b.markdown) for b in usable_blocks)
            if json_has_table_with_bbox:
                pass
            elif not has_table:
                usable_blocks.extend([b for b in page_md_blocks if b.block_type == "table"])
            has_rich_text = any(b.block_type in {"text", "title"} and (b.text or b.markdown) for b in usable_blocks)
            has_title = any(b.block_type == "title" and (b.text or b.markdown) for b in usable_blocks)
            if not has_rich_text:
                usable_blocks.extend([b for b in page_md_blocks if b.block_type in {"text", "title"}])
            elif not has_title:
                usable_blocks.extend([b for b in page_md_blocks if b.block_type == "title"])
        return usable_blocks

    def _parse_markdown_fallback(self, markdown: str, page_no: int, width: int, height: int) -> list[OCRBlock]:
        blocks: list[OCRBlock] = []
        if markdown:
            token_re = re.compile(r"(<table\b.*?</table>)|((?:^|\n)(\|.+\|\n\|[-:| ]+\|\n(?:\|.*\|\n?)*))", re.I | re.S | re.M)
            counter = 1
            cursor = 0
            for match in token_re.finditer(markdown):
                before = markdown[cursor:match.start()]
                blocks.extend(self._markdown_text_blocks(before, page_no, width, height, counter))
                counter = len(blocks) + 1
                html_table = (match.group(1) or "").strip()
                md_table = (match.group(2) or "").strip()
                if html_table:
                    blocks.append(
                        OCRBlock(
                            block_id=f"p{page_no}_md{counter}",
                            block_type="table",
                            page_no=page_no,
                            bbox=[0.0, 0.0, float(width), float(height)],
                            markdown="",
                            html=html_table,
                            confidence=0.88,
                            source_engine=f"{self.name}_markdown_fallback",
                            extra={"fallback": "page_html_table"},
                        )
                    )
                elif md_table:
                    blocks.append(
                        OCRBlock(
                            block_id=f"p{page_no}_md{counter}",
                            block_type="table",
                            page_no=page_no,
                            bbox=[0.0, 0.0, float(width), float(height)],
                            markdown=md_table,
                            html=markdown_table_to_html(md_table),
                            confidence=0.88,
                            source_engine=f"{self.name}_markdown_fallback",
                            extra={"fallback": "page_markdown_table"},
                        )
                    )
                counter = len(blocks) + 1
                cursor = match.end()
            rest = markdown[cursor:]
            blocks.extend(self._markdown_text_blocks(rest, page_no, width, height, counter))
        blocks = self._split_merged_table_blocks(blocks)

        if not blocks and normalize_whitespace(markdown):
            blocks.append(
                OCRBlock(
                    block_id=f"p{page_no}_md1",
                    block_type="text",
                    page_no=page_no,
                    bbox=[0.0, 0.0, float(width), float(height)],
                    text=normalize_whitespace(markdown),
                    markdown=markdown,
                    confidence=0.82,
                    source_engine=f"{self.name}_markdown_fallback",
                    extra={"fallback": "page_markdown_full"},
                )
            )
        return blocks

    def _markdown_text_blocks(self, text: str, page_no: int, width: int, height: int, start_idx: int) -> list[OCRBlock]:
        pieces: list[OCRBlock] = []
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text or "") if normalize_whitespace(p)]
        idx = start_idx
        for para_idx, para in enumerate(paragraphs):
            raw = para.strip()
            normalized_lines = [line.strip() for line in raw.splitlines() if line.strip()]
            prev_para = paragraphs[para_idx - 1] if para_idx > 0 else ""
            next_para = paragraphs[para_idx + 1] if para_idx + 1 < len(paragraphs) else ""
            if len(normalized_lines) == 1 and self._looks_like_contextual_title(normalized_lines[0], prev_para=prev_para, next_para=next_para):
                block_type = "title"
            else:
                block_type = "title" if raw.startswith("#") else "text"
            clean = normalize_whitespace(re.sub(r"^(?:#{1,6}\s*|[■☑]\s*|\d+\.\s*)", "", raw))
            if not clean:
                continue
            pieces.append(
                OCRBlock(
                    block_id=f"p{page_no}_md{idx}",
                    block_type=block_type,
                    page_no=page_no,
                    bbox=[0.0, 0.0, float(width), float(height)],
                    text=clean,
                    markdown=raw,
                    confidence=0.82,
                    source_engine=f"{self.name}_markdown_fallback",
                    extra={"fallback": "page_markdown_text"},
                )
            )
            idx += 1
        return pieces

    def _fill_table_markdown(self, blocks: list[OCRBlock]) -> list[OCRBlock]:
        result: list[OCRBlock] = []
        for block in blocks:
            if block.block_type == "table" and block.html and not block.markdown:
                md = html_table_to_markdown(block.html)
                if md:
                    block = OCRBlock(
                        block_id=block.block_id,
                        block_type=block.block_type,
                        page_no=block.page_no,
                        bbox=block.bbox,
                        text=block.text,
                        html=block.html,
                        markdown=md,
                        confidence=block.confidence,
                        review=block.review,
                        review_reason=list(block.review_reason),
                        source_engine=block.source_engine,
                        extra={**block.extra, "md_from_html": True},
                    )
            result.append(block)
        return result

    def _repair_html_table_structure(self, blocks: list[OCRBlock]) -> list[OCRBlock]:
        """Apply structural repairs to table HTML (rowspan fixes, empty cell cleanup)."""
        result: list[OCRBlock] = []
        for block in blocks:
            if block.block_type == "table" and block.html:
                repaired = repair_html_table_spans(block.html)
                if repaired != block.html:
                    block = OCRBlock(
                        block_id=block.block_id,
                        block_type=block.block_type,
                        page_no=block.page_no,
                        bbox=block.bbox,
                        text=block.text,
                        html=repaired,
                        markdown=block.markdown,
                        confidence=block.confidence,
                        review=block.review,
                        review_reason=list(block.review_reason),
                        source_engine=block.source_engine,
                        extra={**block.extra, "structure_repaired": True},
                    )
            result.append(block)
        return result

    def _has_meaningful_content(self, block: OCRBlock) -> bool:
        if block.block_type == "table":
            return bool(block.html or block.markdown)
        return bool(block.text or block.markdown or block.html)

    def _split_merged_table_blocks(self, blocks: list[OCRBlock]) -> list[OCRBlock]:
        expanded: list[OCRBlock] = []
        for block in blocks:
            if block.block_type != "table" or not block.html:
                expanded.append(block)
                continue
            split_tables = self._split_html_table(block.html)
            if len(split_tables) <= 1:
                expanded.append(block)
                continue
            for idx, html in enumerate(split_tables, start=1):
                md = html_table_to_markdown(html)
                expanded.append(
                    OCRBlock(
                        block_id=f"{block.block_id}_s{idx}",
                        block_type="table",
                        page_no=block.page_no,
                        bbox=block.bbox,
                        html=html,
                        markdown=md,
                        confidence=block.confidence,
                        review=block.review,
                        review_reason=list(block.review_reason),
                        source_engine=block.source_engine,
                        extra={**block.extra, "split_from": block.block_id, "split_index": idx, "split_count": len(split_tables)},
                    )
                )
        return expanded

    def _split_html_table(self, html: str) -> list[str]:
        table_match = re.search(r"(<table\b[^>]*>)(.*?)(</table>)", html, re.I | re.S)
        if not table_match:
            return [html]
        open_tag, inner_html, close_tag = table_match.groups()
        row_htmls = HTML_TR_RE.findall(inner_html)
        if len(row_htmls) < 6:
            return [html]

        row_cells = [self._html_row_cells(row_html) for row_html in row_htmls]
        split_points: list[int] = []
        for idx in range(3, len(row_cells)):
            if self._is_split_header_row(row_cells, idx):
                split_points.append(idx)
        if not split_points:
            return [html]

        segments: list[str] = []
        start = 0
        for split_idx in split_points + [len(row_htmls)]:
            segment_rows = row_htmls[start:split_idx]
            if segment_rows:
                segments.append(f"{open_tag}{''.join(segment_rows)}{close_tag}")
            start = split_idx
        return segments or [html]

    def _html_row_cells(self, row_html: str) -> list[str]:
        cells: list[str] = []
        for cell_html in HTML_CELL_RE.findall(row_html):
            plain = unescape(HTML_TAG_RE.sub(" ", cell_html))
            plain = normalize_whitespace(plain)
            if plain:
                cells.append(plain)
        return cells

    @staticmethod
    def _count_data_numeric(text: str) -> int:
        """Count numeric tokens excluding circled/parenthesized footnote numbers."""
        cleaned = _CIRCLED_NUM_RE.sub("", text)
        return len(NUMERIC_TOKEN_RE.findall(cleaned))

    def _is_split_header_row(self, rows: list[list[str]], idx: int) -> bool:
        current = rows[idx]
        prev = rows[idx - 1] if idx - 1 >= 0 else []
        if not current or not prev:
            return False
        current_text = " ".join(current)
        current_numeric = self._count_data_numeric(current_text)
        prev_numeric = len(NUMERIC_TOKEN_RE.findall(" ".join(prev)))
        first_cell = current[0] if current else ""
        if first_cell in {"합계", "계", "소계"}:
            return False
        current_headerish = bool(TABLE_SPLIT_HEADER_RE.search(current_text)) and current_numeric <= 1
        prev_dataish = prev_numeric >= 2
        column_shift = abs(len(current) - len(prev)) >= 3
        repeated_header = bool(FIRST_CELL_SPLIT_RE.search(first_cell))
        # Path 1: current looks like a header row, prev has data, and columns shift or header restarts
        if current_headerish and prev_dataish and (column_shift or repeated_header):
            return True
        # Path 2: strong repeated header keyword + column shift (covers footnote-laden headers)
        if repeated_header and column_shift:
            return True
        return False

    def _extract_table_html_from_text(self, text: str) -> str:
        if not text:
            return ""
        match = HTML_TABLE_RE.search(text)
        return match.group(1).strip() if match else ""

    def _repair_table_blocks_from_pdf_text(
        self,
        blocks: list[OCRBlock],
        pdf_words: list[tuple[float, float, float, float, str]],
    ) -> list[OCRBlock]:
        repaired: list[OCRBlock] = []
        prev_title = ""
        for block in blocks:
            if block.block_type == "title" and block.text:
                prev_title = block.text
            if block.block_type == "table" and block.html and pdf_words:
                # Phase 1: existing top-left header repair
                html = self._repair_top_left_header_from_pdf(block.html, block.bbox, pdf_words, prev_title)
                # Phase 2: rowspan conflict detection using PDF word positions
                if block.bbox and len(block.bbox) >= 4:
                    html = self._repair_rowspan_conflicts_from_pdf(html, block.bbox, pdf_words)
                if html != block.html:
                    block = OCRBlock(
                        block_id=block.block_id,
                        block_type=block.block_type,
                        page_no=block.page_no,
                        bbox=block.bbox,
                        text=block.text,
                        html=html,
                        markdown=block.markdown,
                        confidence=block.confidence,
                        review=block.review,
                        review_reason=list(block.review_reason),
                        source_engine=block.source_engine,
                        extra={**block.extra, "pdf_text_repaired": True},
                    )
            repaired.append(block)
        return repaired

    def _repair_top_left_header_from_pdf(
        self,
        html: str,
        bbox: list[float],
        pdf_words: list[tuple[float, float, float, float, str]],
        context_title: str = "",
    ) -> str:
        rows = HTML_ROW_RE.findall(html)
        if len(rows) < 2:
            return html
        first_open, first_inner, first_close = rows[0]
        original_first_row = "".join(rows[0])
        first_cell_match = HTML_FIRST_CELL_RE.match(first_inner)
        if not first_cell_match:
            return html

        tag, attrs, inner = first_cell_match.groups()
        cell_text = normalize_whitespace(unescape(HTML_TAG_RE.sub(" ", inner)))
        if not cell_text:
            return html

        spans = {m.group("name").lower(): int(m.group("value")) for m in ATTR_SPAN_RE.finditer(attrs or "")}
        rowspan = spans.get("rowspan", 1)
        colspan = spans.get("colspan", 1)
        if rowspan < 2 or colspan < 2:
            return html

        candidate_words = self._table_top_left_words(bbox, pdf_words, context_title=context_title)
        if len(candidate_words) < 2:
            return html
        if candidate_words[-1] != cell_text or candidate_words[0] == cell_text:
            return html

        left_header = candidate_words[0]
        second_attrs = self._replace_or_drop_span(attrs, "colspan", colspan - 1)
        second_attrs = self._replace_or_drop_span(second_attrs, "rowspan", rowspan)
        second_cell = f"<{tag}{second_attrs}>{inner}</{tag}>"
        first_cell = f"<{tag} rowspan=\"{rowspan}\">{left_header}</{tag}>"
        rebuilt_inner = first_cell + second_cell + first_inner[first_cell_match.end():]
        rebuilt_first_row = f"{first_open}{rebuilt_inner}{first_close}"
        return html.replace(original_first_row, rebuilt_first_row, 1)

    def _table_top_left_words(
        self,
        bbox: list[float],
        pdf_words: list[tuple[float, float, float, float, str]],
        *,
        context_title: str = "",
    ) -> list[str]:
        if len(bbox) < 4:
            return []
        x0, y0, x1, y1 = bbox[:4]
        width = max(1.0, x1 - x0)
        height = max(1.0, y1 - y0)
        page_x1 = max((wx1 for _, _, wx1, _, _ in pdf_words), default=x1)
        page_y1 = max((wy1 for _, _, _, wy1, _ in pdf_words), default=y1)
        use_title_anchor = (
            bool(context_title)
            and width >= page_x1 * 0.8
            and height >= page_y1 * 0.8
        )

        if use_title_anchor:
            title_y = self._find_title_anchor_y(context_title, pdf_words)
            if title_y is not None:
                x0 = 0.0
                y0 = title_y + 8.0
                region_x1 = min(page_x1, 420.0)
                region_y1 = min(page_y1, y0 + 110.0)
            else:
                region_x1 = x0 + min(width * 0.15, 220.0)
                region_y1 = y0 + min(height * 0.18, 120.0)
        else:
            region_x1 = x0 + min(width * 0.15, 220.0)
            region_y1 = y0 + min(height * 0.18, 120.0)

        candidates: list[tuple[float, float, str]] = []
        for wx0, wy0, wx1, wy1, text in pdf_words:
            cx = (wx0 + wx1) / 2.0
            cy = (wy0 + wy1) / 2.0
            if x0 <= cx <= region_x1 and y0 <= cy <= region_y1:
                token = normalize_whitespace(text)
                if token:
                    candidates.append((cy, cx, token))

        candidates.sort(key=lambda item: (item[1], item[0]))
        seen: set[str] = set()
        ordered: list[str] = []
        for _, _, token in candidates:
            if token in seen:
                continue
            seen.add(token)
            ordered.append(token)
        return ordered

    def _find_title_anchor_y(
        self,
        context_title: str,
        pdf_words: list[tuple[float, float, float, float, str]],
    ) -> float | None:
        normalized_title = normalize_whitespace(re.sub(r"^(?:\d+\.\s*|#+\s*)", "", context_title))
        if not normalized_title:
            return None
        for _, wy0, _, wy1, token in pdf_words:
            normalized_token = normalize_whitespace(token)
            if normalized_token and normalized_title in normalized_token:
                return (wy0 + wy1) / 2.0
        for _, wy0, _, wy1, token in pdf_words:
            normalized_token = normalize_whitespace(token)
            if (
                normalized_token
                and len(normalized_token) >= max(4, len(normalized_title) // 2)
                and normalized_token in normalized_title
            ):
                return (wy0 + wy1) / 2.0
        return None

    def _replace_or_drop_span(self, attrs: str, name: str, value: int) -> str:
        if value <= 1:
            return re.sub(rf"""\s*\b{name}\s*=\s*["']?\d+["']?""", "", attrs, flags=re.I)
        if re.search(rf"""\b{name}\s*=""", attrs, flags=re.I):
            return re.sub(rf"""(\b{name}\s*=\s*["']?)\d+(["']?)""", rf"\g<1>{value}\2", attrs, flags=re.I)
        return f'{attrs} {name}="{value}"'

    def _repair_rowspan_conflicts_from_pdf(
        self,
        html: str,
        bbox: list[float],
        pdf_words: list[tuple[float, float, float, float, str]],
    ) -> str:
        """Detect rowspan conflicts using PDF text layer word positions.

        Strategy: For each cell with rowspan > 1 in row 0, check if the PDF
        text layer has words in the spanned rows that fall within the same
        column region. If so, the rowspan is too large and should be reduced.
        """
        rows = HTML_ROW_RE.findall(html)
        if len(rows) < 2:
            return html

        if len(bbox) < 4:
            return html
        x0, y0, x1, y1 = bbox[:4]
        table_width = max(1.0, x1 - x0)
        table_height = max(1.0, y1 - y0)

        # Get PDF words in table area
        table_words = [
            (wx0, wy0, wx1, wy1, token)
            for wx0, wy0, wx1, wy1, token in pdf_words
            if x0 <= wx0 and wx1 <= x1 and y0 <= wy0 and wy1 <= y1
        ]
        if not table_words:
            return html

        # Parse first row cells to find rowspan > 1 cells
        first_row_inner = rows[0][1] if len(rows[0]) >= 3 else ""
        first_row_open = rows[0][0] if len(rows[0]) >= 3 else "<tr>"
        first_row_close = rows[0][2] if len(rows[0]) >= 3 else "</tr>"

        modified = False
        result_html = html

        for cell_match in HTML_CELL_RE.finditer(first_row_inner):
            cell_full = cell_match.group(0)
            cell_text = normalize_whitespace(unescape(HTML_TAG_RE.sub(" ", cell_match.group(1))))
            if not cell_text:
                continue

            spans = {m.group("name").lower(): int(m.group("value")) for m in ATTR_SPAN_RE.finditer(cell_full)}
            rs = spans.get("rowspan", 1)
            cs = spans.get("colspan", 1)
            if rs <= 1:
                continue

            # Find the x-position of this cell in the first row
            cell_x_start = self._cell_x_position(first_row_inner, cell_match.start())
            if cell_x_start is None:
                continue
            cell_col_width = table_width / max(1, self._count_first_row_cols(first_row_inner))
            cell_x_end = cell_x_start + cell_col_width * cs

            # Map cell position to PDF coordinates
            pdf_cell_x0 = x0 + cell_x_start
            pdf_cell_x1 = min(x1, x0 + cell_x_end)

            # Check if any PDF word in the next row (row 1) falls within this cell's x range
            if len(rows) < 2:
                continue
            second_row_inner = rows[1][1] if len(rows[1]) >= 3 else ""
            second_row_y_range = self._estimate_row_y_range(y0, table_height, len(rows), 1)

            conflicting = False
            for wx0, wy0, wx1, wy1, token in table_words:
                token_norm = normalize_whitespace(token)
                if not token_norm:
                    continue
                # Check if word is in row 1 area and within this cell's x range
                word_cx = (wx0 + wx1) / 2.0
                word_cy = (wy0 + wy1) / 2.0
                if second_row_y_range and second_row_y_range[0] <= word_cy <= second_row_y_range[1]:
                    if pdf_cell_x0 <= word_cx <= pdf_cell_x1:
                        # There's a word in the spanned area — rowspan conflicts
                        conflicting = True
                        break

            if conflicting and rs > 1:
                new_rs = rs - 1
                new_attrs = self._replace_or_drop_span(
                    cell_match.group(0)[:cell_match.start() - cell_match.start()],
                    "rowspan", new_rs,
                )
                # Rebuild the cell tag
                tag_match = re.match(r"<(t[dh])(\b[^>]*)>", cell_full, re.I)
                if tag_match:
                    tag = tag_match.group(1)
                    old_attrs = tag_match.group(2)
                    new_attrs = self._replace_or_drop_span(old_attrs, "rowspan", new_rs)
                    new_cell = f"<{tag}{new_attrs}>{cell_match.group(1)}</{tag}>"
                    result_html = result_html.replace(cell_full, new_cell, 1)
                    modified = True

        return result_html

    def _cell_x_position(self, row_inner: str, cell_start: int) -> float | None:
        """Estimate the x-position (0-based fraction) of a cell within a row."""
        cells_before = list(HTML_CELL_RE.finditer(row_inner, 0, cell_start))
        if not cells_before and cell_start == 0:
            return 0.0
        # Approximate: count colspan of cells before this one
        total_before = 0
        for m in cells_before:
            spans = {k.lower(): int(v) for k, v in ATTR_SPAN_RE.finditer(m.group(0))}
            total_before += spans.get("colspan", 1)
        return float(total_before) if cells_before else None

    def _count_first_row_cols(self, row_inner: str) -> int:
        """Count total columns in a row (sum of colspan)."""
        total = 0
        for m in HTML_CELL_RE.finditer(row_inner):
            spans = {k.lower(): int(v) for k, v in ATTR_SPAN_RE.finditer(m.group(0))}
            total += spans.get("colspan", 1)
        return max(1, total)

    def _estimate_row_y_range(self, table_y0: float, table_height: float, num_rows: int, row_idx: int) -> tuple[float, float] | None:
        """Estimate the y-range of a row within the table bbox."""
        if num_rows <= 0 or row_idx >= num_rows:
            return None
        row_height = table_height / num_rows
        y_start = table_y0 + row_idx * row_height
        y_end = table_y0 + (row_idx + 1) * row_height
        return (y_start, y_end)

    def _looks_like_title(self, text: str) -> bool:
        candidate = normalize_whitespace(text)
        if not candidate:
            return False
        return bool(SECTION_TITLE_RE.match(candidate))

    def _looks_like_contextual_title(self, text: str, *, prev_para: str = "", next_para: str = "") -> bool:
        candidate = normalize_whitespace(text)
        if not candidate:
            return False
        if self._looks_like_title(candidate):
            return True
        if len(candidate) > 40:
            return False
        if "\n" in candidate:
            return False

        prev_norm = normalize_whitespace(prev_para)
        next_norm = normalize_whitespace(next_para)
        next_has_table = "<table" in (next_para or "").lower() or "|" in next_para
        prev_has_table = "<table" in (prev_para or "").lower() or "|" in prev_para
        numeric_count = len(NUMERIC_TOKEN_RE.findall(candidate))

        # Treat short standalone lines between tables/sections as titles only when context supports it.
        if next_has_table and numeric_count <= 1:
            return True
        if prev_has_table and next_has_table and numeric_count <= 1:
            return True
        if candidate.endswith(":") and next_has_table:
            return True
        if prev_norm == "" and next_has_table and numeric_count <= 1:
            return True
        return False
