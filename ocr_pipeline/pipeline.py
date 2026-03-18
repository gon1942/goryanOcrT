from __future__ import annotations

from pathlib import Path

from .config import PipelineConfig
from .compare import build_document_comparison, render_comparison_html
from .engines.paddle_ocr_engine import PaddleOCRAdapter
from .engines.paddle_vl_engine import PaddleVLAdapter
from .engines.surya_engine import SuryaAdapter
from .html_renderer import render_document_html
from .preprocessing import preprocess_image
from .profile import profile_input
from .quality import score_document
from .review_overlay import draw_overlay
from .schemas import OCRDocument
from .utils.io import ensure_dir, save_json, save_text


class OCRPipeline:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()

    def run(self, input_path: str | Path, output_dir: str | Path) -> OCRDocument:
        input_path = Path(input_path)
        output_dir = ensure_dir(output_dir)
        profile = profile_input(input_path)
        save_json(output_dir / "input_profile.json", profile.to_dict())

        effective_input = input_path
        if input_path.suffix.lower() != ".pdf" and self.config.preprocess.enabled:
            pre_dir = ensure_dir(output_dir / "preprocessed")
            effective_input = preprocess_image(
                input_path,
                pre_dir / input_path.name,
                deskew=self.config.preprocess.deskew,
                denoise=self.config.preprocess.denoise,
                adaptive_threshold=self.config.preprocess.adaptive_threshold,
                strengthen_table_lines=self.config.preprocess.strengthen_table_lines,
                line_kernel_scale=self.config.preprocess.line_kernel_scale,
            )

        engine = self._select_engine(profile)
        work_dir = ensure_dir(output_dir / engine.name)
        doc = engine.process(effective_input, work_dir, profile)
        doc = score_document(doc, self.config.review)

        save_json(output_dir / "result.pretty.json", doc.to_dict())
        save_text(output_dir / "result.md", self._render_markdown(doc))
        if self.config.output.save_html:
            save_text(output_dir / "result.html", render_document_html(doc))
        if self.config.output.save_comparison:
            comparison = build_document_comparison(input_path, doc, output_dir)
            save_json(output_dir / "comparison" / "result.compare.json", comparison.to_dict())
            save_text(output_dir / "comparison" / "result.compare.html", render_comparison_html(comparison, doc))
        if self.config.output.save_overlay:
            overlay_dir = ensure_dir(output_dir / "review_overlays")
            for page in doc.pages:
                if page.source_image:
                    try:
                        draw_overlay(page, overlay_dir / f"page_{page.page_no:04d}.png")
                    except Exception:
                        pass
        return doc

    def _select_engine(self, profile):
        paddle_vl = PaddleVLAdapter(
            device=self.config.device,
            merge_tables=self.config.merge_cross_page_tables,
            relevel_titles=self.config.relevel_titles,
            concatenate_pages=self.config.concatenate_pages,
        )
        paddle_ocr = PaddleOCRAdapter(device=self.config.device, language=self.config.language)
        surya = SuryaAdapter()
        requested = self.config.engine
        if requested == "paddle_vl":
            if not paddle_vl.available():
                raise RuntimeError("PaddleOCRVL을 사용할 수 없습니다. `paddlepaddle` 또는 `paddlepaddle-gpu`와 `paddleocr[doc-parser]` 설치를 확인하세요.")
            return paddle_vl
        if requested == "paddle_ocr":
            if not paddle_ocr.available():
                raise RuntimeError("PaddleOCR를 사용할 수 없습니다. `paddlepaddle` 또는 `paddlepaddle-gpu`와 `paddleocr[all]` 설치를 확인하세요.")
            return paddle_ocr
        if requested == "surya":
            if not surya.available():
                raise RuntimeError("Surya를 사용할 수 없습니다.")
            return surya

        # auto routing
        if profile.routing_hint == "prefer_paddle_vl" and paddle_vl.available():
            return paddle_vl
        if profile.routing_hint == "prefer_paddle_ocr" and paddle_ocr.available():
            return paddle_ocr
        if paddle_vl.available() and (profile.layout_type in {"table_heavy", "mixed"} or profile.file_type == "pdf" or profile.pdf_type in {"image", "unknown"}):
            return paddle_vl
        if paddle_ocr.available():
            return paddle_ocr
        if surya.available():
            return surya
        raise RuntimeError("사용 가능한 OCR 엔진이 없습니다. requirements와 설치 상태를 확인하세요.")

    def _render_markdown(self, doc: OCRDocument) -> str:
        lines = [f"# {doc.source_file}", ""]
        for page in doc.pages:
            lines.append(f"## Page {page.page_no}")
            lines.append("")
            if page.review_reason:
                lines.append(f"> review: {', '.join(page.review_reason)}")
                lines.append("")
            for block in page.blocks:
                if block.block_type == "title":
                    lines.append(f"### {block.text}")
                elif block.block_type == "table":
                    if block.markdown:
                        lines.append(block.markdown)
                    elif block.html:
                        lines.append(block.html)
                    lines.append("")
                elif block.markdown:
                    lines.append(block.markdown)
                    lines.append("")
                else:
                    if block.text:
                        lines.append(block.text)
                        lines.append("")
        return "\n".join(lines).strip() + "\n"
