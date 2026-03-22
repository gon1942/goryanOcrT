"""Lightweight table structure recognition adapter using paddlex TableRecognitionPipeline.

Much smaller GPU footprint (~500MB) compared to PaddleVL (~8-12GB).
Produces HTML table structure from crop images, suitable for table refiner use.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from ..schemas import OCRBlock, OCRDocument, OCRPage, InputProfile
from ..utils.io import ensure_dir, save_json
from ..utils.text import normalize_whitespace
from .base import BaseOCREngine


class PaddleTableRecognitionAdapter(BaseOCREngine):
    """Wraps paddlex TableRecognitionPipeline for table-only OCR.

    Designed specifically for the TableRefiner — takes a cropped table image
    and returns an OCRDocument containing table blocks with HTML structure.
    """

    name = "table_recognition"

    def __init__(self, *, device: str = "gpu"):
        self.device = device
        self._pipeline: Any | None = None

    def available(self) -> bool:
        try:
            from paddlex import create_pipeline  # type: ignore
            _ = create_pipeline
            return True
        except Exception:
            return False

    def _get_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        from paddlex import create_pipeline  # type: ignore
        self._pipeline = create_pipeline(pipeline="table_recognition", device=self.device)
        return self._pipeline

    def process(self, input_path: str | Path, work_dir: str | Path, profile: InputProfile) -> OCRDocument:
        try:
            from paddlex import create_pipeline  # type: ignore
            _ = create_pipeline
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "TableRecognitionPipeline 실행에 필요한 `paddlex` 모듈이 없습니다. "
                "`pip install paddlex`를 실행하세요."
            ) from exc

        input_path = Path(input_path)
        work_dir = ensure_dir(work_dir)

        pipe = self._get_pipeline()
        results = list(pipe.predict(str(input_path)))

        page = self._parse_results(results, input_path, 1)
        doc = OCRDocument(
            document_id=str(uuid.uuid4()),
            source_file=input_path.name,
            pdf_type=profile.pdf_type,
            page_count=1,
            pages=[page],
            engine_used=self.name,
            metadata={"profile": profile.to_dict()},
        )
        save_json(work_dir / "document_profile.json", profile.to_dict())
        return doc

    def _parse_results(self, results: list[Any], img_path: Path, page_no: int) -> OCRPage:
        from ..pdf_utils import image_size

        width, height = image_size(img_path)
        page = OCRPage(
            page_no=page_no,
            width=width,
            height=height,
            source_image=str(img_path),
        )

        counter = 1
        for res in results:
            html = self._extract_html(res)
            if not html:
                continue
            page.blocks.append(
                OCRBlock(
                    block_id=f"p{page_no}_t{counter}",
                    block_type="table",
                    page_no=page_no,
                    bbox=[0.0, 0.0, float(width), float(height)],
                    text="",
                    html=html,
                    markdown="",
                    confidence=0.85,
                    source_engine=self.name,
                    extra={"refiner_source": "table_recognition"},
                )
            )
            counter += 1
        return page

    def _extract_html(self, res: Any) -> str:
        """Extract table HTML from TableRecognitionPipeline result.

        Result structure:
          res.html  → dict like {'table_1': '<html><body><table>...</table></body></html>'}
          res.json  → dict with res['table_res_list'][0]['pred_html']
        """
        # Try .html attribute first (dict of table_name → html)
        html_attr = getattr(res, "html", None)
        if isinstance(html_attr, dict):
            for _key, value in html_attr.items():
                if isinstance(value, str) and "<table" in value:
                    return value
            return ""

        # Try .json attribute
        json_attr = getattr(res, "json", None)
        if callable(json_attr):
            try:
                json_attr = json_attr()
            except Exception:
                json_attr = None
        if isinstance(json_attr, dict):
            table_res_list = json_attr.get("res", {}).get("table_res_list", [])
            if table_res_list and isinstance(table_res_list[0], dict):
                pred_html = table_res_list[0].get("pred_html", "")
                if isinstance(pred_html, str) and "<table" in pred_html:
                    return pred_html
        return ""

    def close(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.close()
            except Exception:
                pass
            self._pipeline = None
