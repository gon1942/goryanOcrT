from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from ..pdf_utils import image_size, render_pdf_to_images
from ..schemas import OCRBlock, OCRDocument, OCRPage, InputProfile
from ..utils.io import ensure_dir, load_json, save_json
from ..utils.text import normalize_whitespace
from .base import BaseOCREngine


class PaddleOCRAdapter(BaseOCREngine):
    name = "paddle_ocr"

    def __init__(self, *, device: str = "gpu", language: str = "korean"):
        self.device = device
        self.language = language

    def available(self) -> bool:
        try:
            import paddle  # type: ignore
            from paddleocr import PaddleOCR  # type: ignore
            _ = paddle
            _ = PaddleOCR
            return True
        except Exception:
            return False

    def process(self, input_path: str | Path, work_dir: str | Path, profile: InputProfile) -> OCRDocument:
        try:
            import paddle  # type: ignore
            from paddleocr import PaddleOCR  # type: ignore
            _ = paddle
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PaddleOCR 실행에 필요한 `paddle` 모듈이 없습니다. "
                "공식 문서에 따라 `paddlepaddle` 또는 `paddlepaddle-gpu`를 먼저 설치하세요."
            ) from exc

        input_path = Path(input_path)
        work_dir = ensure_dir(work_dir)
        raw_dir = ensure_dir(work_dir / "raw_paddle_ocr")

        ocr = PaddleOCR(
            lang=self.language,
            device=self.device,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

        if input_path.suffix.lower() == ".pdf":
            image_dir = ensure_dir(work_dir / "rendered_pages")
            inputs = render_pdf_to_images(input_path, image_dir)
        else:
            inputs = [input_path]

        pages: list[OCRPage] = []
        for idx, img_path in enumerate(inputs, start=1):
            page_dir = ensure_dir(raw_dir / f"page_{idx:04d}")
            result = list(ocr.predict(str(img_path)))
            raw_items: list[dict[str, Any]] = []
            for item_idx, res in enumerate(result, start=1):
                try:
                    res.save_to_json(save_path=str(page_dir))
                except Exception:
                    pass
                raw_items.append(self._result_to_data(res))
            raw_json_path = page_dir / "raw_result.json"
            save_json(raw_json_path, raw_items)
            page = self._parse_page(raw_items, img_path, idx)
            page.raw_paths = {"json": str(raw_json_path)}
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

    def _parse_page(self, raw_items: list[dict[str, Any]], img_path: Path, page_no: int) -> OCRPage:
        width, height = image_size(img_path)
        page = OCRPage(page_no=page_no, width=width, height=height, source_image=str(img_path))
        blocks: list[OCRBlock] = []
        counter = 1
        for item in raw_items:
            rec_texts = item.get("rec_texts") or []
            rec_scores = item.get("rec_scores") or []
            dt_polys = item.get("dt_polys") or []
            for text, score, poly in zip(rec_texts, rec_scores, dt_polys):
                blocks.append(
                    OCRBlock(
                        block_id=f"p{page_no}_t{counter}",
                        block_type="text",
                        page_no=page_no,
                        bbox=self._poly_to_bbox(poly),
                        text=normalize_whitespace(str(text)),
                        confidence=float(score) if isinstance(score, (int, float)) else 0.75,
                        source_engine=self.name,
                    )
                )
                counter += 1
        page.blocks = blocks
        return page

    def _result_to_data(self, res: Any) -> dict[str, Any]:
        for method_name in ("to_dict", "json"):
            method = getattr(res, method_name, None)
            if callable(method):
                data = method()
                if isinstance(data, dict):
                    return data
        # Paddle result objects often expose arrays via attributes.
        data = {}
        for key in ("rec_texts", "rec_scores", "dt_polys", "textline_orientation_angles"):
            if hasattr(res, key):
                data[key] = getattr(res, key)
        # Some versions store data in `res` or `_res`
        for key in ("res", "_res"):
            if hasattr(res, key) and isinstance(getattr(res, key), dict):
                data.update(getattr(res, key))
        return data

    def _poly_to_bbox(self, poly: Any) -> list[float]:
        if isinstance(poly, (list, tuple)) and poly and isinstance(poly[0], (list, tuple)):
            xs = [float(p[0]) for p in poly]
            ys = [float(p[1]) for p in poly]
            return [min(xs), min(ys), max(xs), max(ys)]
        if isinstance(poly, (list, tuple)) and len(poly) == 4:
            return [float(v) for v in poly]
        return [0.0, 0.0, 10.0, 10.0]
