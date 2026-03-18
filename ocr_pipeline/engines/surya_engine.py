from __future__ import annotations

import uuid
from pathlib import Path

from ..schemas import OCRDocument, InputProfile
from .base import BaseOCREngine


class SuryaAdapter(BaseOCREngine):
    name = "surya"

    def available(self) -> bool:
        try:
            import surya  # type: ignore
            _ = surya
            return True
        except Exception:
            return False

    def process(self, input_path: str | Path, work_dir: str | Path, profile: InputProfile) -> OCRDocument:
        raise NotImplementedError(
            "Surya 어댑터는 자리만 만들어 둔 상태입니다. 프로젝트에서는 PaddleOCR-VL 결과와 비교 검증기로 붙이는 것을 권장합니다."
        )
