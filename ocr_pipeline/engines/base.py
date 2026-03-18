from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..schemas import OCRDocument, InputProfile


class BaseOCREngine(ABC):
    name = "base"

    @abstractmethod
    def available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def process(self, input_path: str | Path, work_dir: str | Path, profile: InputProfile) -> OCRDocument:
        raise NotImplementedError
