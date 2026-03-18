from __future__ import annotations

from .schemas import OCRDocument


def choose_best_document(primary: OCRDocument, secondary: OCRDocument | None = None) -> OCRDocument:
    if secondary is None:
        return primary
    if secondary.overall_confidence > primary.overall_confidence:
        return secondary
    return primary
