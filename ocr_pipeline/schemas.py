from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


BBox = list[float]


@dataclass(slots=True)
class OCRBlock:
    block_id: str
    block_type: str
    page_no: int
    bbox: BBox
    text: str = ""
    html: str = ""
    markdown: str = ""
    confidence: float = 0.0
    review: bool = False
    review_reason: list[str] = field(default_factory=list)
    source_engine: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OCRPage:
    page_no: int
    width: int
    height: int
    source_image: str
    page_confidence: float = 0.0
    needs_review: bool = False
    review_reason: list[str] = field(default_factory=list)
    blocks: list[OCRBlock] = field(default_factory=list)
    raw_paths: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_no": self.page_no,
            "width": self.width,
            "height": self.height,
            "source_image": self.source_image,
            "page_confidence": self.page_confidence,
            "needs_review": self.needs_review,
            "review_reason": self.review_reason,
            "blocks": [block.to_dict() for block in self.blocks],
            "raw_paths": self.raw_paths,
            "extra": self.extra,
        }


@dataclass(slots=True)
class OCRDocument:
    document_id: str
    source_file: str
    pdf_type: str
    page_count: int
    overall_confidence: float = 0.0
    review_required_pages: list[int] = field(default_factory=list)
    pages: list[OCRPage] = field(default_factory=list)
    engine_used: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source_file": self.source_file,
            "pdf_type": self.pdf_type,
            "page_count": self.page_count,
            "overall_confidence": self.overall_confidence,
            "review_required_pages": self.review_required_pages,
            "engine_used": self.engine_used,
            "metadata": self.metadata,
            "pages": [page.to_dict() for page in self.pages],
        }


@dataclass(slots=True)
class InputProfile:
    source_file: str
    file_type: str
    pdf_type: str = "unknown"
    page_count: int = 1
    has_text_layer: bool = False
    layout_type: str = "unknown"
    table_complexity: str = "unknown"
    quality_tier: str = "unknown"
    comparison_mode: str = "review_only"
    routing_hint: str = "prefer_auto"
    probable_table_pages: list[int] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
