from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


EngineName = Literal["auto", "paddle_vl", "paddle_ocr", "surya"]


@dataclass(slots=True)
class PreprocessConfig:
    enabled: bool = True
    deskew: bool = True
    denoise: bool = True
    adaptive_threshold: bool = False
    strengthen_table_lines: bool = True
    line_kernel_scale: int = 40


@dataclass(slots=True)
class ReviewConfig:
    page_score_threshold: float = 0.70
    block_score_threshold: float = 0.65
    numeric_mismatch_penalty: float = 0.10
    empty_table_penalty: float = 0.20


@dataclass(slots=True)
class OutputConfig:
    save_raw: bool = True
    save_markdown: bool = True
    save_html: bool = True
    save_comparison: bool = True
    save_overlay: bool = True
    save_intermediate_images: bool = False


@dataclass(slots=True)
class PipelineConfig:
    engine: EngineName = "auto"
    language: str = "korean"
    device: str = "gpu"
    use_gpu_if_available: bool = True
    merge_cross_page_tables: bool = True
    relevel_titles: bool = True
    concatenate_pages: bool = False
    workers: int = 1
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    temp_dir: Optional[Path] = None
