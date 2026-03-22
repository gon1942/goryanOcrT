from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


EngineName = Literal["auto", "paddle_vl", "paddle_ocr", "surya", "table_recognition"]


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
class TableRefineConfig:
    enabled: bool = True
    engine: EngineName = "table_recognition"
    model_path: Optional[str] = None
    """Custom fine-tuned SLANet model directory (e.g. models/slanet_custom_v1/export)."""
    dpi: int = 400
    padding: int = 24
    max_tables_per_page: int = 8
    min_crop_width: int = 160
    min_crop_height: int = 80
    prefer_larger_table: bool = True


@dataclass(slots=True)
class LoRAConfig:
    enabled: bool = False
    adapter_path: Optional[str] = None
    """Path to LoRA adapter directory (e.g. models/lora_vl_v1/final)."""
    lora_rank: int = 8
    lora_alpha: int = 16


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
    table_refine: TableRefineConfig = field(default_factory=TableRefineConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    temp_dir: Optional[Path] = None
