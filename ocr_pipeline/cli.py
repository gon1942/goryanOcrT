from __future__ import annotations

import argparse
from pathlib import Path

from .config import PipelineConfig
from .pipeline import OCRPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="정확도 중심 문서 OCR 파이프라인")
    parser.add_argument("--input", required=True, help="입력 파일 경로 (PDF 또는 이미지)")
    parser.add_argument("--output-dir", required=True, help="출력 디렉토리")
    parser.add_argument("--engine", default="auto", choices=["auto", "paddle_vl", "paddle_ocr", "surya"])
    parser.add_argument("--language", default="korean")
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--disable-preprocess", action="store_true")
    parser.add_argument("--no-merge-cross-page-tables", action="store_true")
    parser.add_argument("--no-relevel-titles", action="store_true")
    parser.add_argument("--concatenate-pages", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PipelineConfig(
        engine=args.engine,
        language=args.language,
        device=args.device,
        merge_cross_page_tables=not args.no_merge_cross_page_tables,
        relevel_titles=not args.no_relevel_titles,
        concatenate_pages=args.concatenate_pages,
    )
    config.preprocess.enabled = not args.disable_preprocess
    pipeline = OCRPipeline(config)
    doc = pipeline.run(Path(args.input), Path(args.output_dir))
    print(f"done: pages={doc.page_count}, engine={doc.engine_used}, confidence={doc.overall_confidence:.3f}")
    if doc.review_required_pages:
        print("review pages:", ", ".join(str(x) for x in doc.review_required_pages))


if __name__ == "__main__":
    main()
