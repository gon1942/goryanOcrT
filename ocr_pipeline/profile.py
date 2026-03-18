from __future__ import annotations

import re
from pathlib import Path

from .pdf_utils import detect_pdf_text_layer, extract_pdf_page_texts, get_pdf_page_count
from .schemas import InputProfile


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
TABLE_HEADER_RE = re.compile(
    r"(구분|품목수|수량|금액|목표|실적|차이|달성율|진척율|매출비율|대상건수|준수건수|합계|계획)",
    re.I,
)
SECTION_TITLE_RE = re.compile(r"^(?:\d+\.\s*|[■☑#])")
NUMERIC_TOKEN_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?%?")


def profile_input(input_path: str | Path) -> InputProfile:
    path = Path(input_path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        has_text, pdf_type = detect_pdf_text_layer(path)
        page_count = get_pdf_page_count(path)
        page_texts = extract_pdf_page_texts(path) if has_text else []
        layout_type, probable_table_pages, layout_stats = _classify_layout(page_texts, page_count=page_count)
        table_complexity = _classify_table_complexity(page_texts, probable_table_pages)
        return InputProfile(
            source_file=str(path),
            file_type="pdf",
            pdf_type=pdf_type,
            page_count=page_count,
            has_text_layer=has_text,
            layout_type=layout_type,
            table_complexity=table_complexity,
            quality_tier="digital" if has_text else "scan",
            comparison_mode="text_diff" if has_text else "review_only",
            routing_hint=_routing_hint(layout_type, has_text),
            probable_table_pages=probable_table_pages,
            languages=["ko", "en"],
            extra=layout_stats,
        )
    if suffix in IMAGE_SUFFIXES:
        return InputProfile(
            source_file=str(path),
            file_type="image",
            pdf_type="image",
            has_text_layer=False,
            layout_type="unknown",
            table_complexity="unknown",
            quality_tier="scan",
            comparison_mode="review_only",
            routing_hint="prefer_paddle_ocr",
            languages=["ko", "en"],
        )
    raise ValueError(f"지원하지 않는 입력 형식입니다: {path}")


def _classify_layout(page_texts: list[str], *, page_count: int) -> tuple[str, list[int], dict[str, object]]:
    if not page_texts:
        return "unknown", [], {"page_count": page_count, "profile_basis": "no_text_layer"}

    page_scores: list[float] = []
    probable_table_pages: list[int] = []
    section_titles = 0
    numeric_tokens = 0
    header_hits = 0
    short_lines = 0
    line_count = 0

    for idx, text in enumerate(page_texts, start=1):
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        score = _page_table_score(lines)
        page_scores.append(score)
        if score >= 1.8:
            probable_table_pages.append(idx)
        section_titles += sum(1 for line in lines if SECTION_TITLE_RE.match(line))
        header_hits += sum(1 for line in lines if TABLE_HEADER_RE.search(line))
        line_count += len(lines)
        short_lines += sum(1 for line in lines if len(line) <= 12)
        numeric_tokens += sum(len(NUMERIC_TOKEN_RE.findall(line)) for line in lines)

    avg_score = sum(page_scores) / len(page_scores)
    short_line_ratio = short_lines / max(1, line_count)
    numeric_density = numeric_tokens / max(1, line_count)
    table_page_ratio = len(probable_table_pages) / max(1, len(page_texts))

    if table_page_ratio >= 0.6 or avg_score >= 2.2:
        layout_type = "table_heavy"
    elif avg_score >= 1.2 or probable_table_pages:
        layout_type = "mixed"
    else:
        layout_type = "text_heavy"

    return layout_type, probable_table_pages, {
        "page_count": page_count,
        "profile_basis": "text_layer_sample",
        "avg_table_score": round(avg_score, 3),
        "table_page_ratio": round(table_page_ratio, 3),
        "section_title_count": section_titles,
        "table_header_hits": header_hits,
        "numeric_density": round(numeric_density, 3),
        "short_line_ratio": round(short_line_ratio, 3),
    }


def _page_table_score(lines: list[str]) -> float:
    if not lines:
        return 0.0
    header_hits = sum(1 for line in lines if TABLE_HEADER_RE.search(line))
    numeric_lines = sum(1 for line in lines if len(NUMERIC_TOKEN_RE.findall(line)) >= 2)
    short_lines = sum(1 for line in lines if len(line) <= 12)
    section_titles = sum(1 for line in lines if SECTION_TITLE_RE.match(line))
    return (
        header_hits * 0.45
        + numeric_lines * 0.12
        + short_lines * 0.03
        + section_titles * 0.10
    )


def _classify_table_complexity(page_texts: list[str], probable_table_pages: list[int]) -> str:
    if not probable_table_pages:
        return "none"
    texts = [page_texts[idx - 1] for idx in probable_table_pages if 0 <= idx - 1 < len(page_texts)]
    joined = "\n".join(texts)
    section_titles = len(SECTION_TITLE_RE.findall(joined))
    headers = len(TABLE_HEADER_RE.findall(joined))
    numeric_tokens = len(NUMERIC_TOKEN_RE.findall(joined))
    if len(probable_table_pages) >= 2 or section_titles >= 5 or headers >= 12 or numeric_tokens >= 120:
        return "complex"
    return "simple"


def _routing_hint(layout_type: str, has_text_layer: bool) -> str:
    if layout_type in {"table_heavy", "mixed"}:
        return "prefer_paddle_vl"
    if has_text_layer:
        return "prefer_paddle_ocr"
    return "prefer_paddle_vl"
