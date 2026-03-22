from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

from PIL import Image


def _lazy_import_fitz():
    try:
        import fitz  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyMuPDF(fitz)가 필요합니다. `pip install pymupdf` 후 다시 실행하세요.") from exc
    return fitz


def detect_pdf_text_layer(pdf_path: str | Path, sample_pages: int = 3) -> tuple[bool, str]:
    fitz = _lazy_import_fitz()
    doc = fitz.open(str(pdf_path))
    max_pages = min(sample_pages, len(doc))
    total_chars = 0
    for i in range(max_pages):
        page = doc.load_page(i)
        total_chars += len(page.get_text("text").strip())
    has_text = total_chars > 30
    pdf_type = "text" if has_text else "image"
    return has_text, pdf_type


def get_pdf_page_count(pdf_path: str | Path) -> int:
    fitz = _lazy_import_fitz()
    doc = fitz.open(str(pdf_path))
    return len(doc)


def extract_pdf_page_texts(pdf_path: str | Path) -> list[str]:
    fitz = _lazy_import_fitz()
    doc = fitz.open(str(pdf_path))
    texts: list[str] = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        texts.append(page.get_text("text").strip())
    return texts


def extract_pdf_page_words(
    pdf_path: str | Path,
    page_no: int,
    *,
    scale_to: tuple[int, int] | None = None,
) -> list[tuple[float, float, float, float, str]]:
    fitz = _lazy_import_fitz()
    doc = fitz.open(str(pdf_path))
    page_index = max(0, page_no - 1)
    if page_index >= len(doc):
        return []
    page = doc.load_page(page_index)
    rect = page.rect
    sx = 1.0
    sy = 1.0
    if scale_to and rect.width and rect.height:
        sx = float(scale_to[0]) / float(rect.width)
        sy = float(scale_to[1]) / float(rect.height)
    words: list[tuple[float, float, float, float, str]] = []
    for x0, y0, x1, y1, text, *_ in page.get_text("words"):
        token = str(text).strip()
        if not token:
            continue
        words.append((x0 * sx, y0 * sy, x1 * sx, y1 * sy, token))
    return words


def render_pdf_to_images(pdf_path: str | Path, out_dir: str | Path, dpi: int = 200) -> list[Path]:
    fitz = _lazy_import_fitz()
    doc = fitz.open(str(pdf_path))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    outputs: list[Path] = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        out_path = out_dir / f"page_{i + 1:04d}.png"
        pix.save(str(out_path))
        outputs.append(out_path)
    return outputs


def image_size(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size
