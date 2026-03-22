from __future__ import annotations

from pathlib import Path

from PIL import Image

from .schemas import OCRPage
from .utils.io import ensure_dir


def _lazy_import_fitz():
    try:
        import fitz  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyMuPDF(fitz)가 필요합니다. `pip install pymupdf` 후 다시 실행하세요.") from exc
    return fitz


class PDFTableCropper:
    def __init__(self, pdf_path: str | Path, cache_dir: str | Path, *, dpi: int = 400):
        self.pdf_path = Path(pdf_path)
        self.cache_dir = ensure_dir(cache_dir)
        self.dpi = dpi

    def crop_table(
        self,
        page: OCRPage,
        bbox: list[float],
        output_path: str | Path,
        *,
        padding: int = 24,
        min_width: int = 160,
        min_height: int = 80,
    ) -> Path | None:
        page_image = self._render_page(page.page_no)
        img_w, img_h = page_image.size
        if page.width <= 0 or page.height <= 0:
            return None

        sx = img_w / float(page.width)
        sy = img_h / float(page.height)
        x0, y0, x1, y1 = bbox[:4]
        left = max(0, int(round(x0 * sx)) - padding)
        top = max(0, int(round(y0 * sy)) - padding)
        right = min(img_w, int(round(x1 * sx)) + padding)
        bottom = min(img_h, int(round(y1 * sy)) + padding)
        if right - left < min_width or bottom - top < min_height:
            return None

        out_path = Path(output_path)
        ensure_dir(out_path.parent)
        crop = page_image.crop((left, top, right, bottom))
        crop.save(out_path)
        return out_path

    def _render_page(self, page_no: int) -> Image.Image:
        image_path = self.cache_dir / f"page_{page_no:04d}_{self.dpi}dpi.png"
        if image_path.exists():
            return Image.open(image_path).copy()

        fitz = _lazy_import_fitz()
        doc = fitz.open(str(self.pdf_path))
        page = doc.load_page(max(0, page_no - 1))
        zoom = self.dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        pix.save(str(image_path))
        return Image.open(image_path).copy()
