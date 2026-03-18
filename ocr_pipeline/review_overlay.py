from __future__ import annotations

from pathlib import Path

from .schemas import OCRPage


def _lazy_import_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("review overlay 생성을 위해 OpenCV가 필요합니다.") from exc
    return cv2


def draw_overlay(page: OCRPage, output_path: str | Path) -> Path:
    cv2 = _lazy_import_cv2()
    img = cv2.imread(page.source_image)
    if img is None:
        raise FileNotFoundError(page.source_image)
    for block in page.blocks:
        x1, y1, x2, y2 = normalize_bbox(block.bbox)
        color = (0, 0, 255) if block.review else (0, 160, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{block.block_type}:{block.confidence:.2f}"
        cv2.putText(img, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    return output_path


def normalize_bbox(bbox: list[float]) -> tuple[int, int, int, int]:
    if len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        return int(x1), int(y1), int(x2), int(y2)
    if len(bbox) >= 8:
        xs = bbox[0::2]
        ys = bbox[1::2]
        return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    return 0, 0, 10, 10
