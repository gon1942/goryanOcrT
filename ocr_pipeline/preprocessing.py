from __future__ import annotations

from pathlib import Path


def _lazy_import_cv2():
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV와 numpy가 필요합니다. `pip install opencv-python-headless numpy` 후 다시 실행하세요.") from exc
    return cv2, np


def preprocess_image(input_path: str | Path, output_path: str | Path, *,
                     deskew: bool = True,
                     denoise: bool = True,
                     adaptive_threshold: bool = False,
                     strengthen_table_lines: bool = True,
                     line_kernel_scale: int = 40) -> Path:
    cv2, np = _lazy_import_cv2()
    src = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if src is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {input_path}")

    img = src.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if denoise:
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    if deskew:
        angle = estimate_skew_angle(gray)
        if abs(angle) > 0.2:
            gray = rotate_bound_gray(gray, angle)

    if adaptive_threshold:
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
        )

    if strengthen_table_lines:
        gray = enhance_table_lines(gray, line_kernel_scale=line_kernel_scale)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), gray)
    return out


def estimate_skew_angle(gray):
    cv2, np = _lazy_import_cv2()
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thr > 0))
    if coords.size == 0:
        return 0.0
    rect = cv2.minAreaRect(coords[:, ::-1].astype("float32"))
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    return float(angle)


def rotate_bound_gray(gray, angle):
    cv2, np = _lazy_import_cv2()
    h, w = gray.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(gray, matrix, (new_w, new_h), borderValue=255)


def enhance_table_lines(gray, line_kernel_scale: int = 40):
    cv2, np = _lazy_import_cv2()
    inv = cv2.bitwise_not(gray)
    bw = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    h, w = bw.shape[:2]
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // line_kernel_scale), 1))
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h // line_kernel_scale)))
    hor = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hor_kernel, iterations=1)
    ver = cv2.morphologyEx(bw, cv2.MORPH_OPEN, ver_kernel, iterations=1)
    grid = cv2.bitwise_or(hor, ver)
    grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    merged = cv2.bitwise_or(inv, grid)
    result = cv2.bitwise_not(merged)
    return result
