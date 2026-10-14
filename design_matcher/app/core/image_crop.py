"""
Image cropping, perspective warping, and coordinate transformation module.
Ensures zero-loss rectangle cropping and ultra-sharp 4-corner perspective warping:
- Uses original full-resolution BGR numpy array (never downscaled display pixmaps)
- High-quality sub-pixel INTER_CUBIC perspective warping with BORDER_REPLICATE
- Single-pass INTER_LANCZOS4 upscaling for small crops (<1000px long side)
- Edge-preserving unsharp mask sharpening
- Accurate view <-> image coordinate mapping with High-DPI support
- EXIF orientation support via Pillow
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


def load_image_with_exif(path: str | Path) -> np.ndarray:
    """
    Loads an image from file path with Unicode support and applies EXIF orientation.
    Returns BGR uint8 numpy array.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        with Image.open(p) as pil_img:
            transposed = ImageOps.exif_transpose(pil_img)
            rgb = np.array(transposed.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception as exc:
        logger.debug("PIL exif_transpose load fallback (%s)", exc)
        data = np.fromfile(str(p), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Unable to decode image file: {path}")
        return bgr


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 coordinates clockwise: top-left, top-right, bottom-right, bottom-left.
    - Top-left has smallest sum (x + y)
    - Bottom-right has largest sum (x + y)
    - Top-right has smallest diff (y - x)
    - Bottom-left has largest diff (y - x)

    Returns:
        np.ndarray of shape (4, 2) dtype float32.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape((4, 2))
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]       # Top-Left
    rect[2] = pts[np.argmax(s)]       # Bottom-Right

    diff = np.diff(pts, axis=1).reshape(-1)  # diff = y - x
    rect[1] = pts[np.argmin(diff)]    # Top-Right
    rect[3] = pts[np.argmax(diff)]    # Bottom-Left

    return rect


def crop_rect(
    image: np.ndarray,
    rect: Tuple[float, float, float, float] | List[float] | np.ndarray,
) -> np.ndarray:
    """
    Crops a rectangular sub-region from the original image array using numpy slicing.
    Converts to integer coordinates, clamps to image bounds, and slices with numpy:
    `crop = original_bgr[y0:y1, x0:x1].copy()`.
    Zero resampling, zero quality loss.

    Args:
        image: Original full-resolution BGR image.
        rect: [x, y, w, h] rectangle in image coordinates.

    Returns:
        Cropped numpy array.
    """
    if image is None or image.size == 0:
        return image

    h, w = image.shape[:2]
    rx, ry, rw, rh = [float(v) for v in rect]
    x0 = int(round(rx))
    y0 = int(round(ry))
    x1 = int(round(rx + rw))
    y1 = int(round(ry + rh))

    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0

    # Clamp to image bounds
    x0 = max(0, min(w, x0))
    y0 = max(0, min(h, y0))
    x1 = max(0, min(w, x1))
    y1 = max(0, min(h, y1))

    if x1 - x0 < 2 or y1 - y0 < 2:
        return image.copy()

    return image[y0:y1, x0:x1].copy()


def sharpen(img: np.ndarray) -> np.ndarray:
    """
    Gentle unsharp mask filter to restore crisp edges:
    blur = cv2.GaussianBlur(img, (0,0), 1.2)
    out = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    """
    if img is None or img.size == 0:
        return img
    blur = cv2.GaussianBlur(img, (0, 0), 1.2)
    out = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def warp_four_points(
    image: np.ndarray,
    pts: np.ndarray,
    sharpen_output: bool = True,
    upscale_small: bool = True,
) -> np.ndarray:
    """
    Performs high-quality 4-point perspective warp on the original full-resolution image:
    - Orders points clockwise (TL, TR, BR, BL) with sub-pixel float32 precision
    - Computes destination width and height from real edge lengths:
        width  = max(dist(tl, tr), dist(bl, br))
        height = max(dist(tl, bl), dist(tr, br))
    - Warps with cv2.INTER_CUBIC and cv2.BORDER_REPLICATE
    - Upscales small outputs (< 1000px long side) once with INTER_LANCZOS4 to 1200px long side
    - Applies gentle unsharp mask sharpening if sharpen_output=True

    Args:
        image: Original full-resolution BGR image.
        pts: 4 corner points in image coordinates.
        sharpen_output: If True, applies unsharp mask sharpening.
        upscale_small: If True, upscales outputs smaller than 1000px on long side to 1200px.

    Returns:
        Rectified, sharp BGR image.
    """
    if image is None or image.size == 0 or pts is None or len(pts) != 4:
        return image

    ordered = order_points(pts)
    tl, tr, br, bl = ordered

    # Compute Euclidean edge lengths
    width_top = float(np.hypot(tr[0] - tl[0], tr[1] - tl[1]))
    width_bot = float(np.hypot(br[0] - bl[0], br[1] - bl[1]))
    max_w = max(1, int(round(max(width_top, width_bot))))

    height_left = float(np.hypot(bl[0] - tl[0], bl[1] - tl[1]))
    height_right = float(np.hypot(br[0] - tr[0], br[1] - tr[1]))
    max_h = max(1, int(round(max(height_left, height_right))))

    dst_pts = np.array(
        [
            [0.0, 0.0],
            [float(max_w - 1), 0.0],
            [float(max_w - 1), float(max_h - 1)],
            [0.0, float(max_h - 1)],
        ],
        dtype=np.float32,
    )

    src_pts = ordered.astype(np.float32)
    m = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(
        image,
        m,
        (max_w, max_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Upscale once with INTER_LANCZOS4 if smaller than 1000 px on long side
    if upscale_small:
        long_side = max(max_w, max_h)
        if 0 < long_side < 1000:
            scale = 1200.0 / float(long_side)
            new_w = max(1, int(round(max_w * scale)))
            new_h = max(1, int(round(max_h * scale)))
            warped = cv2.resize(warped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    if sharpen_output:
        warped = sharpen(warped)

    return warped


def view_to_image_coords(
    view_x: float,
    view_y: float,
    scale: float,
    offset_x: float,
    offset_y: float,
    img_w: int,
    img_h: int,
    clamp: bool = True,
) -> Tuple[float, float]:
    """
    Converts a point from widget/view coordinates to original image coordinates.
    """
    if scale <= 0:
        return 0.0, 0.0
    ix = (view_x - offset_x) / scale
    iy = (view_y - offset_y) / scale
    if clamp:
        ix = max(0.0, min(float(img_w), ix))
        iy = max(0.0, min(float(img_h), iy))
    return ix, iy


def image_to_view_coords(
    img_x: float,
    img_y: float,
    scale: float,
    offset_x: float,
    offset_y: float,
) -> Tuple[float, float]:
    """
    Converts a point from original image coordinates to widget/view coordinates.
    """
    vx = offset_x + (img_x * scale)
    vy = offset_y + (img_y * scale)
    return vx, vy
