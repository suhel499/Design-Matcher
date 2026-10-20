"""
Image preprocessing and stencil mask normalization.
Converts both catalog tiles and customer photos into a normalized black/white stencil mask.
Convention:
    - Solid metal (panel material): 255 (White)
    - Cut-outs (voids, holes, apertures): 0 (Black)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def remove_pink_watermark(image_bgr: np.ndarray) -> np.ndarray:
    """
    Detects and inpaints pink/red diagonal watermarks on catalog pages and tile crops.
    Preserves underlying neutral silver panels and dark cutout background.
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    b, g, r = cv2.split(image_bgr)

    # Silver panels and navy background have neutral tones where R ≈ G ≈ B
    # Pink watermarks have Red noticeably higher than Green (R > G + 12) and R > 100
    r_int = r.astype(np.int16)
    g_int = g.astype(np.int16)
    diff_rg = r_int - g_int

    # Condition 1: Red channel distinctly higher than Green channel
    rgb_pink = (diff_rg > 12) & (r > 95)

    # Condition 2: HSV magenta/pink hue check (H in [145, 180], S > 18)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    hsv_pink = ((h >= 145) & (h <= 180) | (h >= 0) & (h <= 10)) & (s > 18) & (v > 75)

    watermark_mask = (rgb_pink | hsv_pink).astype(np.uint8) * 255

    if cv2.countNonZero(watermark_mask) < 15:
        return image_bgr

    # Dilate mask slightly to cover antialiased edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated_mask = cv2.dilate(watermark_mask, kernel, iterations=1)

    # Inpaint using Telea fast marching method
    cleaned_bgr = cv2.inpaint(image_bgr, dilated_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return cleaned_bgr


def equalize(gray: np.ndarray) -> np.ndarray:
    """Adaptive lighting & contrast equalization using CLAHE."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    out = clahe.apply(gray)
    return cv2.GaussianBlur(out, (3, 3), 0)


def detect_polarity(binary: np.ndarray) -> bool:
    """
    Returns True if `binary` must be inverted so that solid metal = 255 (white) and cutouts = 0 (black).
    In real laser-cut panels, the perimeter border is solid metal.
    If the perimeter border is already white (> 0.50), polarity is standard -> returns False.
    If the perimeter border is predominantly black (<= 0.50), metal is currently black -> returns True.
    """
    if binary is None or binary.size == 0:
        return False
    h, w = binary.shape[:2]
    bw = max(2, int(w * 0.03))
    bh = max(2, int(h * 0.03))
    border = np.concatenate([
        binary[:bh, :].ravel(),
        binary[-bh:, :].ravel(),
        binary[:, :bw].ravel(),
        binary[:, -bw:].ravel(),
    ])
    white_border = float((border == 255).mean())
    return white_border <= 0.45


def remove_specks(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Removes tiny noise specks in both classes below min_area."""
    out = mask.copy()
    for val in (0, 255):
        inv = (out == val).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, 8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                out[lab == i] = 255 - val
    return out


def is_cad_wireframe(bgr: np.ndarray) -> Tuple[bool, float]:
    """
    Detects if an image is a CAD viewport / DXF screenshot:
    - High background uniformity (> 75% of pixels match background median)
    - Very low border variance (standard deviation < 8.0 on digital/vector canvases)
    - Background median intensity either very high (> 205 for paper/cream canvas) or very low (< 50 for dark modelspace)
    Returns (is_cad, bg_median_intensity).
    """
    if bgr is None or bgr.size == 0:
        return False, 0.0

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr.copy()
    border_px = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    bg_median = float(np.median(border_px))
    border_std = float(np.std(border_px))

    # CAD viewports have pure flat digital backgrounds (border std < 8.0)
    # and extreme background brightness (> 205 for paper/cream / < 50 for dark modelspace)
    if border_std > 8.0 or (50.0 <= bg_median <= 205.0):
        return False, bg_median

    bg_ratio = float(np.mean(np.abs(gray.astype(np.float32) - bg_median) <= 18.0))
    is_cad = (bg_ratio >= 0.75) and (border_std < 8.0)
    return is_cad, bg_median


def compute_otsu_metrics(gray: np.ndarray) -> Tuple[int, float]:
    """Computes optimal Otsu threshold and between-class variance ratio (eta in [0, 1])."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    prob = hist / float(gray.size)
    cum_prob = np.cumsum(prob)
    cum_mean = np.cumsum(prob * np.arange(256))
    global_mean = cum_mean[-1]

    sigma_b = np.zeros(256)
    valid = (cum_prob > 0) & (cum_prob < 1)
    sigma_b[valid] = ((global_mean * cum_prob[valid] - cum_mean[valid]) ** 2) / (cum_prob[valid] * (1 - cum_prob[valid]))
    t_opt = int(np.argmax(sigma_b))
    total_var = np.sum(prob * (np.arange(256) - global_mean) ** 2)
    eta = float(sigma_b[t_opt] / max(total_var, 1e-6))
    return t_opt, eta


def normalize_stencil(
    bgr: np.ndarray,
    crop: bool = True,
    mode: str = "auto",
    enhance: bool = True,
    threshold_offset: int = 0,
    invert: bool = False,
    trim_border_pct: float = 0.0,
) -> Tuple[np.ndarray, dict]:
    """
    Normalizes a customer photo, panel crop, or catalog page into an accurate, clean, production-quality stencil mask:
    - Automatically detects CAD / DXF screenshots and preserves wireframe cutouts
    - Adaptive edge-preserving bilateral filtering to suppress paint grain and camera noise
    - Precision Otsu thresholding calibrated to CNC laser-cut apertures
    - Clean outer perimeter frame enforcement and corner pin suppression for user crops
    - Guaranteed standard polarity: solid metal = 255 (White), laser cutouts = 0 (Black)
    Returns (mask, {"polarity_inverted": bool, "is_cad": bool}).
    """
    if bgr is None or bgr.size == 0:
        return np.zeros((224, 224), dtype=np.uint8), {"polarity_inverted": False, "is_cad": False}

    # 1. CAD Wireframe Detection & Conversion
    is_cad, bg_median = is_cad_wireframe(bgr)
    if is_cad:
        gray_cad = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr.copy()
        if bgr.ndim == 3:
            border_bgr = np.concatenate([bgr[0, :, :], bgr[-1, :, :], bgr[:, 0, :], bgr[:, -1, :]], axis=0)
            bg_bgr = np.median(border_bgr, axis=0)
            diff = np.max(np.abs(bgr.astype(np.float32) - bg_bgr), axis=2)
        else:
            diff = np.abs(gray_cad.astype(np.float32) - bg_median)

        strokes = (diff > 25.0).astype(np.uint8) * 255
        k_size = max(3, int(round(min(gray_cad.shape) * 0.006)))
        if k_size % 2 == 0:
            k_size += 1
        thick_strokes = cv2.dilate(strokes, cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size)))

        if bg_median >= 128.0:
            mask = 255 - thick_strokes
            inv_cad = False
        else:
            mask = thick_strokes
            inv_cad = True

        if crop:
            coords = cv2.findNonZero(thick_strokes)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                pad = max(10, int(0.02 * max(w, h)))
                mask = mask[
                    max(0, y - pad): min(mask.shape[0], y + h + pad),
                    max(0, x - pad): min(mask.shape[1], x + w + pad),
                ]
        if invert:
            mask = cv2.bitwise_not(mask)
            inv_cad = not inv_cad
        return mask, {"polarity_inverted": inv_cad, "is_cad": True}

    # 2. Camera Photo Pipeline
    processed = bgr.copy()
    was_dark_metal = False

    if crop:
        gray_full = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY) if processed.ndim == 3 else processed.copy()
        _, bin_full = cv2.threshold(gray_full, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        border_px = np.concatenate([bin_full[0, :], bin_full[-1, :], bin_full[:, 0], bin_full[:, -1]])
        white_border = float((border_px == 255).mean())

        if white_border > 0.65:
            # White backdrop around dark metal panel (e.g. img_dark_on_light)
            bin_inv = cv2.bitwise_not(bin_full)
            n, lab, stats, _ = cv2.connectedComponentsWithStats((bin_inv == 255).astype(np.uint8), 8)
            if n > 1:
                i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                x, y, w, h = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]), int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
                if w < gray_full.shape[1] - 8 and h < gray_full.shape[0] - 8 and w >= 20 and h >= 20:
                    processed = processed[y : y + h, x : x + w]
                    was_dark_metal = True
        elif white_border < 0.35:
            # Dark backdrop around light metal panel (e.g. img_light_on_dark)
            n, lab, stats, _ = cv2.connectedComponentsWithStats((bin_full == 255).astype(np.uint8), 8)
            if n > 1:
                i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                x, y, w, h = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]), int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
                if w < gray_full.shape[1] - 8 and h < gray_full.shape[0] - 8 and w >= 20 and h >= 20:
                    processed = processed[y : y + h, x : x + w]
                    was_dark_metal = False

    if trim_border_pct > 0.0:
        hp, wp = processed.shape[:2]
        ty = max(1, int(hp * trim_border_pct))
        tx = max(1, int(wp * trim_border_pct))
        if hp > 2 * ty + 10 and wp > 2 * tx + 10:
            processed = processed[ty : hp - ty, tx : wp - tx]

    hp, wp = processed.shape[:2]

    # Remove pink watermark if catalog tile source
    processed = remove_pink_watermark(processed)

    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY) if processed.ndim == 3 else processed.copy()

    # Bilateral smoothing for noise removal without edge softening
    if enhance:
        smooth = cv2.bilateralFilter(gray, d=7, sigmaColor=35, sigmaSpace=35)
    else:
        smooth = gray.copy()

    t_otsu, eta = compute_otsu_metrics(smooth)
    bw_c = max(2, int(wp * 0.03))
    bh_c = max(2, int(hp * 0.03))
    b_strip = np.concatenate([
        smooth[:bh_c, :].ravel(),
        smooth[-bh_c:, :].ravel(),
        smooth[:, :bw_c].ravel(),
        smooth[:, -bw_c:].ravel(),
    ])
    perim_bright_ratio = float((b_strip > t_otsu).mean())
    white_ratio = float((smooth > t_otsu).mean())

    if was_dark_metal:
        is_dark = True
    elif mode == "raw_sheet":
        is_dark = False
    elif mode in {"backlit", "installed_product"}:
        is_dark = True
    elif perim_bright_ratio > 0.60:
        is_dark = False
    elif white_ratio < 0.25:
        is_dark = True
    else:
        is_dark = False

    t_final = np.clip(t_otsu + threshold_offset * 1.5, 20, 235)
    mask = np.zeros((hp, wp), dtype=np.uint8)

    if not is_dark:
        mask[smooth > t_final] = 255
        polarity_inverted = False
    else:
        mask[smooth <= t_final] = 255
        polarity_inverted = True

    min_area = max(8, int(gray.size / 30000))
    mask = remove_specks(mask, min_area=min_area)

    # Perimeter and corners enforcement for user crops and panels with solid metal frames
    if perim_bright_ratio > 0.80:
        cw = max(8, int(wp * 0.06))
        ch = max(8, int(hp * 0.06))
        mask[:ch, :cw] = 255
        mask[:ch, -cw:] = 255
        mask[-ch:, :cw] = 255
        mask[-ch:, -cw:] = 255

        fw = max(2, int(wp * 0.02))
        fh = max(2, int(hp * 0.02))
        mask[:fh, :] = 255
        mask[-fh:, :] = 255
        mask[:, :fw] = 255
        mask[:, -fw:] = 255

    if invert:
        mask = cv2.bitwise_not(mask)
        polarity_inverted = not polarity_inverted

    return mask, {"polarity_inverted": polarity_inverted, "is_cad": False}


def normalize_dxf_screenshot(
    bgr: np.ndarray,
    dilation_ratio: float = 0.004,
    stroke_threshold: float = 25.0,
) -> Tuple[np.ndarray, dict]:
    """
    Converts a DXF viewport screenshot (thin wireframe strokes on a uniform background)
    into a FILLED stencil mask that matches the convention used during vector indexing:
        White (255) = solid metal panel material
        Black  (0)  = laser cut-out loops / holes / slits

    Why this is needed
    ------------------
    During indexing, DXF linework is converted via linework_to_mask() which uses
    Shapely polygonize + XOR flood-fill to produce *filled* regions.
    A screenshot of that same DXF is a wireframe (only the outlines are drawn),
    so it looks nothing like the filled stencil in the FAISS index and embedding space.
    This function bridges that gap by:
      1. Extracting stroke pixels (dark lines on the uniform background).
      2. Thickening strokes to close near-gaps.
      3. Flood-filling from corners to mark the exterior background.
      4. Everything not exterior and not stroke = enclosed void = cut-out (black).

    Returns
    -------
    (mask, info) where mask is uint8 (255=metal, 0=cutout).
    """
    if bgr is None or bgr.size == 0:
        return np.zeros((224, 224), dtype=np.uint8), {
            "polarity_inverted": False, "is_cad": True, "method": "empty"
        }

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr.copy()

    # Detect uniform background intensity using border pixels
    border_px = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    bg_median = float(np.median(border_px))

    # Extract stroke pixels: anything that deviates > threshold from background
    diff = np.abs(gray.astype(np.float32) - bg_median)
    stroke_binary = (diff > stroke_threshold).astype(np.uint8) * 255

    # Thicken strokes to close ~pixel gaps left by anti-aliasing
    k = max(3, int(round(min(gray.shape) * dilation_ratio)))
    if k % 2 == 0:
        k += 1
    struct = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    thick_strokes = cv2.dilate(stroke_binary, struct, iterations=2)

    h, w = thick_strokes.shape[:2]

    # Build padded canvas for flood fill (flood fill needs room at borders)
    flood_canvas = cv2.bitwise_not(thick_strokes)  # 255=open space, 0=wall (stroke)
    pad_canvas = cv2.copyMakeBorder(flood_canvas, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
    mask_ff = np.zeros((h + 4, w + 4), np.uint8)
    # Flood from top-left corner to mark exterior background
    cv2.floodFill(pad_canvas, mask_ff, (0, 0), 0)
    exterior_mask = (pad_canvas[1:h + 1, 1:w + 1] == 0)

    # Build filled stencil:
    #   - Start all white (metal)
    #   - Paint stroke walls black
    #   - Paint enclosed interior voids black (= cut-outs)
    stencil = np.full((h, w), 255, dtype=np.uint8)
    stencil[thick_strokes > 0] = 0

    # Enclosed interior = not exterior AND not a stroke pixel
    interior_mask = (~exterior_mask) & (thick_strokes == 0)
    interior_img = interior_mask.astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(interior_img, connectivity=8)
    total_pixels = h * w
    for lbl in range(1, num_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < max(4, int(0.0002 * total_pixels)):
            continue  # skip tiny noise specks
        if area > 0.85 * total_pixels:
            continue  # skip if this blob is essentially the entire canvas
        stencil[labels == lbl] = 0

    # Crop to bounding box of the stroke content with small padding
    coords = cv2.findNonZero(thick_strokes)
    if coords is not None:
        x, y, cw, ch = cv2.boundingRect(coords)
        pad = max(8, int(0.02 * max(cw, ch)))
        stencil = stencil[
            max(0, y - pad): min(h, y + ch + pad),
            max(0, x - pad): min(w, x + cw + pad),
        ]

    return stencil, {"polarity_inverted": False, "is_cad": True, "method": "dxf_screenshot_fill"}


def convert_tile_to_stencil_mask(tile_bgr: np.ndarray) -> np.ndarray:
    """
    Converts a cropped catalog tile to a clean binary stencil mask.
    Handles metallic reflection gradients and watermark removal.
    Solid silver metal -> 255 (White)
    Dark cut-out apertures -> 0 (Black)
    """
    if tile_bgr is None or tile_bgr.size == 0:
        return np.zeros((224, 224), dtype=np.uint8)

    # Step 1: Remove pink watermark
    cleaned = remove_pink_watermark(tile_bgr)

    # Step 2: Grayscale
    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY) if cleaned.ndim == 3 else cleaned.copy()

    # Step 3: Combine Otsu thresholding and adaptive thresholding
    # Otsu reliably captures large dark voids/cutouts globally, while adaptive
    # thresholding captures fine slits and compensates for 3D metallic reflection gradients.
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu_bin = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    block_size = max(15, int(min(gray.shape) * 0.08))
    if block_size % 2 == 0:
        block_size += 1

    adapt_bin = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        10,
    )

    # Combined: pixel is metal (255) only if both global Otsu and local adaptive agree
    binary = cv2.bitwise_and(otsu_bin, adapt_bin)

    # Morphological clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Ensure correct polarity: metal surface (brighter) must be 255 (white), cutouts/background must be 0 (black)
    if np.any(cleaned_mask == 255) and np.any(cleaned_mask == 0):
        white_intensity = float(np.mean(gray[cleaned_mask == 255]))
        black_intensity = float(np.mean(gray[cleaned_mask == 0]))
        if white_intensity < black_intensity:
            cleaned_mask = cv2.bitwise_not(cleaned_mask)

    return cleaned_mask


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 coordinates clockwise: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left has smallest sum
    rect[2] = pts[np.argmax(s)]  # bottom-right has largest sum

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right has smallest difference
    rect[3] = pts[np.argmax(diff)]  # bottom-left has largest difference
    return rect


def four_point_perspective_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Performs perspective transform on quadrilateral defined by 4 points.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_w = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_h = max(int(height_a), int(height_b))

    if max_w <= 10 or max_h <= 10:
        return image

    dst = np.array(
        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
        dtype=np.float32,
    )

    m = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, m, (max_w, max_h))
    return warped


def enhance_photo_contrast_and_sharpness(
    image_bgr: np.ndarray,
    sharpness_boost: float = 1.4,
    clahe_clip: float = 2.5,
) -> np.ndarray:
    """
    Enhances contrast and sharpness for blurry, low-resolution, or low-contrast customer photos.
    - Uses bilateral filtering to suppress sensor noise & JPEG block artifacts while preserving edges
    - Uses CLAHE on LAB L-channel to reveal faint cut-out apertures under shadows/glare
    - Uses unsharp masking to crispen fine edges, lattices, and small apertures.
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    # 1. Bilateral filter to smooth noise while preserving hard boundaries
    filtered = cv2.bilateralFilter(image_bgr, d=5, sigmaColor=35, sigmaSpace=35)

    # 2. CLAHE on L (luminance) channel in LAB space
    lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_chan)
    lab_eq = cv2.merge([l_eq, a_chan, b_chan])
    enhanced_bgr = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # 3. Unsharp masking to crisp up soft boundaries
    if sharpness_boost > 0.0:
        gauss = cv2.GaussianBlur(enhanced_bgr, (0, 0), sigmaX=1.5)
        sharpened = cv2.addWeighted(
            enhanced_bgr,
            1.0 + sharpness_boost,
            gauss,
            -sharpness_boost,
            0,
        )
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    return enhanced_bgr


def detect_stencil_polarity(mask: np.ndarray, border_ratio: float = 0.04) -> bool:
    """
    Determines whether a binary mask has standard polarity (solid metal = 255, voids = 0)
    or inverted polarity (such as black-painted gates photographed against bright sky/wall).

    Physical manufacturing principle:
    All laser-cut panels require a continuous outer boundary perimeter frame to maintain structural
    rigidity. Thus, the outer perimeter strip (border_ratio margin) is solidly metal.

    Returns:
        True: standard polarity (solid metal is 255/white)
        False: inverted polarity (metal was detected as 0/black, holes as 255/white)
    """
    if mask is None or mask.size == 0:
        return True

    h, w = mask.shape[:2]
    bh = max(2, int(h * border_ratio))
    bw = max(2, int(w * border_ratio))

    top = mask[:bh, :]
    bottom = mask[-bh:, :]
    left = mask[:, :bw]
    right = mask[:, -bw:]

    border_pixels = np.concatenate([top.ravel(), bottom.ravel(), left.ravel(), right.ravel()])
    pct_white = float(np.mean(border_pixels == 255))

    # If the perimeter is less than 40% white, the metal is almost certainly black (inverted)
    return pct_white >= 0.40


def detect_sheet_quadrilateral(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Detects the 4-corner boundary of a metal sheet or panel in real-world customer photos:
    - Raw fabricated metal sheets placed on cardboard or workshop floor (chroma/LAB separation)
    - Panels with reflections/glare and ambient shadows (CLAHE luminance segmentation)
    - Installed panels and gates on walls or frames (gradient edge candidate generator)
    Validates geometric convexity, interior angles (45°-135°), aspect ratio, and edge alignment.
    Returns ordered 4 corners (TL, TR, BR, BL) as float32 np.ndarray, or None if not found.
    """
    if image_bgr is None or image_bgr.size == 0:
        return None

    h, w = image_bgr.shape[:2]
    if h < 20 or w < 20:
        return None

    img_area = float(h * w)

    # Prepare color spaces and gradients
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr.copy()
    edges = cv2.Canny(gray, 40, 130)

    # Multi-strategy binary masks for candidate generation
    masks = []

    # Strategy A: LAB b* channel (separates warm cardboard/wood/floor from neutral/cool metal)
    if image_bgr.ndim == 3:
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        _, _, b_chan = cv2.split(lab)
        _, b_thresh1 = cv2.threshold(b_chan, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        masks.append(b_thresh1)
        _, b_thresh2 = cv2.threshold(b_chan, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks.append(b_thresh2)

    # Strategy B: CLAHE on luminance + Otsu (handles severe lighting gradients and metallic glare)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    _, l_thresh = cv2.threshold(gray_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks.append(l_thresh)
    masks.append(cv2.bitwise_not(l_thresh))

    # Strategy C: Standard smoothed Otsu
    blurred_g = cv2.GaussianBlur(gray, (5, 5), 0)
    _, g_thresh = cv2.threshold(blurred_g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks.append(g_thresh)
    masks.append(cv2.bitwise_not(g_thresh))

    # Strategy D: Dilated edge mask (captures perimeter frame of installed gates / dark panels)
    edge_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated_edges = cv2.dilate(edges, edge_k, iterations=2)
    masks.append(dilated_edges)

    # Morphological closing kernel to bridge internal cutouts
    k_size = max(11, int(min(h, w) * 0.025))
    if k_size % 2 == 0:
        k_size += 1
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))

    candidates = []

    for mask in masks:
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours[:4]:
            area = cv2.contourArea(cnt)
            if area < 0.08 * img_area or area > 0.98 * img_area:
                continue

            # Convex hull bridges concave apertures along the sheet boundary
            hull = cv2.convexHull(cnt)
            peri = cv2.arcLength(hull, True)
            if peri <= 0:
                continue

            for eps in np.linspace(0.012, 0.065, 20):
                approx = cv2.approxPolyDP(hull, eps * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    pts = approx.reshape(4, 2).astype(np.float32)

                    # 1. Check interior angles (must be between 45° and 135°)
                    angles = []
                    valid_angles = True
                    for i in range(4):
                        p_prev = pts[(i - 1) % 4]
                        p_curr = pts[i]
                        p_next = pts[(i + 1) % 4]
                        v1 = p_prev - p_curr
                        v2 = p_next - p_curr
                        denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6
                        cos_ang = np.dot(v1, v2) / denom
                        ang = np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))
                        if ang < 45.0 or ang > 135.0:
                            valid_angles = False
                            break
                        angles.append(ang)

                    if not valid_angles:
                        continue

                    # 2. Check aspect ratio
                    w1 = np.linalg.norm(pts[1] - pts[0])
                    w2 = np.linalg.norm(pts[2] - pts[3])
                    h1 = np.linalg.norm(pts[3] - pts[0])
                    h2 = np.linalg.norm(pts[2] - pts[1])
                    avg_w = (w1 + w2) / 2.0
                    avg_h = (h1 + h2) / 2.0
                    if avg_w < 15 or avg_h < 15:
                        continue
                    ar = avg_w / avg_h
                    if ar < 0.12 or ar > 8.0:
                        continue

                    # 3. Candidate score: composite of area, orthogonality, and edge alignment
                    ortho_score = 1.0 - sum(abs(a - 90.0) for a in angles) / (4.0 * 90.0)
                    area_ratio = area / img_area
                    # Penalize candidates that simply frame the entire photo (> 88% of image)
                    if area_ratio > 0.88:
                        area_score = 0.15
                    else:
                        area_score = min(1.0, area_ratio * 1.5)

                    # Sample along the 4 edges on edge_map to test physical boundary alignment
                    edge_hits = []
                    for i in range(4):
                        p1 = pts[i]
                        p2 = pts[(i + 1) % 4]
                        length = int(np.linalg.norm(p2 - p1))
                        if length > 0:
                            xs = np.linspace(p1[0], p2[0], length).astype(int)
                            ys = np.linspace(p1[1], p2[1], length).astype(int)
                            valid_coords = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
                            if np.any(valid_coords):
                                edge_hits.append(np.mean(edges[ys[valid_coords], xs[valid_coords]]) / 255.0)

                    mean_edge = np.mean(edge_hits) if edge_hits else 0.0

                    # Compute internal laser-cut edge concentration:
                    cand_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(cand_mask, [pts.astype(np.int32)], 255)
                    internal_edges = float(np.mean(edges[cand_mask == 255])) / 255.0 if np.any(cand_mask == 255) else 0.0

                    total_score = (
                        area_score * 0.25
                        + ortho_score * 0.25
                        + mean_edge * 0.25
                        + internal_edges * 0.25
                    )

                    candidates.append((total_score, pts))

    if not candidates:
        return None

    # Pick candidate with highest composite score
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_pts = candidates[0][1]
    return order_points(best_pts)


def convert_customer_crop_to_stencil_mask(
    crop_bgr: np.ndarray,
    mode: str = "auto",
    enhance: bool = True,
    threshold_offset: int = 0,
    invert: bool = False,
    trim_border_pct: float = 0.0,
) -> np.ndarray:
    """
    Transforms any customer photo crop into a normalized binary stencil mask.
    Handles:
    - Raw cut sheets on cardboard/workshop floor (mode="raw_sheet")
    - Installed painted gates, safety doors, panels with frame shadows (mode="installed_product")
    - Backlit silhouettes against daylight/outdoor sky (mode="backlit")
    - Mixed lighting with sky top & ground shadow bottom (mode="mixed_lighting")
    - Blurry / compressed WhatsApp mobile camera shots (mode="blurry")
    - General auto mode (mode="auto")
    """
    mask, _ = normalize_stencil(
        bgr=crop_bgr,
        crop=False,
        mode=mode,
        enhance=enhance,
        threshold_offset=threshold_offset,
        invert=invert,
        trim_border_pct=trim_border_pct,
    )
    return mask


def detect_cutout_subpanels(mask: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    Extracts multi-scale query sub-regions from a customer stencil mask:
    - ('Full', mask)
    - If tall (h / w >= 1.25), extracts Right Section, Center Section, Left Section, Top Half, Bottom Half
    - If wide (w / h >= 1.25), extracts Left Panel, Center Panel, Right Panel, Left Half, Right Half
    - Tightly bounded cutout region ('Aperture Focus') if large empty margins exist
    """
    if mask is None or mask.size == 0:
        return []
    h, w = mask.shape[:2]
    regions: List[Tuple[str, np.ndarray]] = [("Full", mask)]

    # Strip empty outer border if any
    inv = (mask == 0).astype(np.uint8)
    if inv.sum() >= 25:
        ys, xs = np.where(inv > 0)
        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())
        if (x_max - x_min < 0.85 * w) or (y_max - y_min < 0.85 * h):
            bb_crop = mask[y_min : y_max + 1, x_min : x_max + 1]
            if bb_crop.shape[0] >= 20 and bb_crop.shape[1] >= 20 and (bb_crop == 0).sum() >= 20:
                regions.append(("Aperture Focus", bb_crop))

    ar = float(h) / float(max(1, w))
    if ar >= 1.25:
        # Tall panel / safety door / column
        right = mask[:, int(w * 0.45):]
        left = mask[:, :int(w * 0.55)]
        center = mask[:, int(w * 0.15):int(w * 0.85)]
        top = mask[:int(h * 0.55), :]
        bot = mask[int(h * 0.45):, :]
        for lbl, sub in [
            ("Right Section", right),
            ("Left Section", left),
            ("Center Section", center),
            ("Top Half", top),
            ("Bottom Half", bot),
        ]:
            if (sub == 0).sum() >= 25 and sub.shape[0] >= 15 and sub.shape[1] >= 15:
                regions.append((lbl, sub))
    elif ar <= 0.80:
        # Wide panel / sliding gate / transom
        left_3rd = mask[:, :int(w * 0.40)]
        mid_3rd = mask[:, int(w * 0.30):int(w * 0.70)]
        right_3rd = mask[:, int(w * 0.60):]
        left = mask[:, :int(w * 0.55)]
        right = mask[:, int(w * 0.45):]
        for lbl, sub in [
            ("Left Panel", left_3rd),
            ("Center Panel", mid_3rd),
            ("Right Panel", right_3rd),
            ("Left Half", left),
            ("Right Half", right),
        ]:
            if (sub == 0).sum() >= 25 and sub.shape[0] >= 15 and sub.shape[1] >= 15:
                regions.append((lbl, sub))

    return regions


def auto_focus_cutout_box(image_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Finds the tightest bounding box of the decorative laser-cut pattern inside a photo or crop,
    ignoring blank outer margins, door frames, or wall backgrounds.
    Returns (x, y, w, h) in image pixel coordinates, or None if not found.
    """
    if image_bgr is None or image_bgr.size == 0:
        return None
    h, w = image_bgr.shape[:2]
    mask = convert_customer_crop_to_stencil_mask(image_bgr, mode="auto")
    cutouts = cv2.bitwise_not(mask)
    if (cutouts == 255).sum() < 30:
        return None

    # Find coordinates of all cutout pixels
    ys, xs = np.where(cutouts == 255)
    if len(xs) == 0:
        return None

    # Percentile bounds to reject rogue speckles
    x_min = int(np.percentile(xs, 1))
    x_max = int(np.percentile(xs, 99))
    y_min = int(np.percentile(ys, 1))
    y_max = int(np.percentile(ys, 99))

    # Add small 2% padding
    pad_x = max(4, int(w * 0.02))
    pad_y = max(4, int(h * 0.02))
    bx = max(0, x_min - pad_x)
    by = max(0, y_min - pad_y)
    bw = min(w - bx, (x_max - x_min) + 2 * pad_x)
    bh = min(h - by, (y_max - y_min) + 2 * pad_y)

    if bw < 20 or bh < 20:
        return None
    return (bx, by, bw, bh)


def convert_photo_to_stencil_mask(
    photo_bgr: np.ndarray,
    auto_perspective: bool = True,
    invert: bool = False,
    mode: str = "auto",
    enhance: bool = True,
    threshold_offset: int = 0,
    trim_border_pct: float = 0.0,
) -> np.ndarray:
    """
    Converts a customer photo into a normalized stencil mask:
    - Supports 'auto', 'raw_sheet', 'installed_product', 'backlit', and 'blurry' modes
    - Rectifies perspective from detected 4-corner sheet boundaries if requested
    - Enhances contrast & sharpens blurry/low-contrast photos
    - Enforces correct stencil polarity: solid metal = 255 (White), cutouts = 0 (Black)
    """
    if photo_bgr is None or photo_bgr.size == 0:
        return np.zeros((224, 224), dtype=np.uint8)

    processed = photo_bgr.copy()

    # Step 1: Perspective rectification if requested
    if auto_perspective:
        quad = detect_sheet_quadrilateral(processed)
        if quad is not None:
            processed = four_point_perspective_transform(processed, quad)

    # Step 2: Convert to stencil mask
    return convert_customer_crop_to_stencil_mask(
        processed,
        mode=mode,
        enhance=enhance,
        threshold_offset=threshold_offset,
        invert=invert,
        trim_border_pct=trim_border_pct,
    )


def letterbox_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    pad_value: int = 0,
) -> np.ndarray:
    """
    Resizes an image preserving aspect ratio to fit within target_size (W, H),
    centering the image and padding remaining space with pad_value.
    """
    target_w, target_h = target_size
    h, w = image.shape[:2]

    if h == 0 or w == 0:
        if image.ndim == 2:
            return np.full((target_h, target_w), pad_value, dtype=np.uint8)
        return np.full((target_h, target_w, image.shape[2]), pad_value, dtype=np.uint8)

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    if image.ndim == 2 and set(np.unique(image)).issubset({0, 255}):
        _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

    if image.ndim == 2:
        canvas = np.full((target_h, target_w), pad_value, dtype=np.uint8)
    else:
        channels = image.shape[2]
        canvas = np.full((target_h, target_w, channels), pad_value, dtype=np.uint8)

    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2

    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
    return canvas


def prepare_mask_for_model(mask_224: np.ndarray) -> np.ndarray:
    """
    Prepares a (224, 224) binary mask for DINOv2 input:
    - Converts 1-channel mask to 3-channel RGB float32 [0.0, 1.0]
    - Normalizes with ImageNet mean and std as float32 tensor (3, 224, 224).
    """
    if mask_224.ndim == 2:
        rgb = cv2.cvtColor(mask_224, cv2.COLOR_GRAY2RGB)
    else:
        rgb = mask_224.copy()

    float_img = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (float_img - mean) / std

    chw = np.transpose(normalized, (2, 0, 1))
    return chw


def main() -> None:
    """CLI runner for normalize_stencil: python preprocess.py input.jpg output.png"""
    import sys
    from pathlib import Path

    if len(sys.argv) < 3:
        print("Usage: python preprocess.py input.jpg output.png", file=sys.stderr)
        sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]
    try:
        data = np.fromfile(src, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        img = None
    if img is None:
        img = cv2.imread(src, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"Cannot read {src}")
    m, info = normalize_stencil(img)
    ext = Path(dst).suffix
    success, enc = cv2.imencode(ext if ext else ".png", m)
    if success:
        enc.tofile(dst)
    else:
        cv2.imwrite(dst, m)
    print(info, "metal %.1f%%" % (100 * (m == 255).mean()))


if __name__ == "__main__":
    main()
