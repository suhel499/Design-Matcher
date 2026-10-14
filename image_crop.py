"""
Root shim for image cropping and warping.
"""

from __future__ import annotations

import sys
from pathlib import Path

dm_dir = Path(__file__).resolve().parent / "design_matcher"
if str(dm_dir) not in sys.path:
    sys.path.insert(0, str(dm_dir))

from app.core.image_crop import (  # noqa: F401
    crop_rect,
    image_to_view_coords,
    load_image_with_exif,
    order_points,
    sharpen,
    view_to_image_coords,
    warp_four_points,
)
