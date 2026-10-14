"""
Canonical mask preparation and embedding feature extraction.
Produces standardized 256x256 binary masks and L2-normalized float32 vectors.
Supports Classical (self-contained, zero-dependency), Torch, and ONNX embedders.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from .config import config, EngineConfig

logger = logging.getLogger(__name__)


def canonical_mask(
    mask: np.ndarray,
    target_size: int = config.MASK_SIZE,
) -> Tuple[np.ndarray, float]:
    """
    Standardizes a binary stencil mask into a canonical representation:
    1. Crops to the metal bounding box (outer boundary of solid metal).
    2. Resizes the longer side to target_size (256) with INTER_AREA keeping aspect ratio.
    3. Pads to target_size x target_size with 255 (metal).
    4. Re-binarizes at threshold 127.

    Returns:
        (canonical_mask, aspect_ratio):
            canonical_mask: (target_size, target_size) uint8 array, 255=metal, 0=cutouts.
            aspect_ratio: float (width / height of the original unpadded crop).
    """
    if mask is None or mask.size == 0:
        return np.full((target_size, target_size), 255, dtype=np.uint8), 1.0

    # Ensure single channel uint8
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = (mask > 127).astype(np.uint8) * 255

    # Find metal bounding box (non-zero pixels)
    metal_pts = np.argwhere(mask == 255)
    if len(metal_pts) < 10:
        # Fallback: if inverted or no white pixels, look for any non-zero
        metal_pts = np.argwhere(mask > 0)
    if len(metal_pts) < 10:
        # Entirely black mask
        return np.full((target_size, target_size), 255, dtype=np.uint8), 1.0

    ymin, xmin = metal_pts.min(axis=0)
    ymax, xmax = metal_pts.max(axis=0)

    crop = mask[ymin : ymax + 1, xmin : xmax + 1]
    crop_h, crop_w = crop.shape[:2]

    aspect_ratio = float(crop_w) / float(max(1, crop_h))

    # Resize longer side to target_size
    max_side = max(crop_w, crop_h)
    scale = float(target_size) / float(max(1, max_side))
    new_w = max(1, min(target_size, int(round(crop_w * scale))))
    new_h = max(1, min(target_size, int(round(crop_h * scale))))

    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pad with 255 (metal)
    padded = np.full((target_size, target_size), 255, dtype=np.uint8)
    pad_y = (target_size - new_h) // 2
    pad_x = (target_size - new_w) // 2
    padded[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    # Re-binarize cleanly
    _, canonical = cv2.threshold(padded, 127, 255, cv2.THRESH_BINARY)
    return canonical, aspect_ratio


class BaseEmbedder(ABC):
    """Abstract interface for mask embedders."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns feature vector dimension."""
        pass

    @abstractmethod
    def embed_batch(
        self,
        masks: List[np.ndarray],
        aspect_ratios: Optional[List[float]] = None,
    ) -> np.ndarray:
        """
        Embeds a batch of canonical 256x256 masks.
        Returns: np.ndarray of shape (N, dimension) dtype float32, L2-normalized.
        """
        pass

    def embed(self, mask: np.ndarray, aspect_ratio: float = 1.0) -> np.ndarray:
        """Embeds a single canonical mask. Returns float32 vector of shape (dimension,)."""
        return self.embed_batch([mask], [aspect_ratio])[0]


class ClassicalEmbedder(BaseEmbedder):
    """
    Self-contained, fast, and offline classical embedder.
    Requires no neural network weights or external models:
    - Downscaled 32x32 spatial mask (1024-d)
    - Normalized distance transform spatial grid 16x16 (256-d)
    - Appends aspect ratio & metal fill ratio (2-d)
    Total dimension: 1282.
    """

    @property
    def dimension(self) -> int:
        return 1282

    def embed_batch(
        self,
        masks: List[np.ndarray],
        aspect_ratios: Optional[List[float]] = None,
    ) -> np.ndarray:
        if not masks:
            return np.empty((0, self.dimension), dtype=np.float32)

        if aspect_ratios is None:
            aspect_ratios = [1.0] * len(masks)

        features: List[np.ndarray] = []
        for mask, ar in zip(masks, aspect_ratios):
            # Downscale 32x32
            d32 = cv2.resize(mask, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            spatial_vec = d32.flatten()

            # Distance transform of cutouts (distance from holes to metal)
            cutouts = (mask == 0).astype(np.uint8)
            dt = cv2.distanceTransform(cutouts, cv2.DIST_L2, 5)
            dt_max = float(dt.max())
            if dt_max > 0:
                dt = dt / dt_max
            dt16 = cv2.resize(dt, (16, 16), interpolation=cv2.INTER_AREA).flatten()

            # Fill ratio (fraction of white metal)
            fill_ratio = float((mask == 255).mean())

            # Combined vector with weighted auxiliary features
            ar_feat = float(np.clip(ar / 5.0, 0.0, 2.0))
            aux = np.array([ar_feat, fill_ratio], dtype=np.float32)

            vec = np.concatenate([spatial_vec, dt16, aux]).astype(np.float32)
            # L2 normalization
            norm = float(np.linalg.norm(vec))
            if norm > 1e-12:
                vec = vec / norm
            features.append(vec)

        return np.stack(features, axis=0)


class TorchEmbedder(BaseEmbedder):
    """
    PyTorch-based image encoder (ResNet backbone) run on CPU in eval mode.
    Outputs L2-normalized deep representations + aspect & fill features.
    """

    def __init__(self, model_name: str = "resnet18") -> None:
        import torch
        import torchvision.models as models

        self._torch = torch
        self._device = torch.device("cpu")

        if model_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if hasattr(models, "ResNet50_Weights") else None
            backbone = models.resnet50(weights=weights)
            self._dim = 2048 + 2
        else:
            weights = models.ResNet18_Weights.DEFAULT if hasattr(models, "ResNet18_Weights") else None
            backbone = models.resnet18(weights=weights)
            self._dim = 512 + 2

        backbone.fc = torch.nn.Identity()  # type: ignore[assignment]
        backbone.eval()
        self.model = backbone.to(self._device)

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_batch(
        self,
        masks: List[np.ndarray],
        aspect_ratios: Optional[List[float]] = None,
    ) -> np.ndarray:
        if not masks:
            return np.empty((0, self.dimension), dtype=np.float32)

        if aspect_ratios is None:
            aspect_ratios = [1.0] * len(masks)

        batch_tensors: List[self._torch.Tensor] = []
        aux_features: List[List[float]] = []

        for mask, ar in zip(masks, aspect_ratios):
            # Convert 1-ch mask to 3-ch float32 [0, 1] normalized with ImageNet stats
            rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
            # ImageNet mean & std
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            norm_rgb = (rgb - mean) / std
            tensor = self._torch.from_numpy(norm_rgb.transpose(2, 0, 1))
            batch_tensors.append(tensor)

            fill_ratio = float((mask == 255).mean())
            ar_feat = float(np.clip(ar / 5.0, 0.0, 2.0))
            aux_features.append([ar_feat, fill_ratio])

        x = self._torch.stack(batch_tensors, dim=0).to(self._device)
        with self._torch.no_grad():
            embeddings = self.model(x).cpu().numpy().astype(np.float32)

        aux_arr = np.array(aux_features, dtype=np.float32)
        combined = np.concatenate([embeddings, aux_arr], axis=1)

        # L2 normalize
        norms = np.linalg.norm(combined, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return (combined / norms).astype(np.float32)


def get_default_embedder(
    embedder_type: Optional[str] = None,
    cfg: EngineConfig = config,
) -> BaseEmbedder:
    """
    Factory function returning the configured embedder.
    Defaults safely to ClassicalEmbedder for instant zero-dependency execution.
    """
    kind = (embedder_type or cfg.EMBEDDER_TYPE).lower()

    if kind in ("torch", "resnet", "resnet18"):
        try:
            return TorchEmbedder("resnet18")
        except Exception as exc:
            logger.warning("TorchEmbedder unavailable (%s). Falling back to ClassicalEmbedder.", exc)
            return ClassicalEmbedder()

    # Default robust classical embedder
    return ClassicalEmbedder()
