"""
DINOv2-small embedding engine using PyTorch and HuggingFace Transformers on CPU.
Produces 384-dimensional L2-normalized vector embeddings for 224x224 letterboxed stencil masks.
Supports offline execution and local model caching.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch

try:
    from transformers import AutoImageProcessor, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    AutoImageProcessor = None  # type: ignore
    AutoModel = None  # type: ignore
    HAS_TRANSFORMERS = False

from app.core.preprocess import letterbox_image, prepare_mask_for_model


logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "facebook/dinov2-small"
EMBEDDING_DIM = 384


def get_default_model_cache_dir() -> Path:
    """Returns local writable path for caching HuggingFace models."""
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path.home() / ".cache"
    cache_path = base / "design_matcher" / "models"
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def find_bundled_model_dir() -> Optional[Path]:
    """Locates pre-packaged DINOv2 model directory in PyInstaller bundles or project."""
    candidates = []
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        meipass = Path(sys._MEIPASS)
        candidates.append(meipass / "models" / "dinov2-small")
        candidates.append(meipass / "dinov2-small")
    if getattr(sys, "frozen", False):
        exe_parent = Path(sys.executable).resolve().parent
        candidates.append(exe_parent / "models" / "dinov2-small")
        candidates.append(exe_parent / "_internal" / "models" / "dinov2-small")

    # Local repo candidate
    repo_models = Path(__file__).resolve().parent.parent.parent.parent / "models" / "dinov2-small"
    candidates.append(repo_models)

    for c in candidates:
        if c.is_dir() and (c / "config.json").exists():
            return c
    return None


class DINOv2Embedder:
    """Embeds binary stencil masks using facebook/dinov2-small on CPU."""

    _instance: Optional["DINOv2Embedder"] = None

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL_NAME,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.device = torch.device("cpu")
        self.cache_dir = Path(cache_dir) if cache_dir else get_default_model_cache_dir()
        self.model_name_or_path = model_name_or_path
        self._model: Optional[AutoModel] = None
        self._processor: Optional[AutoImageProcessor] = None

    @classmethod
    def get_instance(cls) -> "DINOv2Embedder":
        """Singleton accessor to reuse model in memory."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self) -> None:
        """Loads model and processor if not already loaded."""
        if self._model is not None:
            return

        global AutoImageProcessor, AutoModel, HAS_TRANSFORMERS
        if not HAS_TRANSFORMERS or AutoImageProcessor is None or AutoModel is None:
            try:
                from transformers import AutoImageProcessor as _AIP, AutoModel as _AM
                AutoImageProcessor = _AIP
                AutoModel = _AM
                HAS_TRANSFORMERS = True
            except ImportError as exc:
                logger.error("HuggingFace 'transformers' is not installed: %s", exc)
                raise RuntimeError(
                    "HuggingFace 'transformers' library is required for the DINOv2 embedding model but is not installed.\n"
                    "Please install it using: pip install -r requirements.txt\n"
                    "Or rebuild the application with package_windows.bat."
                ) from exc

        # 1. Check for bundled offline model weights first
        bundled_dir = find_bundled_model_dir()
        if bundled_dir:
            logger.info("Loading bundled offline DINOv2 model from: %s", bundled_dir)
            try:
                self._processor = AutoImageProcessor.from_pretrained(
                    str(bundled_dir),
                    local_files_only=True,
                )
                self._model = AutoModel.from_pretrained(
                    str(bundled_dir),
                    local_files_only=True,
                )
                self._model.to(self.device)
                self._model.eval()
                logger.info("Bundled DINOv2 model successfully initialized.")
                return
            except Exception as exc_bundled:
                logger.warning("Failed loading bundled model from %s (%s). Falling back...", bundled_dir, exc_bundled)

        # 2. Check local user cache directory
        local_dir = self.cache_dir / "dinov2-small"
        target_path = str(local_dir) if (local_dir.is_dir() and (local_dir / "config.json").exists()) else self.model_name_or_path

        logger.info("Loading DINOv2 model from '%s' (cache: %s)...", target_path, self.cache_dir)
        try:
            self._processor = AutoImageProcessor.from_pretrained(
                target_path,
                cache_dir=str(self.cache_dir),
                local_files_only=local_dir.is_dir(),
            )
            self._model = AutoModel.from_pretrained(
                target_path,
                cache_dir=str(self.cache_dir),
                local_files_only=local_dir.is_dir(),
            )
        except Exception:
            # Fall back to online download if local files only failed
            logger.info("Downloading DINOv2 model from Hugging Face hub...")
            self._processor = AutoImageProcessor.from_pretrained(
                self.model_name_or_path,
                cache_dir=str(self.cache_dir),
            )
            self._model = AutoModel.from_pretrained(
                self.model_name_or_path,
                cache_dir=str(self.cache_dir),
            )

        self._model.to(self.device)
        self._model.eval()
        logger.info("DINOv2 model ready on CPU.")

    def embed_masks(
        self,
        masks: List[np.ndarray],
        batch_size: int = 16,
    ) -> np.ndarray:
        """
        Embeds a list of binary masks (any shape, will be letterboxed to 224x224).

        Args:
            masks: List of uint8 mask arrays.
            batch_size: Batch size for inference on CPU.

        Returns:
            numpy.ndarray: L2-normalized float32 array of shape (N, 384).
        """
        if not masks:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        self._load_model()
        all_embeddings: List[np.ndarray] = []

        # Process in batches to maintain predictable memory footprint
        for i in range(0, len(masks), batch_size):
            batch_masks = masks[i : i + batch_size]
            tensor_list = []

            for m in batch_masks:
                # Ensure mask is letterboxed to 224x224
                if m.shape[:2] != (224, 224):
                    m_224 = letterbox_image(m, target_size=(224, 224), pad_value=0)
                else:
                    m_224 = m

                # Prepare CHW normalized float32 tensor
                chw = prepare_mask_for_model(m_224)
                tensor_list.append(torch.from_numpy(chw))

            batch_tensor = torch.stack(tensor_list).to(self.device)

            with torch.no_grad():
                outputs = self._model(pixel_values=batch_tensor)
                # DINOv2 CLS token is at position 0 in last_hidden_state
                cls_tokens = outputs.last_hidden_state[:, 0, :]
                # L2 normalize so inner product equals cosine similarity
                normalized = torch.nn.functional.normalize(cls_tokens, p=2, dim=1)
                all_embeddings.append(normalized.cpu().numpy().astype(np.float32))

        return np.vstack(all_embeddings)

    def embed_single_mask(self, mask: np.ndarray) -> np.ndarray:
        """Embeds a single mask array. Returns (384,) float32 vector."""
        vectors = self.embed_masks([mask], batch_size=1)
        return vectors[0]
