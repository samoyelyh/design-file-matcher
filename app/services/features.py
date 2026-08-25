from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from app.config import settings

from .media import open_visual


class ImageFeatureProvider(Protocol):
    device: str

    def encode(self, paths: Sequence[Path]) -> np.ndarray: ...


class OpenClipFeatureProvider:
    """Versioned OpenCLIP provider, independent from assignment policy."""

    feature_version = "openclip_masked_v1"

    def __init__(self) -> None:
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "未安装 OpenCLIP。请执行: pip install -e .[clip]"
            ) from exc
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        pretrained = settings.clip_pretrained
        transform_options = {}
        if pretrained == "openai":
            pretrained_cfg = dict(open_clip.get_pretrained_cfg(settings.clip_model, pretrained))
            checkpoint = open_clip.download_pretrained(
                pretrained_cfg,
                prefer_hf_hub=False,
            )
            pretrained = checkpoint
            transform_options = {
                "force_quick_gelu": bool(pretrained_cfg.get("quick_gelu", False)),
                "image_mean": pretrained_cfg.get("mean"),
                "image_std": pretrained_cfg.get("std"),
                "image_interpolation": pretrained_cfg.get("interpolation"),
                "image_resize_mode": pretrained_cfg.get("resize_mode"),
                "weights_only": False,
            }
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            settings.clip_model,
            pretrained=pretrained,
            device=self.device,
            **transform_options,
        )
        self.model.eval()

    def encode(self, paths: Sequence[Path]) -> np.ndarray:
        images = [self.preprocess(open_visual(path)) for path in paths]
        tensor = self.torch.stack(images).to(self.device)
        with self.torch.inference_mode():
            features = self.model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return features.float().cpu().numpy()


class SimilarityMatrixBuilder:
    def build(self, query_vectors: np.ndarray, main_vectors: np.ndarray) -> np.ndarray:
        query = np.asarray(query_vectors, dtype=np.float32)
        main = np.asarray(main_vectors, dtype=np.float32)
        query /= np.clip(np.linalg.norm(query, axis=1, keepdims=True), 1e-12, None)
        main /= np.clip(np.linalg.norm(main, axis=1, keepdims=True), 1e-12, None)
        return query @ main.T
