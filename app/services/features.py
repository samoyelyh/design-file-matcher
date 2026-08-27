from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from app.config import settings

from .media import open_visual


class ImageFeatureProvider(Protocol):
    device: str

    def encode(self, paths: Sequence[Path]) -> np.ndarray: ...


class DinoFeatureProvider:
    """Configurable DINOv3/DINOv2 CLS embedding provider with safe fallback."""

    model_ids = {
        "dinov3": lambda: settings.dinov3_model,
        "dinov2": lambda: settings.dinov2_model,
    }

    def __init__(self, backend: str | None = None) -> None:
        # Set this before importing transformers/huggingface_hub so all model
        # metadata and weight requests use the configured domestic mirror.
        os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint)
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "未安装 DINO 依赖。请执行: pip install -e .[dino]"
            ) from exc

        requested = (backend or settings.embedding_backend).casefold()
        if requested not in self.model_ids:
            raise ValueError("EMBEDDING_BACKEND 只支持 dinov3 或 dinov2")
        attempts = [requested]
        if (
            requested == "dinov3"
            and settings.dino_allow_fallback
            and "dinov2" not in attempts
        ):
            attempts.append("dinov2")

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.requested_backend = requested
        self.load_warning: str | None = None
        errors: list[str] = []
        for candidate in attempts:
            model_name = self.model_ids[candidate]()
            model_source = model_name
            local_only = False
            if candidate == "dinov2":
                local_dir = Path(settings.dinov2_local_dir)
                required = {
                    "config.json",
                    "preprocessor_config.json",
                    "model.safetensors",
                }
                if local_dir.is_dir() and all(
                    (local_dir / filename).is_file() for filename in required
                ):
                    model_source = str(local_dir)
                    local_only = True
            try:
                self.processor = AutoImageProcessor.from_pretrained(
                    model_source,
                    local_files_only=local_only,
                    use_fast=True,
                )
                self.model = AutoModel.from_pretrained(
                    model_source,
                    use_safetensors=True,
                    local_files_only=local_only,
                ).to(self.device)
                self.model.eval()
                self.backend = candidate
                self.model_name = model_name
                self.feature_version = f"{candidate}_cls_cosine_v1"
                self.embedding_dim = int(self.model.config.hidden_size)
                if candidate != requested:
                    self.load_warning = (
                        f"{requested} 加载失败，已自动回退 {candidate}"
                    )
                break
            except Exception as exc:
                errors.append(f"{candidate}: {type(exc).__name__}: {str(exc).splitlines()[0]}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            detail = "；".join(errors)
            raise RuntimeError(f"DINO 模型加载失败：{detail}")

    def encode(self, paths: Sequence[Path]) -> np.ndarray:
        images = [open_visual(Path(path)) for path in paths]
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
            features = getattr(outputs, "pooler_output", None)
            if features is None:
                features = outputs.last_hidden_state[:, 0]
            features = features / features.norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)
        return features.float().cpu().numpy()

    def empty_cache(self) -> None:
        if self.device == "cuda":
            self.torch.cuda.empty_cache()


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
        self.model_name = f"{settings.clip_model}:{settings.clip_pretrained}"
        self.embedding_dim = int(getattr(self.model.visual, "output_dim", 0)) or None

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
