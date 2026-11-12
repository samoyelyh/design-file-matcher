from __future__ import annotations

import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.config import settings
from app.models import Batch, MainImage, MaskProfile, MatchedFile

from .cropper import MaskedCropProcessor
from .masks import MAIN_IMAGE_CATEGORY
from .references import ProductEvidence, ProductScoreMatrix


DINOV2_MODEL = "DINOV2"
DINOV3_MODEL = "DINOV3"
DINO_MODELS = {DINOV2_MODEL, DINOV3_MODEL}


def _model_spec(model_key: str) -> tuple[str, str, str]:
    if model_key == DINOV2_MODEL:
        return settings.dinov2_model_id, settings.dinov2_model_dir, "DINOv2 Base ViT-B/14（GPU）"
    if model_key == DINOV3_MODEL:
        return settings.dinov3_model_id, settings.dinov3_model_dir, "DINOv3 ViT-B/16（GPU）"
    raise ValueError(f"不支持的 DINO 模型: {model_key}")


@lru_cache(maxsize=2)
def model_status(model_key: str = DINOV3_MODEL) -> dict:
    model_id, configured_dir, label = _model_spec(model_key)
    model_dir = Path(configured_dir)
    config_ready = (model_dir / "config.json").is_file()
    weights_ready = any(model_dir.glob("*.safetensors")) or any(model_dir.glob("pytorch_model*.bin"))
    processor_ready = (model_dir / "preprocessor_config.json").is_file()
    dependency_ready = True
    dependency_error = None
    cuda_available = False
    device_name = "CPU"
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel  # noqa: F401

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover - host-specific
        dependency_ready = False
        dependency_error = str(exc)
    installed = config_ready and weights_ready and processor_ready and dependency_ready
    return {
        "id": model_key,
        "label": label,
        "installed": installed,
        "selectable": installed and cuda_available,
        "model_id": model_id,
        "model_dir": str(model_dir),
        "cache_dir": settings.dinov3_cache_dir,
        "cuda_available": cuda_available,
        "device": f"CUDA / {device_name}" if cuda_available else device_name,
        "dependency_ready": dependency_ready,
        "error": dependency_error,
        "requires_license_acceptance": model_key == DINOV3_MODEL and not weights_ready,
    }


def require_model_ready(model_key: str = DINOV3_MODEL) -> dict:
    status = model_status(model_key)
    if status["selectable"]:
        return status
    if not status["installed"]:
        if model_key == DINOV3_MODEL:
            raise ValueError(f"DINOv3 权重尚未安装或无访问权限：{status['model_dir']}")
        raise ValueError(f"DINOv2 权重不完整：{status['model_dir']}")
    raise ValueError(f"{status['label']} 需要可用的 NVIDIA CUDA 显卡")


class DINOEncoder:
    """Lazy local-only DINO encoder; matching never downloads weights."""

    _lock = threading.Lock()
    _loaded: dict[str, tuple[object, object, object]] = {}

    def __init__(self, model_key: str) -> None:
        if model_key not in DINO_MODELS:
            raise ValueError(f"不支持的 DINO 模型: {model_key}")
        self.model_key = model_key

    def _load(self):
        require_model_ready(self.model_key)
        with self._lock:
            if self.model_key not in self._loaded:
                import torch
                from transformers import AutoImageProcessor, AutoModel

                _model_id, model_dir, _label = _model_spec(self.model_key)
                processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True, use_fast=True)
                model = AutoModel.from_pretrained(
                    model_dir,
                    local_files_only=True,
                    dtype=torch.float16,
                ).eval().to("cuda")
                self._loaded[self.model_key] = processor, model, torch.device("cuda")
        return self._loaded[self.model_key]

    def encode_paths(self, paths: list[str | Path]) -> np.ndarray:
        if not paths:
            return np.empty((0, 0), dtype=np.float32)
        import torch
        from PIL import Image

        from .media import open_visual

        processor, model, device = self._load()
        rows: list[np.ndarray] = []
        batch_size = max(1, settings.dinov3_batch_size)
        for start in range(0, len(paths), batch_size):
            images: list[Image.Image] = []
            try:
                images = [open_visual(Path(path)).convert("RGB") for path in paths[start:start + batch_size]]
                inputs = processor(images=images, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(device, non_blocking=True)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    output = model(pixel_values=pixel_values)
                    vectors = getattr(output, "pooler_output", None)
                    if vectors is None:
                        vectors = output.last_hidden_state[:, 0]
                    vectors = torch.nn.functional.normalize(vectors.float(), dim=-1)
                rows.append(vectors.cpu().numpy())
            finally:
                for image in images:
                    image.close()
        return np.concatenate(rows, axis=0).astype(np.float32, copy=False)


def build_dino_product_matrices(
    db,
    batch: Batch,
    mains: list[MainImage],
    files_by_category: dict[str, list[MatchedFile]],
    profiles: dict[str, MaskProfile],
    model_key: str,
) -> tuple[dict[str, np.ndarray], dict[str, ProductScoreMatrix]]:
    """Build each full Query x Product cosine matrix from manual masked crops."""
    processor = MaskedCropProcessor()
    main_assets = [processor.process_main(db, batch, main, profiles[MAIN_IMAGE_CATEGORY]).crop for main in mains]
    encoder = DINOEncoder(model_key)
    main_vectors = encoder.encode_paths([asset.asset_path for asset in main_assets])
    matrices: dict[str, np.ndarray] = {}
    scores: dict[str, ProductScoreMatrix] = {}
    for category, files in files_by_category.items():
        query_assets = [processor.process_file(db, batch, item, profiles[category]).crop for item in files]
        query_vectors = encoder.encode_paths([asset.asset_path for asset in query_assets])
        matrix = np.clip(query_vectors @ main_vectors.T, 0.0, 1.0).astype(np.float32)
        for item in files:
            item.match_method = f"{model_key}_MANUAL_MASK"
            item.phash_best_region_type = None
            item.phash_distance = None
        evidence = [[ProductEvidence(
            aggregated_score=float(matrix[row, column]),
            best_distance=None,
            second_distance=None,
            third_distance=None,
            support_count=1,
            reference_id=None,
            reference_category=None,
        ) for column in range(len(mains))] for row in range(len(files))]
        matrices[category] = matrix
        scores[category] = ProductScoreMatrix(matrix=matrix, evidence=evidence)
    return matrices, scores


def build_dinov3_product_matrices(*args, **kwargs):
    return build_dino_product_matrices(*args, **kwargs, model_key=DINOV3_MODEL)
