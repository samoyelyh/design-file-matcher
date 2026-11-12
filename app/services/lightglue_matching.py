from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock

import numpy as np

from app.config import settings
from app.models import Batch

from .media import open_visual
from .phash import source_fingerprint


@dataclass(frozen=True)
class LocalMatchResult:
    raw_match_count: int
    ransac_inlier_count: int
    inlier_ratio: float
    mean_match_confidence: float
    coverage_score: float
    local_score: float

    def to_dict(self) -> dict:
        return asdict(self)


_runtime = None
_runtime_lock = Lock()


def dependency_status() -> dict:
    try:
        import cv2  # noqa: F401
        import torch
        from lightglue import ALIKED, LightGlue  # noqa: F401
    except Exception as exc:
        return {
            "installed": False,
            "selectable": False,
            "error": str(exc),
            "device": "未安装",
        }
    checkpoints = Path(settings.lightglue_model_dir) / "checkpoints"
    ready = (
        (checkpoints / "aliked-n16.pth").is_file()
        and (checkpoints / "aliked_lightglue_v0-1_arxiv.pth").is_file()
    )
    return {
        "installed": True,
        "selectable": ready,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "model_dir": settings.lightglue_model_dir,
        "extractor": settings.lightglue_extractor,
        "weights_ready": ready,
    }


class LightGlueRuntime:
    def __init__(self) -> None:
        import torch
        from lightglue import ALIKED, DISK, SIFT, LightGlue

        Path(settings.lightglue_model_dir).mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(settings.lightglue_model_dir)
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        extractor_name = settings.lightglue_extractor.upper()
        extractors = {
            "ALIKED": lambda: ALIKED(max_num_keypoints=settings.lightglue_max_keypoints),
            "DISK": lambda: DISK(max_num_keypoints=settings.lightglue_max_keypoints),
            "SIFT": lambda: SIFT(max_num_keypoints=settings.lightglue_max_keypoints),
        }
        if extractor_name not in extractors:
            raise ValueError(f"不支持的 LightGlue 特征提取器: {extractor_name}")
        self.extractor_name = extractor_name
        self.extractor = extractors[extractor_name]().eval().to(self.device)
        self.matcher = LightGlue(features=extractor_name.casefold()).eval().to(self.device)

    def _feature_cache_path(self, batch: Batch | None, source_path: str,
                            profile_key: str = "full") -> Path:
        size, mtime_ns = source_fingerprint(Path(source_path))
        key = (
            f"{source_path}|{size}|{mtime_ns}|{profile_key}|{self.extractor_name}|"
            f"{settings.lightglue_max_keypoints}|{settings.lightglue_resize}|"
            f"{settings.lightglue_algorithm_version}"
        )
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:28]
        root = (
            Path(batch.root_path) / ".design_matcher" / str(batch.id)
            if batch is not None
            else Path(settings.lightglue_model_dir) / "pair_test_cache"
        )
        return root / "features" / "lightglue" / f"{digest}.pt"

    def extract(self, source_path: str, *, batch: Batch | None = None,
                profile_key: str = "full") -> dict:
        from lightglue.utils import numpy_image_to_torch

        cache_path = self._feature_cache_path(batch, source_path, profile_key)
        if cache_path.is_file():
            try:
                cached = self.torch.load(cache_path, map_location="cpu", weights_only=True)
                return {key: value.to(self.device) if self.torch.is_tensor(value) else value for key, value in cached.items()}
            except (OSError, RuntimeError, ValueError, KeyError):
                pass
        # cv2.imread cannot reliably open Chinese UNC paths on Windows. Reuse
        # the project's Pillow/PSD-safe loader, then convert to LightGlue's
        # expected RGB tensor without copying the source file locally.
        pil_image = open_visual(Path(source_path)).convert("RGB")
        image = numpy_image_to_torch(np.asarray(pil_image).copy()).to(self.device)
        with self.torch.inference_mode():
            features = self.extractor.extract(image, resize=settings.lightglue_resize)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save({key: value.detach().cpu() if self.torch.is_tensor(value) else value for key, value in features.items()}, cache_path)
        return features

    def match_features(self, features0: dict, features1: dict) -> LocalMatchResult:
        import cv2
        from lightglue.utils import rbd

        with self.torch.inference_mode():
            output = self.matcher({"image0": features0, "image1": features1})
        f0, f1, result = [rbd(value) for value in (features0, features1, output)]
        matches = result["matches"].detach().cpu().numpy()
        raw_count = int(len(matches))
        if raw_count:
            points0 = f0["keypoints"][matches[:, 0]].detach().cpu().numpy().astype(np.float32)
            points1 = f1["keypoints"][matches[:, 1]].detach().cpu().numpy().astype(np.float32)
            scores = result.get("scores")
            mean_confidence = float(scores.mean().item()) if scores is not None and len(scores) else 0.0
        else:
            points0 = points1 = np.empty((0, 2), dtype=np.float32)
            mean_confidence = 0.0
        inlier_mask = np.zeros(raw_count, dtype=bool)
        if raw_count >= 4:
            _, mask = cv2.findHomography(
                points0, points1, cv2.RANSAC, settings.lightglue_ransac_threshold
            )
            if mask is not None:
                inlier_mask = mask.reshape(-1).astype(bool)
        inlier_count = int(inlier_mask.sum())
        inlier_ratio = inlier_count / raw_count if raw_count else 0.0
        coverage = self._coverage(points0[inlier_mask], f0.get("image_size"), points1[inlier_mask], f1.get("image_size"))
        count_score = 1.0 - math.exp(-inlier_count / 40.0)
        weights = (
            settings.local_inlier_ratio_weight,
            settings.local_confidence_weight,
            settings.local_coverage_weight,
            settings.local_count_weight,
        )
        total = sum(weights)
        local_score = (
            inlier_ratio * weights[0]
            + mean_confidence * weights[1]
            + coverage * weights[2]
            + count_score * weights[3]
        ) / max(total, 1e-8)
        return LocalMatchResult(
            raw_match_count=raw_count,
            ransac_inlier_count=inlier_count,
            inlier_ratio=float(np.clip(inlier_ratio, 0, 1)),
            mean_match_confidence=float(np.clip(mean_confidence, 0, 1)),
            coverage_score=float(np.clip(coverage, 0, 1)),
            local_score=float(np.clip(local_score, 0, 1)),
        )

    @staticmethod
    def _one_image_coverage(points: np.ndarray, image_size) -> float:
        if not len(points):
            return 0.0
        if image_size is None:
            width = max(float(points[:, 0].max()), 1.0)
            height = max(float(points[:, 1].max()), 1.0)
        else:
            values = image_size.detach().cpu().numpy().reshape(-1) if hasattr(image_size, "detach") else np.asarray(image_size).reshape(-1)
            width, height = float(values[0]), float(values[1])
        x = np.clip((points[:, 0] / max(width, 1.0) * 3).astype(int), 0, 2)
        y = np.clip((points[:, 1] / max(height, 1.0) * 3).astype(int), 0, 2)
        return len(set(zip(x.tolist(), y.tolist()))) / 9.0

    @classmethod
    def _coverage(cls, points0: np.ndarray, size0, points1: np.ndarray, size1) -> float:
        return math.sqrt(cls._one_image_coverage(points0, size0) * cls._one_image_coverage(points1, size1))


def get_runtime() -> LightGlueRuntime:
    global _runtime
    if _runtime is None:
        with _runtime_lock:
            if _runtime is None:
                _runtime = LightGlueRuntime()
    return _runtime


def pair_test(query_path: str, reference_path: str) -> dict:
    runtime = get_runtime()
    query = runtime.extract(query_path)
    reference = runtime.extract(reference_path)
    return {
        "query": query_path,
        "reference": reference_path,
        "device": str(runtime.device),
        "extractor": runtime.extractor_name,
        **runtime.match_features(query, reference).to_dict(),
    }
