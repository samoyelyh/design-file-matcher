from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import settings
from app.models import Batch, MainImage, MaskProfile, MatchedFile

from .media import open_visual
from .phash import source_fingerprint


@dataclass(frozen=True)
class ColorFeature:
    hsv: np.ndarray
    lab_hist: np.ndarray
    lab_mean: np.ndarray
    cache_path: str
    cached: bool


def _normalized_hist(values: np.ndarray, bins: int, value_range: tuple[float, float],
                     weights: np.ndarray | None = None) -> np.ndarray:
    hist, _ = np.histogram(values, bins=bins, range=value_range, weights=weights)
    hist = hist.astype(np.float32)
    total = float(hist.sum())
    return hist / total if total > 0 else hist


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    values = rgb.astype(np.float32) / 255.0
    linear = np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)
    xyz = linear @ np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32).T
    xyz /= np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    delta = 6 / 29
    f = np.where(xyz > delta ** 3, np.cbrt(xyz), xyz / (3 * delta ** 2) + 4 / 29)
    return np.stack([
        116 * f[:, 1] - 16,
        500 * (f[:, 0] - f[:, 1]),
        200 * (f[:, 1] - f[:, 2]),
    ], axis=1).astype(np.float32)


def _feature_from_pixels(pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if pixels.size == 0:
        raise ValueError("手动遮罩内没有有效颜色像素")
    sample_step = max(1, len(pixels) // 250_000)
    pixels = pixels[::sample_step]
    hsv = np.asarray(Image.fromarray(pixels.reshape(-1, 1, 3), "RGB").convert("HSV"), dtype=np.float32).reshape(-1, 3)
    hue = _normalized_hist(hsv[:, 0], 24, (0, 256), weights=0.25 + 0.75 * hsv[:, 1] / 255.0)
    saturation = _normalized_hist(hsv[:, 1], 8, (0, 256))
    hsv_feature = np.concatenate([hue, saturation]).astype(np.float32)
    hsv_feature /= max(float(np.linalg.norm(hsv_feature)), 1e-8)

    lab = _rgb_to_lab(pixels)
    lab_hist, _, _ = np.histogram2d(lab[:, 1], lab[:, 2], bins=(12, 12), range=((-128, 128), (-128, 128)))
    lab_hist = lab_hist.astype(np.float32).reshape(-1)
    lab_hist /= max(float(np.linalg.norm(lab_hist)), 1e-8)
    lab_mean = np.mean(lab, axis=0).astype(np.float32)
    return hsv_feature, lab_hist, lab_mean


def _masked_pixels(source_path: str, profile: MaskProfile) -> np.ndarray:
    image = open_visual(Path(source_path)).convert("RGB")
    with Image.open(profile.mask_path) as stored:
        mask = stored.convert("L").resize(image.size, Image.Resampling.NEAREST)
    mask_values = np.asarray(mask) >= 128
    crop = (profile.crop_left, profile.crop_top, profile.crop_right, profile.crop_bottom)
    if all(value is not None for value in crop):
        left, top, right, bottom = (float(value) for value in crop)
        crop_mask = np.zeros_like(mask_values)
        crop_mask[
            max(0, round(top * image.height)):min(image.height, round(bottom * image.height)),
            max(0, round(left * image.width)):min(image.width, round(right * image.width)),
        ] = True
        mask_values &= crop_mask
    return np.asarray(image, dtype=np.uint8)[mask_values]


def _cache_path(batch: Batch, kind: str, object_id: int, source_path: str,
                profile: MaskProfile) -> Path:
    size, mtime_ns = source_fingerprint(Path(source_path))
    key = f"{source_path}|{size}|{mtime_ns}|{profile.id}|{profile.version}|{settings.color_feature_version}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:24]
    return Path(batch.root_path) / ".design_matcher" / str(batch.id) / "features" / "color" / settings.color_feature_version / f"{kind}_{object_id}_{digest}.npz"


def extract_color_feature(batch: Batch, source_path: str, profile: MaskProfile,
                          *, kind: str, object_id: int) -> ColorFeature:
    path = _cache_path(batch, kind, object_id, source_path, profile)
    if path.is_file():
        try:
            with np.load(path, allow_pickle=False) as data:
                return ColorFeature(data["hsv"], data["lab_hist"], data["lab_mean"], str(path), True)
        except (OSError, ValueError, KeyError):
            pass
    hsv, lab_hist, lab_mean = _feature_from_pixels(_masked_pixels(source_path, profile))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, hsv=hsv, lab_hist=lab_hist, lab_mean=lab_mean)
    return ColorFeature(hsv, lab_hist, lab_mean, str(path), False)


def main_color_features(batch: Batch, mains: list[MainImage], profile: MaskProfile) -> list[ColorFeature]:
    return [extract_color_feature(batch, main.file_path, profile, kind="main", object_id=main.id) for main in mains]


def query_color_features(batch: Batch, files: list[MatchedFile], profile: MaskProfile) -> list[ColorFeature]:
    return [extract_color_feature(batch, item.current_path, profile, kind="file", object_id=item.id) for item in files]


def color_score(query: ColorFeature, reference: ColorFeature) -> float:
    hsv_score = float(np.clip(np.dot(query.hsv, reference.hsv), 0.0, 1.0))
    lab_hist_score = float(np.clip(np.dot(query.lab_hist, reference.lab_hist), 0.0, 1.0))
    delta_e = float(np.linalg.norm(query.lab_mean - reference.lab_mean))
    lab_mean_score = float(np.exp(-delta_e / 28.0))
    total_weight = settings.color_hsv_weight + settings.color_lab_hist_weight + settings.color_lab_mean_weight
    return float(np.clip((
        hsv_score * settings.color_hsv_weight
        + lab_hist_score * settings.color_lab_hist_weight
        + lab_mean_score * settings.color_lab_mean_weight
    ) / max(total_weight, 1e-8), 0.0, 1.0))


def build_color_matrix(batch: Batch, files: list[MatchedFile], mains: list[MainImage],
                       query_profile: MaskProfile, main_profile: MaskProfile) -> tuple[np.ndarray, dict]:
    queries = query_color_features(batch, files, query_profile)
    references = main_color_features(batch, mains, main_profile)
    matrix = np.asarray([
        [color_score(query, reference) for reference in references]
        for query in queries
    ], dtype=np.float32)
    return matrix, {
        "query_features": len(queries),
        "main_features": len(references),
        "cache_hits": sum(item.cached for item in [*queries, *references]),
        "cache_total": len(queries) + len(references),
    }


def color_top_n_recall(matrix: np.ndarray, expected_columns: list[int], top_n: int = 50) -> dict:
    if matrix.shape[0] != len(expected_columns):
        raise ValueError("颜色矩阵与 Ground Truth 数量不一致")
    keep = min(max(1, int(top_n)), matrix.shape[1]) if matrix.shape[1] else 0
    hits = 0
    ranks: list[int] = []
    for row, expected in enumerate(expected_columns):
        order = np.argsort(-matrix[row])
        rank_positions = np.flatnonzero(order == expected)
        if not len(rank_positions):
            continue
        rank = int(rank_positions[0]) + 1
        ranks.append(rank)
        hits += int(rank <= keep)
    return {
        "top_n": keep,
        "eligible": len(expected_columns),
        "hits": hits,
        "recall": hits / len(expected_columns) if expected_columns else None,
        "mean_rank": float(np.mean(ranks)) if ranks else None,
        "max_rank": max(ranks) if ranks else None,
    }
