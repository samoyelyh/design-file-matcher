"""LightGlue + Color auxiliary scoring for the full-Qwen matching pipeline.

Qwen still compares every query with every main image. Color selects only which
pairs receive the more expensive LightGlue refinement, following the existing
LIGHTGLUE_COLOR engine; it never filters a Qwen comparison or candidate row.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

import numpy as np

from app.config import BASE_DIR, settings
from app.models import Batch, MainImage, MatchedFile
from app.services.color_features import (
    ColorFeature,
    _feature_from_pixels,
    _masked_pixels,
    color_score,
    extract_color_feature,
)
from app.services.cropper import MaskedCropProcessor
from app.services.lightglue_matching import LocalMatchResult, dependency_status, get_runtime
from app.services.masks import MAIN_IMAGE_CATEGORY, active_manual_profile, profile_storage_valid
from app.services.media import open_visual
from app.services.pair_cache import PairResultCache


FUSION_VERSION = "qwen_lightglue_color_fusion_v1"


def qwen_support_score(match: bool, confidence: float) -> float:
    """Convert the model's self-reported verdict confidence into match support."""
    value = float(np.clip(confidence, 0.0, 1.0))
    return value if match else 1.0 - value


def fused_score(
    qwen_score: float,
    lightglue_color_score: float | None,
    *,
    qwen_weight: float = 0.70,
    lightglue_color_weight: float = 0.30,
) -> float:
    """Combine available normalized scores; a missing auxiliary score is omitted."""
    q_weight = max(0.0, float(qwen_weight))
    visual_weight = max(0.0, float(lightglue_color_weight)) if lightglue_color_score is not None else 0.0
    total = q_weight + visual_weight
    if total <= 0:
        return float(np.clip(qwen_score, 0.0, 1.0))
    numerator = q_weight * float(np.clip(qwen_score, 0.0, 1.0))
    if visual_weight:
        numerator += visual_weight * float(np.clip(lightglue_color_score, 0.0, 1.0))
    return float(np.clip(numerator / total, 0.0, 1.0))


def lightglue_color_score(color_value: float | None, local_score: float | None) -> float | None:
    """Follow the existing LightGlueColorEngine score-matrix contract."""
    if color_value is None:
        return None
    value = float(color_value) * settings.lightglue_color_weight
    if local_score is not None:
        value += float(local_score) * settings.lightglue_local_weight
    return float(np.clip(value, 0.0, 1.0))


def _full_image_color(path: str) -> ColorFeature:
    image = open_visual(Path(path)).convert("RGB")
    pixels = np.asarray(image, dtype=np.uint8).reshape(-1, 3)
    hsv, lab_hist, lab_mean = _feature_from_pixels(pixels)
    return ColorFeature(hsv, lab_hist, lab_mean, "", False)


def _valid_profile(db, profile) -> bool:
    return bool(
        profile
        and Path(profile.mask_path).is_file()
        and profile_storage_valid(db, profile)
    )


class QwenFusionScorer:
    """Score a query against the category's complete main-image set."""

    def __init__(self, db, batch: Batch, category: str, mains: list[MainImage], *,
                 qwen_weight: float | None = None,
                 lightglue_color_weight: float | None = None) -> None:
        self.db = db
        self.batch = batch
        self.category = category
        self.mains = mains
        self.qwen_weight = settings.qwen_fusion_qwen_weight if qwen_weight is None else qwen_weight
        self.lightglue_color_weight = (
            settings.qwen_fusion_lightglue_color_weight
            if lightglue_color_weight is None else lightglue_color_weight
        )
        self.color_weight = settings.lightglue_color_weight
        self.local_weight = settings.lightglue_local_weight
        self.top_n = min(max(0, settings.lightglue_prefilter_top_n), len(mains))
        self.main_names = {main.id: main.basename for main in mains}
        self.query_profile = active_manual_profile(db, batch.id, category)
        self.main_profile = active_manual_profile(db, batch.id, MAIN_IMAGE_CATEGORY)
        self.use_roi = _valid_profile(db, self.query_profile) and _valid_profile(db, self.main_profile)
        self.roi_error: str | None = None
        self.processor = MaskedCropProcessor()
        self.main_local_paths = {main.id: main.file_path for main in mains}

        if self.use_roi:
            try:
                self.main_local_paths = {
                    main.id: self.processor.process_main(db, batch, main, self.main_profile).crop.asset_path
                    for main in mains
                }
                db.commit()
            except Exception as exc:
                self.use_roi = False
                self.roi_error = str(exc)
                self.main_local_paths = {main.id: main.file_path for main in mains}

        self.main_color_features: list[ColorFeature | None] = []
        self.color_mode = "mask_roi" if self.use_roi else "full_image_fallback"
        try:
            if self.use_roi:
                self.main_color_features = [
                    extract_color_feature(batch, main.file_path, self.main_profile,
                                          kind="main", object_id=main.id)
                    for main in mains
                ]
            else:
                self.main_color_features = [_full_image_color(main.file_path) for main in mains]
        except Exception as exc:
            # A stale or invalid mask must not prevent Qwen matching. Switch the
            # entire category to consistent full-image Color features.
            self.use_roi = False
            self.color_mode = "full_image_fallback"
            self.roi_error = self.roi_error or str(exc)
            self.main_local_paths = {main.id: main.file_path for main in mains}
            try:
                self.main_color_features = [_full_image_color(main.file_path) for main in mains]
            except Exception as color_exc:
                self.main_color_features = [None] * len(mains)
                self.roi_error = f"{self.roi_error or 'ROI Color失败'}；全图Color失败：{color_exc}"

        self.lg_status = dependency_status()
        self.lightglue_available = bool(self.lg_status.get("selectable"))
        self.lightglue_error = self.lg_status.get("error") if not self.lightglue_available else None
        self._runtime = None
        self._runtime_error: str | None = None
        category_key = hashlib.sha256(category.encode("utf-8")).hexdigest()[:20]
        self.local_cache = PairResultCache(
            BASE_DIR / "data" / "pair_cache" / "qwen_fusion" / str(batch.id) / f"{category_key}.json",
            {
                "version": FUSION_VERSION,
                "color_feature_version": settings.color_feature_version,
                "extractor": settings.lightglue_extractor,
                "max_keypoints": settings.lightglue_max_keypoints,
                "resize": settings.lightglue_resize,
                "color_weight": self.color_weight,
                "local_weight": self.local_weight,
                "roi": self.use_roi,
                "query_profile": [self.query_profile.id, self.query_profile.version] if self.use_roi else None,
                "main_profile": [self.main_profile.id, self.main_profile.version] if self.use_roi else None,
            },
        )

    @property
    def planned_refinements(self) -> int:
        return self.top_n if self.lightglue_available else 0

    def snapshot(self) -> dict:
        return {
            "version": FUSION_VERSION,
            "weights": {
                "qwen": float(self.qwen_weight),
                "lightglue_color": float(self.lightglue_color_weight),
                "color_within_lightglue_color": float(self.color_weight),
                "local_within_lightglue_color": float(self.local_weight),
            },
            "color_top_n": self.top_n if self.lightglue_available else 0,
            "roi_mode": "manual_mask_roi" if self.use_roi else "full_image_fallback",
            "roi_error": self.roi_error,
            "lightglue_available": self.lightglue_available,
            "lightglue_error": self.lightglue_error or self._runtime_error,
            "device": self.lg_status.get("device", "未安装"),
            "score_note": "融合分仅用于排序和Top2候选选择，不是概率；Qwen confidence仍为模型自报值。",
        }

    def _runtime_instance(self):
        if self._runtime is None and self._runtime_error is None:
            try:
                self._runtime = get_runtime()
            except Exception as exc:
                self._runtime_error = str(exc)
        return self._runtime

    def _query_features(self, item: MatchedFile, *, use_roi: bool):
        if use_roi:
            crop = self.processor.process_file(self.db, self.batch, item, self.query_profile).crop.asset_path
            feature = extract_color_feature(self.batch, item.current_path, self.query_profile,
                                            kind="file", object_id=item.id)
            return crop, feature
        return item.current_path, _full_image_color(item.current_path)

    def _full_main_colors(self) -> list[ColorFeature | None]:
        if self.color_mode == "full_image_fallback" and all(value is not None for value in self.main_color_features):
            return self.main_color_features
        try:
            return [_full_image_color(main.file_path) for main in self.mains]
        except Exception:
            return [None] * len(self.mains)

    def score_query(
        self,
        item: MatchedFile,
        pair_rows: list[dict],
        on_candidate_done: Callable[[int, str], None] | None = None,
    ) -> tuple[list[dict], dict]:
        rows = [dict(row) for row in pair_rows]
        use_roi = self.use_roi
        try:
            query_local_path, query_color = self._query_features(item, use_roi=use_roi)
            self.db.commit()
        except Exception as exc:
            use_roi = False
            self.roi_error = self.roi_error or f"Query ROI失败：{exc}"
            query_local_path = item.current_path
            try:
                query_color = _full_image_color(item.current_path)
            except Exception as color_exc:
                query_color = None
                self.roi_error = f"{self.roi_error}；Query Color失败：{color_exc}"

        colors: list[float | None] = []
        color_features = self.main_color_features if use_roi == self.use_roi else self._full_main_colors()
        for index in range(len(self.mains)):
            try:
                ref = color_features[index] if index < len(color_features) else None
                colors.append(color_score(query_color, ref) if query_color is not None and ref is not None else None)
            except Exception:
                colors.append(None)

        valid_indices = [index for index, row in enumerate(rows) if not row.get("error")]
        color_ranked = sorted(
            valid_indices,
            key=lambda index: (colors[index] is None, -(colors[index] or 0.0), index),
        )
        selected = set(color_ranked[:self.top_n]) if self.lightglue_available else set()
        if not selected and self.lightglue_available and valid_indices:
            selected = set(sorted(
                valid_indices,
                key=lambda index: (-qwen_support_score(
                    bool(rows[index].get("match")), float(rows[index].get("confidence", 0.0))
                ), index),
            )[:self.top_n])

        runtime = self._runtime_instance() if selected else None
        selected_rows: dict[int, dict] = {}
        for index, row in enumerate(rows):
            local_details = None
            if index in selected:
                if runtime is not None and not self._runtime_error:
                    try:
                        main = self.mains[index]
                        local_query_path = query_local_path if use_roi else item.current_path
                        local_main_path = self.main_local_paths[main.id] if use_roi else main.file_path
                        cache_key = self.local_cache.key(local_query_path, local_main_path)
                        cached = self.local_cache.get(cache_key)
                        if cached is not None:
                            local = LocalMatchResult(**cached)
                        else:
                            query_features = runtime.extract(
                                local_query_path, batch=self.batch,
                                profile_key=(f"mask:{self.query_profile.id}:v{self.query_profile.version}" if use_roi else "full"),
                            )
                            main_features = runtime.extract(
                                local_main_path, batch=self.batch,
                                profile_key=(f"mask:{self.main_profile.id}:v{self.main_profile.version}" if use_roi else "full"),
                            )
                            local = runtime.match_features(query_features, main_features)
                            self.local_cache.rows[cache_key] = local.to_dict()
                        local_details = local.to_dict()
                    except Exception as exc:
                        self._runtime_error = str(exc)
                selected_rows[index] = local_details or {}

            color_value = colors[index]
            local_score = (local_details or {}).get("local_score")
            if local_score is not None and color_value is not None:
                visual_score = lightglue_color_score(color_value, local_score)
                visual_mode = "lightglue_color_refined"
            elif color_value is not None:
                # Match the existing engine contract exactly: unrefined pairs
                # keep Color * color_weight in the full matrix.
                visual_score = lightglue_color_score(color_value, None)
                visual_mode = "color_only_proxy"
            else:
                visual_score = None
                visual_mode = "qwen_only"

            if row.get("error"):
                support = None
                combined = None
            else:
                support = qwen_support_score(bool(row.get("match")), float(row.get("confidence", 0.0)))
                combined = fused_score(
                    support,
                    visual_score,
                    qwen_weight=self.qwen_weight,
                    lightglue_color_weight=self.lightglue_color_weight,
                )
            row.update({
                "main_basename": self.main_names.get(row.get("main_id"), ""),
                "qwen_support_score": support,
                "color_score": color_value,
                "local_score": local_score,
                "lightglue_color_score": visual_score,
                "lightglue_refined": bool(local_score is not None),
                "fusion_score": combined,
                "fusion_mode": visual_mode,
                "fusion_weights": {
                    "qwen": float(self.qwen_weight),
                    "lightglue_color": float(self.lightglue_color_weight),
                },
                "lightglue_diagnostics": local_details or None,
                "lightglue_error": (
                    self._runtime_error or self.lightglue_error
                    if index in selected and local_score is None else None
                ),
            })
            if on_candidate_done:
                on_candidate_done(index + 1, f"{item.original_filename} 融合评分 {index + 1}/{len(rows)}")

        self.local_cache.save()
        # Keep stable source order in the evidence array; rank_pairwise handles ordering.
        metadata = self.snapshot()
        metadata["query_roi_mode"] = "manual_mask_roi" if use_roi else "full_image_fallback"
        if not use_roi and self.use_roi:
            metadata["query_roi_error"] = self.roi_error
        return rows, metadata
