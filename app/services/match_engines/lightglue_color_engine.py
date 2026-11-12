from __future__ import annotations

from collections import defaultdict
import hashlib
from time import monotonic

import numpy as np

from app.config import BASE_DIR, settings
from app.services.color_features import build_color_matrix
from app.services.cropper import MaskedCropProcessor
from app.services.lightglue_matching import LocalMatchResult, dependency_status, get_runtime
from app.services.pair_cache import PairResultCache
from app.services.masks import MAIN_IMAGE_CATEGORY
from app.services.references import ProductEvidence, ProductScoreMatrix

from .base import EngineContext, EngineOutput, MatchEngine


class LightGlueColorEngine(MatchEngine):
    name = "LIGHTGLUE_COLOR"
    label = "LightGlue + Color（实验）"

    def config_snapshot(self) -> dict:
        return {
            "engine": self.name,
            "version": settings.lightglue_algorithm_version,
            "extractor": settings.lightglue_extractor,
            "max_keypoints": settings.lightglue_max_keypoints,
            "resize": settings.lightglue_resize,
            "prefilter_top_n": settings.lightglue_prefilter_top_n,
            "color_feature_version": settings.color_feature_version,
            "color_weight": settings.lightglue_color_weight,
            "local_weight": settings.lightglue_local_weight,
            "local_weights": {
                "inlier_ratio": settings.local_inlier_ratio_weight,
                "confidence": settings.local_confidence_weight,
                "coverage": settings.local_coverage_weight,
                "count": settings.local_count_weight,
            },
            "ransac_threshold": settings.lightglue_ransac_threshold,
            "min_score": settings.lightglue_min_score,
            "min_margin": settings.lightglue_min_margin,
            "unmatched_score": settings.unmatched_score,
            "model_dir": settings.lightglue_model_dir,
        }

    def status(self) -> dict:
        return {
            **dependency_status(),
            "id": self.name,
            "label": self.label,
            "description": "局部图案与颜色联合匹配，重点区分号码、印花、袖口和配色",
        }

    def build_score_matrices(self, context: EngineContext) -> EngineOutput:
        status = dependency_status()
        if not status.get("selectable"):
            raise ValueError(f"LightGlue + Color 尚未就绪：{status.get('error') or settings.lightglue_model_dir}")
        runtime = get_runtime()
        processor = MaskedCropProcessor()
        main_profile = context.profiles[MAIN_IMAGE_CATEGORY]
        main_crops = {
            main.id: processor.process_main(context.db, context.batch, main, main_profile).crop.asset_path
            for main in context.mains
        }
        context.db.commit()
        positions: dict[str, list[int]] = defaultdict(list)
        for index, item in enumerate(context.files):
            positions[item.category].append(index)
        category_matrices: dict[str, np.ndarray] = {}
        product_scores: dict[str, ProductScoreMatrix] = {}
        main_feature_cache: dict[int, dict] = {}
        total_pairs = sum(len(rows) * min(settings.lightglue_prefilter_top_n, len(context.mains)) for rows in positions.values())
        completed_pairs = 0
        last_progress = monotonic()
        for category, rows in positions.items():
            files = [context.files[index] for index in rows]
            profile = context.profiles[category]
            context.progress(
                "COLOR_PREFILTER", completed_pairs, total_pairs, 18,
                f"正在计算 {category} 的完整 HSV / Lab Color Matrix",
                str(runtime.device),
            )
            color_matrix, _cache = build_color_matrix(
                context.batch, files, context.mains, profile, main_profile
            )
            # Every cell starts with a valid Color-only score. TopN cells are
            # refined below; non-TopN cells remain in the complete matrix.
            matrix = color_matrix * settings.lightglue_color_weight
            details: list[list[dict]] = [[{
                "final_score": float(matrix[row, column]),
                "color_score": float(color_matrix[row, column]),
                "local_score": None,
                "refined": False,
            } for column in range(len(context.mains))] for row in range(len(files))]
            top_n = min(settings.lightglue_prefilter_top_n, len(context.mains))
            cache = PairResultCache(
                BASE_DIR / "data" / "pair_cache" / str(context.batch.id)
                / f"{hashlib.sha256(category.encode('utf-8')).hexdigest()}.json",
                {**self.config_snapshot(), "device": str(runtime.device),
                 "main_profile": [main_profile.id, main_profile.version],
                 "query_profile": [profile.id, profile.version]},
            )
            for row, item in enumerate(files):
                crop = processor.process_file(context.db, context.batch, item, profile).crop
                query_features = None
                selected = np.argsort(-color_matrix[row])[:top_n]
                for column_value in selected:
                    column = int(column_value)
                    main = context.mains[column]
                    cache_key = cache.key(crop.asset_path, main_crops[main.id])
                    cached = cache.get(cache_key)
                    if cached is not None:
                        local = LocalMatchResult(**cached)
                    else:
                        if query_features is None:
                            query_features = runtime.extract(
                                crop.asset_path, batch=context.batch,
                                profile_key=f"mask:{profile.id}:v{profile.version}",
                            )
                        if main.id not in main_feature_cache:
                            main_feature_cache[main.id] = runtime.extract(
                                main_crops[main.id], batch=context.batch,
                                profile_key=f"mask:{main_profile.id}:v{main_profile.version}",
                            )
                        local = runtime.match_features(query_features, main_feature_cache[main.id])
                        cache.rows[cache_key] = local.to_dict()
                    final_score = (
                        float(color_matrix[row, column]) * settings.lightglue_color_weight
                        + local.local_score * settings.lightglue_local_weight
                    )
                    matrix[row, column] = final_score
                    details[row][column] = {
                        "final_score": float(final_score),
                        "color_score": float(color_matrix[row, column]),
                        "local_score": local.local_score,
                        "raw_matches": local.raw_match_count,
                        "ransac_inliers": local.ransac_inlier_count,
                        "inlier_ratio": local.inlier_ratio,
                        "mean_match_confidence": local.mean_match_confidence,
                        "coverage_score": local.coverage_score,
                        "refined": True,
                    }
                    completed_pairs += 1
                    if completed_pairs == total_pairs or monotonic() - last_progress >= 1:
                        context.progress(
                            "LIGHTGLUE_REFINE", completed_pairs, total_pairs,
                            20 + 68 * completed_pairs / max(total_pairs, 1),
                            f"LightGlue 精排 {completed_pairs} / {total_pairs}，本分类复用 {cache.hits} 对",
                            str(runtime.device),
                        )
                        last_progress = monotonic()
                item.match_method = "LIGHTGLUE_COLOR"
                item.phash_best_region_type = None
                item.phash_distance = None
            cache.save()
            evidence = [[ProductEvidence(
                aggregated_score=float(matrix[row, column]),
                best_distance=None,
                second_distance=None,
                third_distance=None,
                support_count=int(details[row][column].get("ransac_inliers", 0)),
                reference_id=None,
                reference_category=None,
                details=details[row][column],
            ) for column in range(len(context.mains))] for row in range(len(files))]
            category_matrices[category] = matrix.astype(np.float32)
            product_scores[category] = ProductScoreMatrix(matrix.astype(np.float32), evidence)
            context.db.commit()
        return EngineOutput(
            category_matrices=category_matrices,
            product_scores=product_scores,
            algorithm_version=settings.lightglue_algorithm_version,
            model_name=f"{settings.lightglue_extractor}+LightGlue+HSV/Lab",
            min_score=settings.lightglue_min_score,
            min_margin=settings.lightglue_min_margin,
            unmatched_score=settings.unmatched_score,
            device_label=str(runtime.device),
            assignment_input="complete_color_base_lightglue_refined_product_score_matrix",
            candidate_input="color_all_products_lightglue_color_topn",
            feature_version_prefix="manual_masks_hsv_lab_aliked_lightglue",
            artifacts=[],
        )
