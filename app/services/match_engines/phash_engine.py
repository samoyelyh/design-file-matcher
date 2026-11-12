from __future__ import annotations

from collections import defaultdict

import numpy as np
from PIL import Image

from app.config import settings
from app.services.main_body import ensure_batch_main_bodies
from app.services.masks import MAIN_IMAGE_CATEGORY
from app.services.phash_matching import build_masked_phash_matrix, masked_query_hashes
from app.services.references import ProductScoreMatrix, build_product_score_matrix

from .base import EngineContext, EngineOutput, MatchEngine


class PHashEngine(MatchEngine):
    name = "PHASH"
    label = "pHash（生产默认）"

    def config_snapshot(self) -> dict:
        return {
            "engine": self.name,
            "version": (
                settings.product_reference_algorithm_version
                if settings.use_product_reference_scoring
                else settings.masked_algorithm_version
            ),
            "hash_bits": 64,
            "reference_scoring": settings.use_product_reference_scoring,
            "max_references_per_category": settings.max_references_per_category,
            "reference_weights": [
                settings.reference_weight_1,
                settings.reference_weight_2,
                settings.reference_weight_3,
            ],
            "min_score": settings.match_min_score,
            "min_margin": settings.match_min_margin,
            "unmatched_score": settings.unmatched_score,
        }

    def status(self) -> dict:
        return {**super().status(), "device": "CPU / pHash", "description": "速度最快，适合整体结构接近的图片"}

    def build_score_matrices(self, context: EngineContext) -> EngineOutput:
        db, batch, mains, files, profiles = (
            context.db, context.batch, context.mains, context.files, context.profiles
        )
        context.progress("MAIN_BODY_PHASH", 0, len(mains), 2, "正在读取主图手动遮罩并生成 pHash", None)
        main_profile = profiles[MAIN_IMAGE_CATEGORY]
        crop_box = (
            float(main_profile.crop_left), float(main_profile.crop_top),
            float(main_profile.crop_right), float(main_profile.crop_bottom),
        )
        with Image.open(main_profile.mask_path) as stored_mask:
            main_mask = stored_mask.convert("L")
            main_mask.load()
        artifacts = ensure_batch_main_bodies(
            db, batch, mains, manual_box=crop_box, manual_mask=main_mask,
            manual_mask_path=main_profile.mask_path,
        )
        db.commit()
        positions: dict[str, list[int]] = defaultdict(list)
        for index, item in enumerate(files):
            positions[item.category].append(index)
        category_matrices: dict[str, np.ndarray] = {}
        product_scores: dict[str, ProductScoreMatrix] = {}
        categories = list(positions)
        for category_index, category in enumerate(categories, 1):
            profile = profiles[category]
            category_files = [files[index] for index in positions[category]]
            context.progress(
                "MASKED_CROP", category_index - 1, len(categories),
                25 + 60 * (category_index - 1) / max(len(categories), 1),
                f"正在生成 {category} Masked Crop 并计算 pHash（Mask v{profile.version}）",
                "CPU / pHash",
            )
            query_hashes = masked_query_hashes(db, batch, category_files, profile)
            category_matrices[category] = build_masked_phash_matrix(
                db, batch, category_files, mains, profile, query_hashes=query_hashes
            )
            product_scores[category] = build_product_score_matrix(
                db, query_hashes, category, mains,
                max_manual_references=(
                    settings.max_references_per_category
                    if settings.use_product_reference_scoring else 0
                ),
                query_file_ids=[item.id for item in category_files],
            )
            db.commit()
        return EngineOutput(
            category_matrices=category_matrices,
            product_scores=product_scores,
            algorithm_version=self.config_snapshot()["version"],
            model_name="pHash64",
            min_score=settings.match_min_score,
            min_margin=settings.match_min_margin,
            unmatched_score=settings.unmatched_score,
            device_label="CPU / pHash",
            assignment_input="complete_manual_mask_product_score_matrix" if settings.use_product_reference_scoring else "complete_manual_mask_phash_matrix",
            candidate_input="manual_mask_product_references_v1" if settings.use_product_reference_scoring else "manual_mask_main_reference_only",
            feature_version_prefix="manual_masks",
            artifacts=artifacts,
        )
