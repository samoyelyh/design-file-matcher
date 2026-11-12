from __future__ import annotations

from collections import defaultdict

from app.config import settings
from app.services.dinov3 import DINOV2_MODEL, DINOV3_MODEL, build_dino_product_matrices, model_status, require_model_ready

from .base import EngineContext, EngineOutput, MatchEngine


class DinoEngine(MatchEngine):
    name = "DINO"
    label = "DINOv2 Base ViT-B/14（GPU）"

    def __init__(self, backend: str = DINOV2_MODEL) -> None:
        self.backend = backend if backend in {DINOV2_MODEL, DINOV3_MODEL} else DINOV2_MODEL

    def config_snapshot(self) -> dict:
        status = model_status(self.backend)
        return {
            "engine": self.name,
            "version": settings.dinov2_algorithm_version if self.backend == DINOV2_MODEL else settings.dinov3_algorithm_version,
            "backend": self.backend,
            "model_id": status["model_id"],
            "model_dir": status["model_dir"],
            "min_score": settings.dinov3_min_score,
            "min_margin": settings.dinov3_min_margin,
            "unmatched_score": settings.unmatched_score,
        }

    def status(self) -> dict:
        status = model_status(self.backend)
        return {
            **status,
            "id": self.name,
            "backend": self.backend,
            "label": self.label,
            "description": "视觉特征模型，适合整体视觉语义匹配",
        }

    def build_score_matrices(self, context: EngineContext) -> EngineOutput:
        state = require_model_ready(self.backend)
        context.progress("DINO_LOADING", 0, len(context.mains), 2, f"正在加载 {state['label']} 到显卡", state["device"])
        positions: dict[str, list[int]] = defaultdict(list)
        for index, item in enumerate(context.files):
            positions[item.category].append(index)
        files_by_category = {
            category: [context.files[index] for index in rows]
            for category, rows in positions.items()
        }
        context.progress("DINO_EMBEDDING", 0, len(context.files), 30, "正在用 GPU 计算手动裁剪遮罩后的 DINO 特征", state["device"])
        matrices, scores = build_dino_product_matrices(
            context.db, context.batch, context.mains, files_by_category,
            context.profiles, self.backend,
        )
        context.db.commit()
        snapshot = self.config_snapshot()
        return EngineOutput(
            category_matrices=matrices,
            product_scores=scores,
            algorithm_version=snapshot["version"],
            model_name=snapshot["model_id"],
            min_score=settings.dinov3_min_score,
            min_margin=settings.dinov3_min_margin,
            unmatched_score=settings.unmatched_score,
            device_label=state["device"],
            assignment_input=f"complete_manual_mask_{self.backend.casefold()}_cosine_matrix",
            candidate_input=f"manual_mask_{self.backend.casefold()}",
            feature_version_prefix="manual_masks",
        )
