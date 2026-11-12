from __future__ import annotations

from app.models import Batch

from .base import MatchEngine
from .dino_engine import DinoEngine
from .phash_engine import PHashEngine
from .lightglue_color_engine import LightGlueColorEngine
from .visionmatch_engine import VisionMatchEngine
from app.services.qwen_lab import status as qwen_status


ALIASES = {
    "PHASH_REFERENCE": "PHASH",
    "DINOV2": "DINO",
    "DINOV3": "DINO",
}


def normalize_engine_name(value: str | None) -> str:
    name = (value or "PHASH").strip().upper()
    return ALIASES.get(name, name)


def get_match_engine(batch: Batch) -> MatchEngine:
    stored_engine = getattr(batch, "match_engine", None)
    # Compatibility for batches/tests created before match_engine existed.
    if batch.matching_model in {"DINOV2", "DINOV3"} and stored_engine in {None, "", "PHASH"}:
        stored_engine = "DINO"
    name = normalize_engine_name(stored_engine or batch.matching_model)
    if name == "QWEN":
        raise ValueError("QWEN 使用专用视觉队列，不经过传统匹配引擎")
    if name == "PHASH":
        return PHashEngine()
    if name == "DINO":
        backend = batch.matching_model if batch.matching_model in {"DINOV2", "DINOV3"} else "DINOV2"
        return DinoEngine(backend)
    if name == "LIGHTGLUE_COLOR":
        return LightGlueColorEngine()
    if name == "VISIONMATCH":
        return VisionMatchEngine()
    raise ValueError(f"不支持的匹配引擎: {name}")


def engine_catalog() -> list[dict]:
    state = qwen_status()
    qwen_models = state.get("vision_models", [])
    return [PHashEngine().status(), DinoEngine().status(), LightGlueColorEngine().status(), VisionMatchEngine().status(), {
        "id": "QWEN", "label": "AI视觉匹配（Qwen）", "selectable": bool(state.get("reachable") and qwen_models),
        "installed": bool(qwen_models), "device": "Ollama · GPU concurrency=1", "model": state.get("selected_model"), "models": qwen_models, "error": state.get("error")
    }]
