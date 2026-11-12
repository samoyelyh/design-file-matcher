from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sqlalchemy.orm import Session

from app.models import Batch, MainImage, MaskProfile, MatchedFile
from app.services.references import ProductScoreMatrix


ProgressCallback = Callable[[str, int, int, float, str, str | None], None]


@dataclass
class EngineContext:
    db: Session
    batch: Batch
    mains: list[MainImage]
    files: list[MatchedFile]
    profiles: dict[str, MaskProfile]
    progress: ProgressCallback


@dataclass
class EngineOutput:
    category_matrices: dict[str, np.ndarray]
    product_scores: dict[str, ProductScoreMatrix]
    algorithm_version: str
    model_name: str
    min_score: float
    min_margin: float
    unmatched_score: float
    device_label: str
    assignment_input: str
    candidate_input: str
    feature_version_prefix: str
    artifacts: list[object] = field(default_factory=list)


class MatchEngine(ABC):
    name: str
    label: str

    @abstractmethod
    def build_score_matrices(self, context: EngineContext) -> EngineOutput:
        """Return complete score matrices; larger values must always be better."""

    @abstractmethod
    def config_snapshot(self) -> dict:
        """Return the immutable configuration recorded on each batch/run."""

    def status(self) -> dict:
        return {
            "id": self.name,
            "label": self.label,
            "installed": True,
            "selectable": True,
        }
