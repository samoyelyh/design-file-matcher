from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.db import Base
from app.models import Batch, MainImage, MatchedFile
from app.services.assignment import (
    Assignment,
    GlobalAssignmentSolver,
    MatchConfidenceEvaluator,
)
from app.services.matcher import match_batch
from app.services.scanner import scan_batch


def solve(matrix, dummy=0.50):
    return GlobalAssignmentSolver(dummy).solve(np.asarray(matrix, dtype=np.float32))


def test_top1_collision_is_resolved_globally():
    matrix = [[.95, .90, .70], [.94, .89, .88], [.93, .87, .86]]
    assignments = solve(matrix)
    real = [item.column for item in assignments if item.column is not None]
    assert len(real) == len(set(real)) == 3


def test_clear_diagonal_is_matched():
    matrix = np.asarray([[.95, .60, .55], [.58, .94, .57], [.59, .56, .93]])
    evaluator = MatchConfidenceEvaluator(.80, .01)
    results = [evaluator.evaluate(matrix, item) for item in solve(matrix)]
    assert [item.status for item in results] == ["MATCHED"] * 3


def test_small_row_margin_is_ambiguous():
    matrix = np.asarray([[.824, .820]])
    result = MatchConfidenceEvaluator(.80, .01).evaluate(
        matrix, Assignment(row=0, column=0)
    )
    assert result.status == "AMBIGUOUS"
    assert round(result.row_margin, 3) == .004


def test_score_below_match_threshold_is_low_confidence():
    matrix = np.asarray([[.65, .40]])
    assignment = solve(matrix)[0]
    result = MatchConfidenceEvaluator(.80, .01).evaluate(matrix, assignment)
    assert result.status == "LOW_CONFIDENCE"


def test_dummy_allows_unmatched():
    matrix = np.asarray([[.42, .40]])
    assignment = solve(matrix)[0]
    assert assignment.column is None
    assert MatchConfidenceEvaluator(.80, .01).evaluate(matrix, assignment).status == "UNMATCHED"


def test_automatic_product_count_is_at_most_one_per_category():
    matrix = np.asarray([[.95, .60], [.94, .59], [.93, .58], [.92, .57]])
    real = [item.column for item in solve(matrix) if item.column is not None]
    assert len(real) == len(set(real))


def test_one_to_one_constraint_is_independent_between_categories():
    white_background = solve([[.95]])
    spectator = solve([[.93]])
    assert white_background[0].column == spectator[0].column == 0


class ColorEmbedder:
    device = "TEST/CPU"

    def encode(self, paths):
        vectors = []
        for path in paths:
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.float32).mean(axis=(0, 1))
            vectors.append(rgb / max(float(np.linalg.norm(rgb)), 1e-12))
        return np.asarray(vectors)


def make_image(path: Path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 24), color).save(path)


def test_manual_confirmation_survives_automatic_rerun(tmp_path):
    root = tmp_path / "design"
    make_image(root / "主图" / "RED.jpg", (255, 0, 0))
    make_image(root / "主图" / "GREEN.jpg", (0, 255, 0))
    make_image(root / "观赛图" / "image.jpg", (255, 0, 0))
    engine = create_engine(f"sqlite:///{tmp_path / 'manual.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        batch = Batch(root_path=str(root))
        db.add(batch)
        db.commit()
        scan_batch(db, batch.id)
        match_batch(db, batch.id, ColorEmbedder())
        item = db.scalar(select(MatchedFile))
        green = db.scalar(select(MainImage).where(MainImage.basename == "GREEN"))
        item.matched_main_id = green.id
        item.matched_main_basename = green.basename
        item.manual_product_id = green.id
        item.status = "MANUAL_CONFIRMED"
        item.review_status = "CORRECTED"
        item.confirmed_at = datetime.now(timezone.utc)
        db.commit()

        match_batch(db, batch.id, ColorEmbedder())
        db.refresh(item)
        assert item.matched_main_id == green.id
        assert item.status == "MANUAL_CONFIRMED"
