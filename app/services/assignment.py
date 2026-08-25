from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Assignment:
    row: int
    column: int | None


@dataclass(frozen=True)
class ConfidenceResult:
    status: str
    assigned_score: float | None
    image_best_score: float
    image_second_best_score: float | None
    image_best_column: int
    assigned_rank: int | None
    row_competitor_score: float | None
    row_margin: float | None
    column_competitor_score: float | None
    column_margin: float | None


class GlobalAssignmentSolver:
    """Maximum-weight one-to-one assignment with one dummy slot per image."""

    def __init__(self, unmatched_score: float) -> None:
        self.unmatched_score = unmatched_score

    def solve(self, similarities: np.ndarray) -> list[Assignment]:
        matrix = np.asarray(similarities, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("similarity matrix must be two-dimensional")
        image_count, product_count = matrix.shape
        if image_count == 0:
            return []
        if product_count == 0:
            return [Assignment(row=index, column=None) for index in range(image_count)]

        # Dummy columns are deliberately reusable only through distinct slots.
        # Their identical score means a real product is selected only when the
        # global objective values it above the configured unmatched alternative.
        dummy = np.full((image_count, image_count), self.unmatched_score)
        objective = np.concatenate((matrix, dummy), axis=1)
        rows, columns = linear_sum_assignment(-objective)
        assigned = {int(row): int(column) for row, column in zip(rows, columns)}
        return [
            Assignment(row=row, column=column if column < product_count else None)
            for row, column in sorted(assigned.items())
        ]


class MatchConfidenceEvaluator:
    def __init__(self, min_score: float, min_margin: float) -> None:
        self.min_score = min_score
        self.min_margin = min_margin

    def evaluate(
        self,
        similarities: np.ndarray,
        assignment: Assignment,
    ) -> ConfidenceResult:
        matrix = np.asarray(similarities, dtype=np.float64)
        row_scores = matrix[assignment.row]
        order = np.argsort(-row_scores, kind="stable")
        best_column = int(order[0])
        best_score = float(row_scores[best_column])
        second_score = float(row_scores[int(order[1])]) if len(order) > 1 else None

        if assignment.column is None:
            return ConfidenceResult(
                status="UNMATCHED",
                assigned_score=None,
                image_best_score=best_score,
                image_second_best_score=second_score,
                image_best_column=best_column,
                assigned_rank=None,
                row_competitor_score=None,
                row_margin=None,
                column_competitor_score=None,
                column_margin=None,
            )

        column = assignment.column
        assigned_score = float(row_scores[column])
        assigned_rank = int(np.flatnonzero(order == column)[0]) + 1
        other_columns = np.delete(row_scores, column)
        row_competitor = float(other_columns.max()) if len(other_columns) else None
        row_margin = (
            assigned_score - row_competitor if row_competitor is not None else None
        )
        column_scores = matrix[:, column]
        other_rows = np.delete(column_scores, assignment.row)
        column_competitor = float(other_rows.max()) if len(other_rows) else None
        column_margin = (
            assigned_score - column_competitor
            if column_competitor is not None
            else None
        )

        if assigned_score < self.min_score:
            status = "LOW_CONFIDENCE"
        elif (
            assigned_rank > 1
            or (row_margin is not None and row_margin < self.min_margin)
            or (column_margin is not None and column_margin < self.min_margin)
        ):
            status = "AMBIGUOUS"
        else:
            status = "MATCHED"
        return ConfidenceResult(
            status=status,
            assigned_score=assigned_score,
            image_best_score=best_score,
            image_second_best_score=second_score,
            image_best_column=best_column,
            assigned_rank=assigned_rank,
            row_competitor_score=row_competitor,
            row_margin=row_margin,
            column_competitor_score=column_competitor,
            column_margin=column_margin,
        )
