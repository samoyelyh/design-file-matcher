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


def fill_unmatched_products(
    similarities: np.ndarray,
    assignments: list[Assignment],
) -> list[Assignment]:
    """Globally fill dummy rows into still-unused product columns.

    This second pass never uses Top5 and never assigns in scan order. It solves
    one reduced Hungarian problem over every unmatched row and every remaining
    product column. Rows beyond the available product count stay unmatched.
    """
    matrix = np.asarray(similarities, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("similarity matrix must be two-dimensional")
    image_count, product_count = matrix.shape
    if len(assignments) != image_count:
        raise ValueError("assignment count must equal image count")
    used_columns = {
        assignment.column
        for assignment in assignments
        if assignment.column is not None
    }
    unmatched_rows = [
        assignment.row for assignment in assignments
        if assignment.column is None
    ]
    available_columns = [
        column for column in range(product_count)
        if column not in used_columns
    ]
    if not unmatched_rows or not available_columns:
        return assignments

    reduced = matrix[np.ix_(unmatched_rows, available_columns)]
    reduced_rows, reduced_columns = linear_sum_assignment(-reduced)
    filled = {assignment.row: assignment.column for assignment in assignments}
    for row_index, column_index in zip(reduced_rows, reduced_columns):
        filled[unmatched_rows[int(row_index)]] = available_columns[int(column_index)]
    return [
        Assignment(row=row, column=filled[row])
        for row in sorted(filled)
    ]


def solve_complete_matrix(
    similarities: np.ndarray,
    unmatched_score: float,
    locked_columns_by_row: dict[int, int] | None = None,
    force_assign: bool = False,
) -> list[Assignment]:
    """Solve all pending rows against all products not fixed by manual review.

    ``force_assign`` removes the reusable dummy option. It is used by the
    production naming flow so every available product slot is filled; weak
    scores remain LOW_CONFIDENCE/AMBIGUOUS for review instead of becoming
    UNMATCHED.
    """
    matrix = np.asarray(similarities, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("similarity matrix must be two-dimensional")
    image_count, product_count = matrix.shape
    locked = dict(locked_columns_by_row or {})
    if any(row < 0 or row >= image_count for row in locked):
        raise ValueError("locked row is outside similarity matrix")
    if any(column < 0 or column >= product_count for column in locked.values()):
        raise ValueError("locked product is outside similarity matrix")
    if len(set(locked.values())) != len(locked):
        raise ValueError("manual confirmations contain duplicate products in one category")

    pending_rows = [row for row in range(image_count) if row not in locked]
    locked_columns = set(locked.values())
    available_columns = [
        column for column in range(product_count)
        if column not in locked_columns
    ]
    resolved: dict[int, int | None] = dict(locked)
    if pending_rows:
        reduced = matrix[np.ix_(pending_rows, available_columns)]
        if force_assign:
            rows, columns = linear_sum_assignment(-reduced)
            reduced_assignments = [
                Assignment(row=int(row), column=int(column))
                for row, column in zip(rows, columns)
            ]
        else:
            reduced_assignments = fill_unmatched_products(
                reduced,
                GlobalAssignmentSolver(unmatched_score).solve(reduced),
            )
        for assignment in reduced_assignments:
            resolved[pending_rows[assignment.row]] = (
                available_columns[assignment.column]
                if assignment.column is not None
                else None
            )
    return [Assignment(row=row, column=resolved.get(row)) for row in range(image_count)]


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

    def evaluate_independent(
        self,
        similarities: np.ndarray,
        assignment: Assignment,
    ) -> ConfidenceResult:
        """Evaluate a row against all products without column competition.

        Folder-sharded matching intentionally lets multiple images choose the
        same main image.  Confidence therefore uses only the query row's
        ranking and margin; the one-to-one column margin is not applicable.
        """
        matrix = np.asarray(similarities, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("similarity matrix must be two-dimensional")
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
        row_margin = assigned_score - row_competitor if row_competitor is not None else None
        if assigned_score < self.min_score:
            status = "LOW_CONFIDENCE"
        elif assigned_rank > 1 or (row_margin is not None and row_margin < self.min_margin):
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
            column_competitor_score=None,
            column_margin=None,
        )
