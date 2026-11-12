"""Batch-local review exclusions; never discard matching results or source files."""
from __future__ import annotations

import json

from app.models import Batch, MatchedFile


def ignored_review_categories(batch: Batch) -> set[str]:
    return set(json.loads(batch.match_config_snapshot or "{}").get("review_ignored_categories", []))


def conflict_name_marks(batch: Batch) -> dict[int, int]:
    raw = json.loads(batch.match_config_snapshot or "{}").get("conflict_name_marks", {})
    return {int(key): int(value) for key, value in raw.items()}


def is_review_confirmed(item: MatchedFile) -> bool:
    return bool(
        item.review_status in {"CONFIRMED", "CORRECTED"}
        and item.matched_main_id
        and item.matched_main_basename
    )
