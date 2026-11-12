from __future__ import annotations

import numpy as np

from app.services.lightglue_matching import LightGlueRuntime


def test_coverage_uses_three_by_three_spread_not_only_match_count():
    clustered = np.asarray([[5, 5], [6, 6], [7, 7], [8, 8]], dtype=np.float32)
    spread = np.asarray([
        [5, 5], [50, 5], [95, 5],
        [5, 50], [50, 50], [95, 50],
        [5, 95], [50, 95], [95, 95],
    ], dtype=np.float32)
    size = np.asarray([100, 100], dtype=np.float32)
    clustered_score = LightGlueRuntime._coverage(clustered, size, clustered, size)
    spread_score = LightGlueRuntime._coverage(spread, size, spread, size)
    assert clustered_score == 1 / 9
    assert spread_score == 1.0
