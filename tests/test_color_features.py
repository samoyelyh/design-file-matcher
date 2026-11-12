from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from app.models import Batch, MainImage, MaskProfile, MatchedFile
from app.services.color_features import build_color_matrix, color_top_n_recall


def _profile(path: Path, profile_id: int, category: str) -> MaskProfile:
    Image.new("L", (80, 80), 255).save(path)
    return MaskProfile(
        id=profile_id,
        batch_id=1,
        product_category="",
        image_category=category,
        name=category,
        version=1,
        reference_width=80,
        reference_height=80,
        mask_path=str(path),
        crop_left=0,
        crop_top=0,
        crop_right=1,
        crop_bottom=1,
        creation_method="MANUAL",
    )


def test_hsv_lab_color_matrix_prefers_matching_color_and_caches(tmp_path):
    root = tmp_path / "batch"
    root.mkdir()
    red_main = root / "red-main.png"
    blue_main = root / "blue-main.png"
    red_query = root / "red-query.png"
    Image.new("RGB", (80, 80), (210, 25, 35)).save(red_main)
    Image.new("RGB", (80, 80), (20, 65, 210)).save(blue_main)
    Image.new("RGB", (80, 80), (195, 30, 42)).save(red_query)
    main_mask = _profile(root / "main-mask.png", 1, "主图")
    query_mask = _profile(root / "query-mask.png", 2, "观赛图")
    batch = Batch(id=1, root_path=str(root))
    mains = [
        MainImage(id=1, batch_id=1, basename="red", filename=red_main.name, file_path=str(red_main)),
        MainImage(id=2, batch_id=1, basename="blue", filename=blue_main.name, file_path=str(blue_main)),
    ]
    files = [MatchedFile(
        id=10, batch_id=1, category="观赛图", original_filename=red_query.name,
        original_path=str(red_query), current_filename=red_query.name,
        current_path=str(red_query), extension=".png",
    )]
    matrix, first = build_color_matrix(batch, files, mains, query_mask, main_mask)
    repeated, second = build_color_matrix(batch, files, mains, query_mask, main_mask)
    assert matrix.shape == (1, 2)
    assert matrix[0, 0] > matrix[0, 1]
    assert first["cache_hits"] == 0
    assert second["cache_hits"] == second["cache_total"] == 3
    np.testing.assert_allclose(matrix, repeated)
    assert color_top_n_recall(matrix, [0], 1)["recall"] == 1.0
