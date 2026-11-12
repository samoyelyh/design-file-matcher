from app.services.pair_cache import PairResultCache


def test_pair_cache_reuses_scores_and_invalidates_inputs(tmp_path):
    query = tmp_path / "query.jpg"
    main = tmp_path / "main.jpg"
    query.write_bytes(b"query")
    main.write_bytes(b"main")
    path = tmp_path / "cache.json"
    cache = PairResultCache(path, {"mask": 1})
    key = cache.key(str(query), str(main))
    row = dict(raw_match_count=10, ransac_inlier_count=8, inlier_ratio=.8,
               mean_match_confidence=.9, coverage_score=.5, local_score=.7)
    cache.rows[key] = row
    cache.save()
    restored = PairResultCache(path, {"mask": 1})
    assert restored.get(restored.key(str(query), str(main))) == row
    changed_mask = PairResultCache(path, {"mask": 2})
    assert changed_mask.get(changed_mask.key(str(query), str(main))) is None
    query.write_bytes(b"new query content")
    changed_image = PairResultCache(path, {"mask": 1})
    assert changed_image.get(changed_image.key(str(query), str(main))) is None


def test_corrupt_cache_is_treated_as_miss(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text("broken")
    assert PairResultCache(path, {}).get("missing") is None
