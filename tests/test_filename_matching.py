from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.db import Base
from app.models import Batch, MainImage, MatchedFile
from app.services.filename_matching import associate_filenames, filename_stem
from app.services.matcher import match_batch


@pytest.fixture
def db():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Batch(id=1, root_path="unused", status="SCANNED"))
        session.commit()
        yield session


def main(db, filename, basename=None):
    item = MainImage(batch_id=1, filename=filename, basename=basename or filename_stem(filename), file_path=filename)
    db.add(item)
    db.flush()
    return item


def source(db, filename):
    item = MatchedFile(batch_id=1, category="details", original_filename=filename,
                       current_filename=filename, original_path=filename,
                       current_path=filename, extension=".jpg")
    db.add(item)
    db.flush()
    return item


@pytest.mark.parametrize("filename", ["A001.jpg", "A001.png", "A001.JPG"])
def test_exact_stem_ignores_extension(db, filename):
    target = main(db, "A001.jpg")
    item = source(db, filename)
    assert associate_filenames(db, 1) == 1
    assert item.matched_main_id == target.id
    assert item.match_method == "EXACT_FILENAME"
    assert item.algorithm_status == "EXACT_FILENAME_MATCH"
    assert item.top1_score is None and item.algorithm_score is None


@pytest.mark.parametrize("filename", ["A001-1.jpg", "A001_1.jpg", "A001 copy.jpg", "a001.jpg", "A001 (1).jpg"])
def test_stem_is_not_fuzzily_normalized(db, filename):
    main(db, "A001.jpg")
    item = source(db, filename)
    assert associate_filenames(db, 1) == 0
    assert item.matched_main_id is None


def test_duplicate_main_stems_are_ambiguous(db):
    main(db, "folder1/A001.jpg")
    main(db, "folder2/A001.png", "A001.png")
    item = source(db, "A001.jpg")
    associate_filenames(db, 1)
    assert item.matched_main_id is None
    assert item.match_method == "FILENAME_MATCH_AMBIGUOUS"


def test_mixed_batch_sends_only_three_files_to_visual_engine(db):
    for i in range(10):
        main(db, f"A{i}.jpg")
        source(db, f"A{i}.png" if i < 7 else f"unknown{i}.png")
    associate_filenames(db, 1)
    assert len([f for f in db.scalars(select(MatchedFile)) if f.match_method == "EXACT_FILENAME"]) == 7
    class StopAfterCapture(Exception):
        pass
    class FakeEngine:
        name = "PHASH"
        def config_snapshot(self):
            return {}
        def build_score_matrices(self, context):
            assert len(context.files) == 3
            assert all(f.original_filename.startswith("unknown") for f in context.files)
            raise StopAfterCapture()
    db.commit()
    with patch("app.services.matcher.require_manual_profiles", return_value=[]), patch("app.services.matcher.get_match_engine", return_value=FakeEngine()):
        with pytest.raises(StopAfterCapture):
            match_batch(db, 1)


def test_all_exact_never_loads_visual_model(db):
    main(db, "A001.jpg")
    source(db, "A001.png")
    db.commit()
    with patch("app.services.matcher.get_match_engine", side_effect=AssertionError("must not load")):
        result = match_batch(db, 1)
    assert result["exact_filename_count"] == 1


def test_scan_associates_before_matching(db, tmp_path):
    from PIL import Image
    from app.services.scanner import scan_batch
    from app.services.masks import mask_readiness
    root = tmp_path / "design"
    for folder, names in {"主图": ["A001.jpg", "A002.jpg"], "细节": ["A001.png", "A002.png"]}.items():
        directory = root / folder
        directory.mkdir(parents=True)
        for name in names:
            Image.new("RGB", (32, 32), "red").save(directory / name)
    db.get(Batch, 1).root_path = str(root)
    db.commit()
    assert scan_batch(db, 1)["exact_filename_count"] == 2
    assert mask_readiness(db, 1)["ready"]
    assert scan_batch(db, 1)["exact_filename_count"] == 0


def test_scanner_preserves_duplicate_mains_as_ambiguous(db, tmp_path):
    from PIL import Image
    from app.services.scanner import scan_batch
    root = tmp_path / "design"
    for folder, names in {"主图": ["A001.jpg", "A001.png"], "细节": ["A001.png"]}.items():
        directory = root / folder
        directory.mkdir(parents=True)
        for name in names:
            Image.new("RGB", (32, 32), "red").save(directory / name)
    db.get(Batch, 1).root_path = str(root)
    db.commit()
    result = scan_batch(db, 1)
    assert result["main_count"] == 2
    item = db.scalar(select(MatchedFile))
    assert item.match_method == "FILENAME_MATCH_AMBIGUOUS"
    assert item.matched_main_id is None


def test_visual_assignment_cannot_take_exact_filename_slot(db, tmp_path):
    from PIL import Image, ImageDraw
    from app.services.scanner import scan_batch
    from tests.support import configure_manual_masks
    root = tmp_path / "mixed"
    for directory in (root / "主图", root / "细节"):
        directory.mkdir(parents=True)
    for index, name in enumerate(["A001", "A002"]):
        image = Image.new("RGB", (64, 64), "white")
        ImageDraw.Draw(image).rectangle((10 + index * 5, 8, 50, 48), fill="red")
        image.save(root / "主图" / f"{name}.png")
        image.save(root / "细节" / ("A001.jpg" if index == 0 else "unknown.jpg"))
    batch = db.get(Batch, 1)
    batch.root_path = str(root)
    batch.match_engine = "PHASH"
    db.commit()
    scan_batch(db, 1)
    configure_manual_masks(db, 1)
    match_batch(db, 1)
    files = list(db.scalars(select(MatchedFile)))
    exact = next(f for f in files if f.original_filename == "A001.jpg")
    visual = next(f for f in files if f.original_filename == "unknown.jpg")
    assert exact.match_method == "EXACT_FILENAME"
    assert not exact.candidates and exact.algorithm_score is None
    assert visual.matched_main_id != exact.matched_main_id
    assert visual.matched_main_basename == "A002"
