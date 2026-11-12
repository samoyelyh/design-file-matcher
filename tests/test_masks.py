from __future__ import annotations

from PIL import Image
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.db import Base
from app.models import Batch, MainImage, MaskProfile, MatchedFile
from app.services.masks import (
    mask_readiness,
    profile_storage_valid,
    save_main_profile_version,
    save_profile_version,
)
from app.services.scanner import scan_batch
from tests.support import mask_data_url


def image(path, color=(200, 20, 20)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (80, 60), color).save(path)


def test_mask_profile_is_scoped_and_versioned(tmp_path):
    root = tmp_path / "project"
    image(root / "主图" / "SKU.jpg")
    image(root / "观赛图" / "scene.jpg")
    engine = create_engine(f"sqlite:///{tmp_path / 'mask.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        batch = Batch(root_path=str(root), product_category="橄榄球衣")
        db.add(batch)
        db.commit()
        scan_batch(db, batch.id)
        reference = db.scalar(select(MatchedFile))
        assert mask_readiness(db, batch.id)["missing_categories"] == [
            "主图", "观赛图"
        ]
        profile1 = save_profile_version(
            db,
            batch.id,
            "观赛图",
            mask_data_url((80, 60), (20, 10, 60, 50)),
            reference.id,
            (0.2, 0.1, 0.8, 0.9),
        )
        assert profile1.version == 1
        assert profile1.enabled is True
        assert profile1.product_category == "橄榄球衣"
        assert profile1.creation_method == "MANUAL"
        assert profile1.crop_left == 0.2
        assert profile1.crop_bottom == 0.9
        assert mask_readiness(db, batch.id)["missing_categories"] == ["主图"]

        main = db.scalar(select(MainImage))
        main_profile, _ = save_main_profile_version(
            db,
            batch.id,
            mask_data_url((80, 60)),
            main.id,
            (0.0, 0.0, 1.0, 1.0),
        )
        db.commit()
        assert main_profile.reference_main_id == main.id
        assert mask_readiness(db, batch.id)["ready"] is True

        profile2 = save_profile_version(
            db,
            batch.id,
            "观赛图",
            mask_data_url((80, 60), (10, 5, 70, 55)),
            reference.id,
            (0.1, 0.05, 0.9, 0.95),
        )
        db.refresh(profile1)
        assert profile1.enabled is False
        assert profile2.version == 2 and profile2.enabled is True
        assert len(db.scalars(select(MaskProfile)).all()) == 3


def test_psd_never_creates_mask_profile(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'mask.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        batch = Batch(root_path=str(tmp_path), product_category="橄榄球衣")
        db.add(batch)
        db.commit()
        psd = MatchedFile(
            batch_id=batch.id,
            category="PSD",
            original_filename="source.psd",
            original_path=str(tmp_path / "source.psd"),
            current_filename="source.psd",
            current_path=str(tmp_path / "source.psd"),
            extension=".psd",
        )
        db.add(psd)
        db.commit()
        try:
            save_profile_version(
                db, batch.id, "PSD", mask_data_url(), psd.id, (0, 0, 1, 1)
            )
        except ValueError as exc:
            assert "PSD" in str(exc)
        else:
            raise AssertionError("PSD must not call the mask processor")


def test_mask_files_are_isolated_between_batches_with_the_same_root(tmp_path):
    root = tmp_path / "shared-project"
    root.mkdir()
    engine = create_engine(f"sqlite:///{tmp_path / 'isolated-mask.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        batches = [Batch(root_path=str(root)), Batch(root_path=str(root))]
        db.add_all(batches)
        db.flush()
        references = []
        for batch in batches:
            reference = MatchedFile(
                batch_id=batch.id,
                category="观赛图",
                original_filename="scene.jpg",
                original_path=str(root / "scene.jpg"),
                current_filename="scene.jpg",
                current_path=str(root / "scene.jpg"),
                extension=".jpg",
            )
            db.add(reference)
            references.append(reference)
        db.flush()

        first = save_profile_version(
            db,
            batches[0].id,
            "观赛图",
            mask_data_url(),
            references[0].id,
            (0.0, 0.0, 1.0, 1.0),
        )
        second = save_profile_version(
            db,
            batches[1].id,
            "观赛图",
            mask_data_url(),
            references[1].id,
            (0.0, 0.0, 1.0, 1.0),
        )

        assert first.version == second.version == 1
        assert first.mask_path != second.mask_path
        assert f"batch_{batches[0].id}" in first.mask_path
        assert f"batch_{batches[1].id}" in second.mask_path
        assert profile_storage_valid(db, first) is True
        assert profile_storage_valid(db, second) is True

        # Legacy rows could point to the same physical mask. In that case only
        # the most recently saved row is trustworthy and the older one must be
        # configured again.
        second.mask_path = first.mask_path
        db.flush()
        assert profile_storage_valid(db, first) is False
        assert profile_storage_valid(db, second) is True
