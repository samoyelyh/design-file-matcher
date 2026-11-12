from pathlib import Path

import pytest
from PIL import Image
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session

from app.db import Base
from app.models import Batch, MainImage, MatchedFile
from app.services.rename_plan import RenamePlanService


def make_image(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20), "red").save(path)


def test_all_already_named_files_can_finalize_without_renaming_images(tmp_path, monkeypatch):
    root = tmp_path / "project"
    source = root / "观赛图" / "SKU.jpg"
    make_image(source)
    original = source.read_bytes()
    cache = root / ".design_matcher" / "cache.bin"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"remove only after final confirmation")
    engine = create_engine(f"sqlite:///{tmp_path / 'noop.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        batch = Batch(root_path=str(root))
        db.add(batch)
        db.flush()
        main = MainImage(batch_id=batch.id, basename="SKU", filename="SKU.jpg", file_path="main.jpg")
        db.add(main)
        db.flush()
        item = MatchedFile(batch_id=batch.id, category="观赛图", original_filename=source.name,
                           current_filename=source.name, original_path=str(source), current_path=str(source),
                           extension=".jpg", matched_main_id=main.id, matched_main_basename="SKU",
                           status="MANUAL_CONFIRMED", review_status="CONFIRMED")
        db.add(item)
        db.commit()
        plan = RenamePlanService().generate(db, batch.id)
        assert plan["counts"] == {"NOOP": 1}
        assert cache.exists()

        def forbid_rename(*args):
            raise AssertionError("Already correct image names must remain untouched")

        monkeypatch.setattr("app.services.rename_plan.os.rename", forbid_rename)
        result = RenamePlanService().execute(db, batch.id, plan["plan_group"])
        assert result["counts"] == {"RENAMED": 1}
        assert result["workspace_cleanup"]["removed"]
        assert not cache.parent.exists()
        assert source.read_bytes() == original
        assert item.review_status == "CONFIRMED" and item.matched_main_id == main.id
    engine.dispose()


def test_rename_plan_rejects_duplicate_target_paths(tmp_path):
    root = tmp_path / "project"
    one = root / "观赛图" / "one.jpg"
    two = root / "观赛图" / "two.jpg"
    make_image(one)
    make_image(two)
    cache = root / ".design_matcher" / "cache.bin"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"keep until rename succeeds")
    engine = create_engine(f"sqlite:///{tmp_path / 'rename.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        batch = Batch(root_path=str(root), product_category="橄榄球衣")
        db.add(batch)
        db.flush()
        main = MainImage(
            batch_id=batch.id,
            basename="SKU",
            filename="SKU.jpg",
            file_path=str(root / "主图" / "SKU.jpg"),
        )
        db.add(main)
        db.flush()
        for path in (one, two):
            db.add(MatchedFile(
                batch_id=batch.id,
                category="观赛图",
                original_filename=path.name,
                original_path=str(path),
                current_filename=path.name,
                current_path=str(path),
                extension=".jpg",
                matched_main_id=main.id,
                matched_main_basename=main.basename,
                status="MATCHED", review_status="CONFIRMED",
            ))
        db.commit()
        result = RenamePlanService().generate(db, batch.id)
        assert result["counts"] == {"RENAME_CONFLICT": 2}
        try:
            RenamePlanService().execute(db, batch.id, result["plan_group"])
        except ValueError:
            pass
        else:
            raise AssertionError("存在冲突时必须禁止执行改名")
        assert one.is_file() and two.is_file()
        assert cache.is_file()


def test_rename_plan_allows_safe_two_file_name_swap(tmp_path):
    root = tmp_path / "project"
    first = root / "观赛图" / "SKU-1.jpg"
    second = root / "观赛图" / "SKU-2.jpg"
    make_image(first)
    make_image(second)
    engine = create_engine(f"sqlite:///{tmp_path / 'swap.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        batch = Batch(root_path=str(root))
        db.add(batch)
        db.flush()
        mains = []
        for basename in ("SKU-1", "SKU-2"):
            main = MainImage(
                batch_id=batch.id,
                basename=basename,
                filename=f"{basename}.jpg",
                file_path=str(root / "主图" / f"{basename}.jpg"),
            )
            db.add(main)
            mains.append(main)
        db.flush()
        db.add_all([
            MatchedFile(
                batch_id=batch.id,
                category="观赛图",
                original_filename=first.name,
                original_path=str(first),
                current_filename=first.name,
                current_path=str(first),
                extension=".jpg",
                matched_main_id=mains[1].id,
                matched_main_basename=mains[1].basename,
                status="MATCHED", review_status="CONFIRMED",
            ),
            MatchedFile(
                batch_id=batch.id,
                category="观赛图",
                original_filename=second.name,
                original_path=str(second),
                current_filename=second.name,
                current_path=str(second),
                extension=".jpg",
                matched_main_id=mains[0].id,
                matched_main_basename=mains[0].basename,
                status="MATCHED", review_status="CONFIRMED",
            ),
        ])
        db.commit()

        result = RenamePlanService().generate(db, batch.id)
        assert result["counts"] == {"READY": 2}
        executed = RenamePlanService().execute(db, batch.id, result["plan_group"])

        assert executed["counts"] == {"RENAMED": 2}
        assert first.is_file() and second.is_file()
        assert not list(first.parent.glob("*.design_matcher_tmp_*"))


def test_generate_recovers_replaced_original_without_losing_match(tmp_path):
    root = tmp_path / "project"
    original = root / "观赛图" / "original.jpg"
    make_image(original)
    missing_current = root / "观赛图" / "SKU.jpg"
    engine = create_engine(f"sqlite:///{tmp_path / 'recover.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        batch = Batch(root_path=str(root))
        db.add(batch)
        db.flush()
        main = MainImage(
            batch_id=batch.id,
            basename="SKU",
            filename="SKU.jpg",
            file_path=str(root / "主图" / "SKU.jpg"),
        )
        db.add(main)
        db.flush()
        item = MatchedFile(
            batch_id=batch.id,
            category="观赛图",
            original_filename=original.name,
            original_path=str(original),
            current_filename=missing_current.name,
            current_path=str(missing_current),
            extension=".jpg",
            matched_main_id=main.id,
            matched_main_basename=main.basename,
            status="MANUAL_CONFIRMED",
            review_status="CORRECTED",
        )
        db.add(item)
        db.commit()

        result = RenamePlanService().generate(db, batch.id)

        assert result["recovered_source_count"] == 1
        assert result["counts"] == {"READY": 1}
        assert item.current_path == str(original)
        assert item.matched_main_id == main.id
        assert item.review_status == "CORRECTED"


def test_sqlite_migration_restores_legacy_plan_group(tmp_path, monkeypatch):
    import app.db as database

    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(text("DROP INDEX ix_rename_plans_plan_group"))
        connection.execute(text("ALTER TABLE rename_plans DROP COLUMN plan_group"))
    monkeypatch.setattr(database, "engine", engine)
    database._migrate_sqlite_schema()
    columns = {row["name"] for row in inspect(engine).get_columns("rename_plans")}
    assert "plan_group" in columns


def test_execute_preflights_entire_batch_before_first_rename(tmp_path):
    root = tmp_path / "project"
    paths = [root / "观赛图" / f"{name}.jpg" for name in ("one", "two")]
    for path in paths:
        make_image(path)
    engine = create_engine(f"sqlite:///{tmp_path / 'preflight.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        batch = Batch(root_path=str(root), product_category="橄榄球衣")
        db.add(batch)
        db.flush()
        mains = []
        for name in ("SKU-1", "SKU-2"):
            main = MainImage(
                batch_id=batch.id,
                basename=name,
                filename=f"{name}.jpg",
                file_path=str(root / "主图" / f"{name}.jpg"),
            )
            db.add(main)
            mains.append(main)
        db.flush()
        items = []
        for path, main in zip(paths, mains, strict=True):
            item = MatchedFile(
                batch_id=batch.id,
                category="观赛图",
                original_filename=path.name,
                original_path=str(path),
                current_filename=path.name,
                current_path=str(path),
                extension=".jpg",
                matched_main_id=main.id,
                matched_main_basename=main.basename,
                status="MATCHED", review_status="CONFIRMED",
            )
            db.add(item)
            items.append(item)
        db.commit()
        result = RenamePlanService().generate(db, batch.id)
        items[1].current_path = str(root / "观赛图" / "changed.jpg")
        db.commit()
        try:
            RenamePlanService().execute(db, batch.id, result["plan_group"])
        except ValueError as exc:
            assert "源路径已变化" in str(exc)
        else:
            raise AssertionError("执行前必须拒绝已变化的整批计划")
        assert paths[0].is_file()
        assert not (paths[0].parent / "SKU-1.jpg").exists()


def test_successful_rename_removes_design_matcher(tmp_path):
    root = tmp_path / "project"
    source = root / "观赛图" / "source.jpg"
    make_image(source)
    matcher_root = root / ".design_matcher"
    mask_path = matcher_root / "masks" / "scope" / "mask_v1.png"
    make_image(mask_path)
    cache_path = matcher_root / "1" / "embeddings" / "cache.npy"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_bytes(b"cache")
    engine = create_engine(f"sqlite:///{tmp_path / 'cleanup.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        batch = Batch(root_path=str(root), product_category="橄榄球衣")
        db.add(batch)
        db.flush()
        main = MainImage(
            batch_id=batch.id,
            basename="SKU",
            filename="SKU.jpg",
            file_path=str(root / "主图" / "SKU.jpg"),
        )
        db.add(main)
        db.flush()
        db.add(MatchedFile(
            batch_id=batch.id,
            category="观赛图",
            original_filename=source.name,
            original_path=str(source),
            current_filename=source.name,
            current_path=str(source),
            extension=".jpg",
            matched_main_id=main.id,
            matched_main_basename=main.basename,
            status="MATCHED", review_status="CONFIRMED",
        ))
        db.commit()

        plan = RenamePlanService().generate(db, batch.id)
        result = RenamePlanService().execute(db, batch.id, plan["plan_group"])

        assert result["counts"] == {"RENAMED": 1}
        assert result["workspace_cleanup"]["removed"] is True
        assert not matcher_root.exists()
        assert (root / "观赛图" / "SKU.jpg").is_file()


@pytest.mark.parametrize("skip_white", [False, True])
def test_white_background_category_follows_explicit_exclusion_configuration(tmp_path, monkeypatch, skip_white):
    monkeypatch.setattr("app.config.IGNORED_IMAGE_CATEGORY_KEYWORDS", ("白底",) if skip_white else ())
    root = tmp_path / "project"
    source = root / "白底模特图" / "original.jpg"
    make_image(source)
    engine = create_engine(f"sqlite:///{tmp_path / 'ignored.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        batch = Batch(root_path=str(root), product_category="橄榄球衣")
        db.add(batch)
        db.flush()
        main = MainImage(
            batch_id=batch.id,
            basename="SKU",
            filename="SKU.jpg",
            file_path=str(root / "主图" / "SKU.jpg"),
        )
        db.add(main)
        db.flush()
        db.add(MatchedFile(
            batch_id=batch.id,
            category="白底模特图",
            original_filename=source.name,
            original_path=str(source),
            current_filename=source.name,
            current_path=str(source),
            extension=".jpg",
            matched_main_id=main.id,
            matched_main_basename=main.basename,
            status="MATCHED", review_status="CONFIRMED",
        ))
        db.commit()

        plan = RenamePlanService().generate(db, batch.id)

        if skip_white:
            assert plan["plans"] == []
            assert plan["counts"] == {}
            assert source.is_file()
            assert not (source.parent / "SKU.jpg").exists()
        else:
            assert plan["counts"] == {"READY": 1}
            result = RenamePlanService().execute(db, batch.id, plan["plan_group"])
            assert result["counts"] == {"RENAMED": 1}
            assert (source.parent / "SKU.jpg").is_file()
