from pathlib import Path

from PIL import Image
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session

from app.db import Base
from app.models import Batch, MainImage, MatchedFile
from app.services.rename_plan import RenamePlanService


def make_image(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20), "red").save(path)


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
                status="MATCHED",
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
                status="MATCHED",
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
            status="MATCHED",
        ))
        db.commit()

        plan = RenamePlanService().generate(db, batch.id)
        result = RenamePlanService().execute(db, batch.id, plan["plan_group"])

        assert result["counts"] == {"RENAMED": 1}
        assert result["workspace_cleanup"]["removed"] is True
        assert not matcher_root.exists()
        assert (root / "观赛图" / "SKU.jpg").is_file()


def test_auto_rename_failure_rolls_back_and_marks_visible_conflicts(
    tmp_path, monkeypatch
):
    import app.services.rename_plan as rename_module

    root = tmp_path / "rollback"
    sources = [root / "观赛图" / f"source-{index}.jpg" for index in range(2)]
    for source in sources:
        make_image(source)
    engine = create_engine(f"sqlite:///{tmp_path / 'rollback.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        batch = Batch(root_path=str(root))
        db.add(batch)
        db.flush()
        for index, source in enumerate(sources):
            main = MainImage(
                batch_id=batch.id,
                basename=f"SKU-{index}",
                filename=f"SKU-{index}.jpg",
                file_path=str(root / "主图" / f"SKU-{index}.jpg"),
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
                status="MATCHED",
            ))
        db.commit()
        plan = RenamePlanService().generate(db, batch.id)
        real_rename = rename_module.os.rename
        calls = 0

        def fail_first_final_move(source, target):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise OSError("simulated rename failure")
            return real_rename(source, target)

        monkeypatch.setattr(rename_module.os, "rename", fail_first_final_move)
        result = RenamePlanService().execute_ready(
            db, batch.id, plan["plan_group"]
        )

        assert result["counts"] == {"RENAME_CONFLICT": 2}
        assert all(source.is_file() for source in sources)
        assert not any((source.parent / f"SKU-{index}.jpg").exists()
                       for index, source in enumerate(sources))
        items = db.query(MatchedFile).all()
        assert all(item.is_conflict and item.status == "CONFLICT" for item in items)
