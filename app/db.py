from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, event, inspect, select, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import settings


if settings.database_url.startswith("sqlite:///"):
    db_path = Path(settings.database_url.removeprefix("sqlite:///"))
    db_path.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if settings.database_url.startswith("sqlite") else {},
)
if settings.database_url.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    from . import models

    Base.metadata.create_all(engine)
    if settings.database_url.startswith("sqlite"):
        _migrate_sqlite_schema()
    # A background thread cannot survive a server restart. Make interrupted
    # batches safely resumable instead of leaving the UI stuck on MATCHING.
    with SessionLocal() as db:
        interrupted = db.scalars(select(models.Batch).where(
            models.Batch.status.in_(["QUEUED", "MATCHING"])
        )).all()
        for batch in interrupted:
            batch.status = "SCANNED"
            batch.error_message = "上一次匹配因服务重启中断，可以重新开始"
            progress = db.get(models.BatchProgress, batch.id)
            if progress:
                progress.stage = "READY"
                progress.current = 0
                progress.total = 0
                progress.percent = 0
                progress.message = "等待重新开始"
        db.commit()


def _migrate_sqlite_schema() -> None:
    """Small additive migration layer for existing local, single-file installs."""
    columns = {column["name"] for column in inspect(engine).get_columns("matched_files")}
    additions = {
        "algorithm_main_id": "INTEGER REFERENCES main_images(id) ON DELETE SET NULL",
        "algorithm_score": "FLOAT",
        "image_best_score": "FLOAT",
        "image_second_best_score": "FLOAT",
        "image_best_main_id": "INTEGER REFERENCES main_images(id) ON DELETE SET NULL",
        "assigned_rank": "INTEGER",
        "row_competitor_score": "FLOAT",
        "row_margin": "FLOAT",
        "column_competitor_score": "FLOAT",
        "column_margin": "FLOAT",
        "algorithm_status": "VARCHAR(32)",
        "algorithm_version": "VARCHAR(64)",
        "matching_run_id": "INTEGER REFERENCES matching_runs(id) ON DELETE SET NULL",
        "manual_product_id": "INTEGER REFERENCES main_images(id) ON DELETE SET NULL",
        "operator": "VARCHAR(128)",
        "confirmed_at": "DATETIME",
        "preview_asset_id": "INTEGER REFERENCES derived_image_assets(id) ON DELETE SET NULL",
        "phash_distance": "INTEGER",
        "phash_value": "VARCHAR(16)",
        "phash_source_size": "INTEGER",
        "phash_source_mtime_ns": "INTEGER",
        "phash_best_region_type": "VARCHAR(32)",
        "match_method": "VARCHAR(32)",
        "baseline_top1_main_id": "INTEGER REFERENCES main_images(id) ON DELETE SET NULL",
        "baseline_top1_score": "FLOAT",
        "baseline_top2_score": "FLOAT",
    }
    with engine.begin() as connection:
        for name, sql_type in additions.items():
            if name not in columns:
                connection.execute(text(
                    f'ALTER TABLE matched_files ADD COLUMN "{name}" {sql_type}'
                ))
        connection.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_matched_files_algorithm_main_id "
            "ON matched_files (algorithm_main_id)"
        ))
        connection.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_matched_files_matching_run_id "
            "ON matched_files (matching_run_id)"
        ))
        main_columns = {
            column["name"] for column in inspect(engine).get_columns("main_images")
        }
        main_additions = {
            "body_bbox_x1": "FLOAT",
            "body_bbox_y1": "FLOAT",
            "body_bbox_x2": "FLOAT",
            "body_bbox_y2": "FLOAT",
            "body_mask_path": "TEXT",
            "body_crop_path": "TEXT",
            "body_crop_method": "VARCHAR(32)",
            "body_source_size": "INTEGER",
            "body_source_mtime_ns": "INTEGER",
            "body_feature_version": "VARCHAR(64)",
            "body_updated_at": "DATETIME",
        }
        for name, sql_type in main_additions.items():
            if name not in main_columns:
                connection.execute(text(
                    f'ALTER TABLE main_images ADD COLUMN "{name}" {sql_type}'
                ))
        asset_columns = {
            column["name"]
            for column in inspect(engine).get_columns("derived_image_assets")
        }
        asset_additions = {
            "phash_value": "VARCHAR(16)",
            "phash_source_size": "INTEGER",
            "phash_source_mtime_ns": "INTEGER",
        }
        for name, sql_type in asset_additions.items():
            if name not in asset_columns:
                connection.execute(text(
                    f'ALTER TABLE derived_image_assets ADD COLUMN "{name}" {sql_type}'
                ))
        batch_columns = {
            column["name"] for column in inspect(engine).get_columns("batches")
        }
        if "product_category" not in batch_columns:
            connection.execute(text(
                "ALTER TABLE batches ADD COLUMN product_category VARCHAR(128) "
                "DEFAULT '橄榄球衣' NOT NULL"
            ))
        run_columns = {
            column["name"] for column in inspect(engine).get_columns("matching_runs")
        }
        if "mask_profile_id" not in run_columns:
            connection.execute(text(
                "ALTER TABLE matching_runs ADD COLUMN mask_profile_id INTEGER "
                "REFERENCES mask_profiles(id) ON DELETE SET NULL"
            ))
        if "feature_version" not in run_columns:
            connection.execute(text(
                "ALTER TABLE matching_runs ADD COLUMN feature_version VARCHAR(64)"
            ))
        rename_columns = {
            column["name"] for column in inspect(engine).get_columns("rename_plans")
        }
        if "plan_group" not in rename_columns:
            connection.execute(text(
                "ALTER TABLE rename_plans ADD COLUMN plan_group VARCHAR(36)"
            ))
            connection.execute(text(
                "UPDATE rename_plans SET plan_group = 'legacy-' || batch_id "
                "WHERE plan_group IS NULL"
            ))
        connection.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_rename_plans_plan_group "
            "ON rename_plans (plan_group)"
        ))
        mask_columns = {
            column["name"] for column in inspect(engine).get_columns("mask_profiles")
        }
        if "creation_method" not in mask_columns:
            connection.execute(text(
                "ALTER TABLE mask_profiles ADD COLUMN creation_method VARCHAR(32) "
                "DEFAULT 'LEGACY_PRESET' NOT NULL"
            ))
        if "reference_main_id" not in mask_columns:
            connection.execute(text(
                "ALTER TABLE mask_profiles ADD COLUMN reference_main_id INTEGER "
                "REFERENCES main_images(id) ON DELETE SET NULL"
            ))
        crop_columns = {
            "crop_left": "FLOAT",
            "crop_top": "FLOAT",
            "crop_right": "FLOAT",
            "crop_bottom": "FLOAT",
        }
        for name, sql_type in crop_columns.items():
            if name not in mask_columns:
                connection.execute(text(
                    f'ALTER TABLE mask_profiles ADD COLUMN "{name}" {sql_type}'
                ))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
