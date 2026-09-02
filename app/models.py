from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Batch(Base):
    __tablename__ = "batches"

    id: Mapped[int] = mapped_column(primary_key=True)
    root_path: Mapped[str] = mapped_column(Text)
    product_category: Mapped[str] = mapped_column(String(128), default="橄榄球衣")
    status: Mapped[str] = mapped_column(String(32), default="CREATED", index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    main_images: Mapped[list[MainImage]] = relationship(
        back_populates="batch", cascade="all, delete-orphan"
    )
    matched_files: Mapped[list[MatchedFile]] = relationship(
        back_populates="batch", cascade="all, delete-orphan"
    )
    matching_runs: Mapped[list[MatchingRun]] = relationship(
        back_populates="batch", cascade="all, delete-orphan"
    )


class MatchingRun(Base):
    __tablename__ = "matching_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    batch_id: Mapped[int] = mapped_column(
        ForeignKey("batches.id", ondelete="CASCADE"), index=True
    )
    category: Mapped[str] = mapped_column(String(128), index=True)
    algorithm_version: Mapped[str] = mapped_column(String(64))
    model_name: Mapped[str] = mapped_column(String(128))
    match_min_score: Mapped[float] = mapped_column(Float)
    match_min_margin: Mapped[float] = mapped_column(Float)
    unmatched_score: Mapped[float] = mapped_column(Float)
    mask_profile_id: Mapped[int | None] = mapped_column(
        ForeignKey("mask_profiles.id", ondelete="SET NULL"), nullable=True
    )
    feature_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    image_count: Mapped[int] = mapped_column(Integer)
    product_count: Mapped[int] = mapped_column(Integer)
    summary_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    batch: Mapped[Batch] = relationship(back_populates="matching_runs")


class BatchProgress(Base):
    __tablename__ = "batch_progress"

    batch_id: Mapped[int] = mapped_column(
        ForeignKey("batches.id", ondelete="CASCADE"), primary_key=True
    )
    stage: Mapped[str] = mapped_column(String(32), default="READY")
    current: Mapped[int] = mapped_column(Integer, default=0)
    total: Mapped[int] = mapped_column(Integer, default=0)
    percent: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    device: Mapped[str | None] = mapped_column(String(128), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class MainImage(Base):
    __tablename__ = "main_images"
    __table_args__ = (Index("ix_main_batch_basename", "batch_id", "basename", unique=True),)

    id: Mapped[int] = mapped_column(primary_key=True)
    batch_id: Mapped[int] = mapped_column(ForeignKey("batches.id", ondelete="CASCADE"), index=True)
    basename: Mapped[str] = mapped_column(String(255))
    filename: Mapped[str] = mapped_column(String(512))
    file_path: Mapped[str] = mapped_column(Text)
    thumbnail_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    body_bbox_x1: Mapped[float | None] = mapped_column(Float, nullable=True)
    body_bbox_y1: Mapped[float | None] = mapped_column(Float, nullable=True)
    body_bbox_x2: Mapped[float | None] = mapped_column(Float, nullable=True)
    body_bbox_y2: Mapped[float | None] = mapped_column(Float, nullable=True)
    body_mask_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    body_crop_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    body_crop_method: Mapped[str | None] = mapped_column(String(32), nullable=True)
    body_source_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    body_source_mtime_ns: Mapped[int | None] = mapped_column(Integer, nullable=True)
    body_feature_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    body_updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    batch: Mapped[Batch] = relationship(back_populates="main_images")
    candidates: Mapped[list[MatchCandidate]] = relationship(back_populates="main_image")
    phash_regions: Mapped[list[ImagePHashRegion]] = relationship(
        back_populates="main_image", cascade="all, delete-orphan"
    )


class MatchedFile(Base):
    __tablename__ = "matched_files"

    id: Mapped[int] = mapped_column(primary_key=True)
    batch_id: Mapped[int] = mapped_column(ForeignKey("batches.id", ondelete="CASCADE"), index=True)
    category: Mapped[str] = mapped_column(String(128), index=True)
    original_filename: Mapped[str] = mapped_column(String(512))
    original_path: Mapped[str] = mapped_column(Text)
    current_filename: Mapped[str] = mapped_column(String(512))
    current_path: Mapped[str] = mapped_column(Text)
    extension: Mapped[str] = mapped_column(String(32))
    thumbnail_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    matched_main_id: Mapped[int | None] = mapped_column(
        ForeignKey("main_images.id", ondelete="SET NULL"), nullable=True, index=True
    )
    matched_main_basename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    top1_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    top2_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    top3_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    algorithm_main_id: Mapped[int | None] = mapped_column(
        ForeignKey("main_images.id", ondelete="SET NULL"), nullable=True, index=True
    )
    algorithm_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    image_best_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    image_second_best_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    image_best_main_id: Mapped[int | None] = mapped_column(
        ForeignKey("main_images.id", ondelete="SET NULL"), nullable=True
    )
    assigned_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    row_competitor_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    row_margin: Mapped[float | None] = mapped_column(Float, nullable=True)
    column_competitor_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    column_margin: Mapped[float | None] = mapped_column(Float, nullable=True)
    algorithm_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    algorithm_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    matching_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("matching_runs.id", ondelete="SET NULL"), nullable=True, index=True
    )
    manual_product_id: Mapped[int | None] = mapped_column(
        ForeignKey("main_images.id", ondelete="SET NULL"), nullable=True
    )
    operator: Mapped[str | None] = mapped_column(String(128), nullable=True)
    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    preview_asset_id: Mapped[int | None] = mapped_column(
        ForeignKey("derived_image_assets.id", ondelete="SET NULL"), nullable=True
    )
    phash_distance: Mapped[int | None] = mapped_column(Integer, nullable=True)
    phash_value: Mapped[str | None] = mapped_column(String(16), nullable=True)
    phash_source_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    phash_source_mtime_ns: Mapped[int | None] = mapped_column(Integer, nullable=True)
    phash_best_region_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    match_method: Mapped[str | None] = mapped_column(String(32), nullable=True)
    baseline_top1_main_id: Mapped[int | None] = mapped_column(
        ForeignKey("main_images.id", ondelete="SET NULL"), nullable=True
    )
    baseline_top1_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    baseline_top2_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="PENDING", index=True)
    is_conflict: Mapped[bool] = mapped_column(Boolean, default=False)
    is_low_confidence: Mapped[bool] = mapped_column(Boolean, default=False)
    review_status: Mapped[str] = mapped_column(String(32), default="PENDING", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    renamed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    batch: Mapped[Batch] = relationship(back_populates="matched_files")
    matched_main: Mapped[MainImage | None] = relationship(foreign_keys=[matched_main_id])
    algorithm_main: Mapped[MainImage | None] = relationship(foreign_keys=[algorithm_main_id])
    image_best_main: Mapped[MainImage | None] = relationship(foreign_keys=[image_best_main_id])
    manual_product: Mapped[MainImage | None] = relationship(foreign_keys=[manual_product_id])
    baseline_top1_main: Mapped[MainImage | None] = relationship(
        foreign_keys=[baseline_top1_main_id]
    )
    candidates: Mapped[list[MatchCandidate]] = relationship(
        back_populates="matched_file", cascade="all, delete-orphan", order_by="MatchCandidate.rank"
    )
    operations: Mapped[list[RenameOperation]] = relationship(
        back_populates="matched_file", cascade="all, delete-orphan"
    )


class MatchCandidate(Base):
    __tablename__ = "match_candidates"
    __table_args__ = (Index("ix_candidate_file_rank", "matched_file_id", "rank", unique=True),)

    id: Mapped[int] = mapped_column(primary_key=True)
    matched_file_id: Mapped[int] = mapped_column(
        ForeignKey("matched_files.id", ondelete="CASCADE"), index=True
    )
    main_image_id: Mapped[int] = mapped_column(
        ForeignKey("main_images.id", ondelete="CASCADE"), index=True
    )
    rank: Mapped[int] = mapped_column(Integer)
    similarity: Mapped[float] = mapped_column(Float)

    matched_file: Mapped[MatchedFile] = relationship(back_populates="candidates")
    main_image: Mapped[MainImage] = relationship(back_populates="candidates")


class RenameOperation(Base):
    __tablename__ = "rename_operations"

    id: Mapped[int] = mapped_column(primary_key=True)
    matched_file_id: Mapped[int] = mapped_column(
        ForeignKey("matched_files.id", ondelete="CASCADE"), index=True
    )
    operation: Mapped[str] = mapped_column(String(32))
    from_path: Mapped[str] = mapped_column(Text)
    to_path: Mapped[str] = mapped_column(Text)
    success: Mapped[bool] = mapped_column(Boolean)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    matched_file: Mapped[MatchedFile] = relationship(back_populates="operations")


class MaskProfile(Base):
    __tablename__ = "mask_profiles"
    __table_args__ = (
        Index(
            "ix_mask_profile_scope_version",
            "batch_id", "product_category", "image_category", "version",
            unique=True,
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    batch_id: Mapped[int] = mapped_column(
        ForeignKey("batches.id", ondelete="CASCADE"), index=True
    )
    product_category: Mapped[str] = mapped_column(String(128), index=True)
    image_category: Mapped[str] = mapped_column(String(128), index=True)
    name: Mapped[str] = mapped_column(String(255))
    version: Mapped[int] = mapped_column(Integer)
    reference_width: Mapped[int] = mapped_column(Integer)
    reference_height: Mapped[int] = mapped_column(Integer)
    reference_file_id: Mapped[int | None] = mapped_column(
        ForeignKey("matched_files.id", ondelete="SET NULL"), nullable=True
    )
    mask_path: Mapped[str] = mapped_column(Text)
    crop_left: Mapped[float | None] = mapped_column(Float, nullable=True)
    crop_top: Mapped[float | None] = mapped_column(Float, nullable=True)
    crop_right: Mapped[float | None] = mapped_column(Float, nullable=True)
    crop_bottom: Mapped[float | None] = mapped_column(Float, nullable=True)
    creation_method: Mapped[str] = mapped_column(String(32), default="MANUAL")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class DerivedImageAsset(Base):
    __tablename__ = "derived_image_assets"

    id: Mapped[int] = mapped_column(primary_key=True)
    batch_id: Mapped[int] = mapped_column(
        ForeignKey("batches.id", ondelete="CASCADE"), index=True
    )
    parent_file_id: Mapped[int | None] = mapped_column(
        ForeignKey("matched_files.id", ondelete="CASCADE"), nullable=True, index=True
    )
    parent_main_image_id: Mapped[int | None] = mapped_column(
        ForeignKey("main_images.id", ondelete="CASCADE"), nullable=True, index=True
    )
    asset_type: Mapped[str] = mapped_column(String(32), index=True)
    asset_path: Mapped[str] = mapped_column(Text)
    mask_profile_id: Mapped[int | None] = mapped_column(
        ForeignKey("mask_profiles.id", ondelete="SET NULL"), nullable=True
    )
    feature_version: Mapped[str] = mapped_column(String(64))
    phash_value: Mapped[str | None] = mapped_column(String(16), nullable=True)
    phash_source_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    phash_source_mtime_ns: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class MainImageFeature(Base):
    __tablename__ = "main_image_features"
    __table_args__ = (
        Index(
            "ix_main_feature_version",
            "main_image_id", "feature_type", "feature_version", "mask_profile_id",
            unique=True,
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    main_image_id: Mapped[int] = mapped_column(
        ForeignKey("main_images.id", ondelete="CASCADE"), index=True
    )
    feature_type: Mapped[str] = mapped_column(String(32), index=True)
    feature_version: Mapped[str] = mapped_column(String(64))
    model_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    mask_profile_id: Mapped[int | None] = mapped_column(
        ForeignKey("mask_profiles.id", ondelete="SET NULL"), nullable=True
    )
    phash_value: Mapped[str | None] = mapped_column(String(16), nullable=True)
    feature_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    crop_asset_id: Mapped[int | None] = mapped_column(
        ForeignKey("derived_image_assets.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ImagePHashRegion(Base):
    __tablename__ = "image_phash_regions"
    __table_args__ = (
        Index(
            "ix_phash_region_main_type",
            "main_image_id",
            "region_type",
            unique=True,
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    main_image_id: Mapped[int] = mapped_column(
        ForeignKey("main_images.id", ondelete="CASCADE"), index=True
    )
    region_type: Mapped[str] = mapped_column(String(32), index=True)
    x1: Mapped[float] = mapped_column(Float)
    y1: Mapped[float] = mapped_column(Float)
    x2: Mapped[float] = mapped_column(Float)
    y2: Mapped[float] = mapped_column(Float)
    phash_value: Mapped[str] = mapped_column(String(16))
    feature_version: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    main_image: Mapped[MainImage] = relationship(back_populates="phash_regions")


class RenamePlan(Base):
    __tablename__ = "rename_plans"

    id: Mapped[int] = mapped_column(primary_key=True)
    plan_group: Mapped[str] = mapped_column(String(36), index=True)
    batch_id: Mapped[int] = mapped_column(
        ForeignKey("batches.id", ondelete="CASCADE"), index=True
    )
    source_file_id: Mapped[int] = mapped_column(
        ForeignKey("matched_files.id", ondelete="CASCADE"), index=True
    )
    source_path: Mapped[str] = mapped_column(Text)
    original_filename: Mapped[str] = mapped_column(String(512))
    matched_main_image_id: Mapped[int | None] = mapped_column(
        ForeignKey("main_images.id", ondelete="SET NULL"), nullable=True
    )
    target_filename: Mapped[str | None] = mapped_column(String(512), nullable=True)
    target_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    category: Mapped[str] = mapped_column(String(128), index=True)
    match_status: Mapped[str] = mapped_column(String(32))
    rename_status: Mapped[str] = mapped_column(String(32), index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    executed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
