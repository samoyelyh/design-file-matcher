from __future__ import annotations

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.db import Base
from app.models import Batch, ImagePHashRegion, MainImage, ProductReference
from app.services.references import build_product_score_matrix


def _database(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'references.db'}")
    Base.metadata.create_all(engine)
    return engine


def _hash_with_distance(distance: int) -> str:
    return f"{(1 << distance) - 1:016x}"


def test_multi_reference_consistency_beats_one_accidental_best_match(tmp_path):
    with Session(_database(tmp_path)) as db:
        batch = Batch(root_path=str(tmp_path))
        db.add(batch)
        db.flush()
        stable = MainImage(
            batch_id=batch.id, basename="88421", filename="88421.jpg", file_path="a.jpg"
        )
        accidental = MainImage(
            batch_id=batch.id, basename="88435", filename="88435.jpg", file_path="b.jpg"
        )
        db.add_all([stable, accidental])
        db.flush()
        db.add_all([
            ImagePHashRegion(
                main_image_id=stable.id, region_type="BODY", x1=0, y1=0, x2=1, y2=1,
                phash_value=_hash_with_distance(4), feature_version="test",
            ),
            ImagePHashRegion(
                main_image_id=accidental.id, region_type="BODY", x1=0, y1=0, x2=1, y2=1,
                phash_value=_hash_with_distance(3), feature_version="test",
            ),
        ])
        db.flush()
        db.add_all([
            ProductReference(
                main_image_id=stable.id, category="观赛图", phash=_hash_with_distance(6),
                source_type="MANUAL_CONFIRMED", is_manual_confirmed=True, is_active=True,
            ),
            ProductReference(
                main_image_id=stable.id, category="聚会图", phash=_hash_with_distance(8),
                source_type="MANUAL_CONFIRMED", is_manual_confirmed=True, is_active=True,
            ),
            ProductReference(
                main_image_id=accidental.id, category="观赛图", phash=_hash_with_distance(22),
                source_type="MANUAL_CONFIRMED", is_manual_confirmed=True, is_active=True,
            ),
            ProductReference(
                main_image_id=accidental.id, category="聚会图", phash=_hash_with_distance(25),
                source_type="MANUAL_CONFIRMED", is_manual_confirmed=True, is_active=True,
            ),
        ])
        db.flush()

        scored = build_product_score_matrix(
            db, ["0000000000000000"], "观赛图", [stable, accidental]
        )

        assert scored.matrix[0, 0] > scored.matrix[0, 1]
        stable_evidence = scored.evidence[0][0]
        assert (
            stable_evidence.best_distance,
            stable_evidence.second_distance,
            stable_evidence.third_distance,
        ) == (4, 6, 8)
        assert stable_evidence.support_count == 3
        assert db.scalars(select(ProductReference).where(
            ProductReference.source_type == "MAIN"
        )).all()


def test_equal_reference_distances_do_not_compare_orm_objects(tmp_path):
    with Session(_database(tmp_path)) as db:
        batch = Batch(root_path=str(tmp_path))
        db.add(batch)
        db.flush()
        main = MainImage(
            batch_id=batch.id, basename="SKU", filename="SKU.jpg", file_path="sku.jpg"
        )
        db.add(main)
        db.flush()
        db.add(ImagePHashRegion(
            main_image_id=main.id, region_type="BODY", x1=0, y1=0, x2=1, y2=1,
            phash_value=_hash_with_distance(4), feature_version="test",
        ))
        db.add(ProductReference(
            main_image_id=main.id, category="观赛图", phash=_hash_with_distance(4),
            source_type="MANUAL_CONFIRMED", is_manual_confirmed=True, is_active=True,
        ))
        db.flush()

        scored = build_product_score_matrix(
            db, ["0000000000000000"], "观赛图", [main]
        )

        assert scored.evidence[0][0].best_distance == 4
        assert scored.evidence[0][0].second_distance == 4
