from __future__ import annotations

import tempfile
from time import perf_counter
from pathlib import Path

from PIL import Image, ImageDraw
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.db import Base
from app.models import Batch, MainImage, MaskProfile, MatchCandidate, MatchedFile, ProductSimilarity
from app.services.matcher import match_batch


def patterned(path: Path, color: tuple[int, int, int], text: str) -> None:
    image = Image.new("RGB", (320, 320), "white")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((55, 35, 265, 290), radius=28, fill=color)
    draw.rectangle((75, 80, 245, 100), fill="white")
    draw.text((135, 130), text, fill="white", stroke_width=2, stroke_fill="black")
    image.save(path)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="design-matcher-lightglue-") as directory:
        root = Path(directory)
        paths = {name: root / f"{name}.png" for name in ("red-main", "blue-main", "red-query", "blue-query")}
        patterned(paths["red-main"], (190, 25, 35), "11")
        patterned(paths["blue-main"], (20, 60, 190), "88")
        patterned(paths["red-query"], (180, 30, 40), "11")
        patterned(paths["blue-query"], (25, 65, 180), "88")
        main_mask = root / "main-mask.png"
        query_mask = root / "query-mask.png"
        Image.new("L", (320, 320), 255).save(main_mask)
        Image.new("L", (320, 320), 255).save(query_mask)
        sql = create_engine(f"sqlite:///{root / 'smoke.db'}")
        Base.metadata.create_all(sql)
        with Session(sql, expire_on_commit=False) as db:
            batch = Batch(root_path=str(root), status="SCANNED", match_engine="LIGHTGLUE_COLOR", matching_model="LIGHTGLUE_COLOR")
            db.add(batch)
            db.flush()
            mains = [
                MainImage(batch_id=batch.id, basename="red", filename="red-main.png", file_path=str(paths["red-main"])),
                MainImage(batch_id=batch.id, basename="blue", filename="blue-main.png", file_path=str(paths["blue-main"])),
            ]
            db.add_all(mains)
            db.flush()
            files = [
                MatchedFile(batch_id=batch.id, category="测试图", original_filename="red-query.png", original_path=str(paths["red-query"]), current_filename="red-query.png", current_path=str(paths["red-query"]), extension=".png"),
                MatchedFile(batch_id=batch.id, category="测试图", original_filename="blue-query.png", original_path=str(paths["blue-query"]), current_filename="blue-query.png", current_path=str(paths["blue-query"]), extension=".png"),
            ]
            db.add_all(files)
            profiles = [
                MaskProfile(batch_id=batch.id, product_category="", image_category="主图", name="主图", version=1, reference_width=320, reference_height=320, mask_path=str(main_mask), crop_left=0, crop_top=0, crop_right=1, crop_bottom=1, creation_method="MANUAL"),
                MaskProfile(batch_id=batch.id, product_category="", image_category="测试图", name="测试图", version=1, reference_width=320, reference_height=320, mask_path=str(query_mask), crop_left=0, crop_top=0, crop_right=1, crop_bottom=1, creation_method="MANUAL"),
            ]
            db.add_all(profiles)
            db.commit()
            started = perf_counter()
            result = match_batch(db, batch.id)
            first_seconds = perf_counter() - started
            db.refresh(files[0])
            db.refresh(files[1])
            score_rows = list(db.scalars(select(ProductSimilarity).order_by(
                ProductSimilarity.matched_file_id, ProductSimilarity.main_image_id
            )))
            matrix = [[score_rows[0].score, score_rows[1].score], [score_rows[2].score, score_rows[3].score]]
            assert len(score_rows) == 4
            assert db.query(MatchCandidate).count() == 4
            assert files[0].matched_main_id == mains[0].id
            assert files[1].matched_main_id == mains[1].id
            assert files[0].match_details_json and files[1].match_details_json
            assert matrix[0][0] > matrix[0][1]
            assert matrix[1][1] > matrix[1][0]
            started = perf_counter()
            match_batch(db, batch.id)
            cached_seconds = perf_counter() - started
            cached_scores = list(db.scalars(select(ProductSimilarity.score).order_by(
                ProductSimilarity.matched_file_id, ProductSimilarity.main_image_id
            )))
            assert cached_scores == [value for row in matrix for value in row]
            print({"first_seconds": round(first_seconds, 3), "cached_seconds": round(cached_seconds, 3),
                   "full_matrix_unchanged": True})
            print({"status": result["status"], "engine": batch.match_engine, "matrix": matrix})
        sql.dispose()


if __name__ == "__main__":
    main()
