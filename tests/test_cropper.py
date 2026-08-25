from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.db import Base
from app.models import Batch, DerivedImageAsset, MainImage, MatchedFile
from app.services.cropper import MaskedCropProcessor
from app.services.masks import save_profile_version
from app.services.scanner import scan_batch
from tests.support import mask_data_url


def make_image(path: Path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (120, 80), color).save(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_mask_generates_linked_crop_without_modifying_original(tmp_path):
    root = tmp_path / "project"
    main_path = root / "主图" / "SKU.jpg"
    file_path = root / "观赛图" / "scene.jpg"
    make_image(main_path, (240, 20, 20))
    make_image(file_path, (220, 30, 30))
    before = sha256(file_path)
    engine = create_engine(f"sqlite:///{tmp_path / 'crop.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        batch = Batch(root_path=str(root), product_category="橄榄球衣")
        db.add(batch)
        db.commit()
        scan_batch(db, batch.id)
        item = db.scalar(select(MatchedFile))
        profile = save_profile_version(
            db, batch.id, "观赛图", mask_data_url((120, 80), (30, 20, 90, 60)), item.id
        )
        main = db.scalar(select(MainImage))
        processor = MaskedCropProcessor(padding_ratio=.05)
        file_assets = processor.process_file(db, batch, item, profile)
        main_assets = processor.process_main(db, batch, main, profile)
        db.commit()

        assert Path(file_assets.mask.asset_path).is_file()
        assert Path(file_assets.crop.asset_path).is_file()
        assert Path(main_assets.crop.asset_path).is_file()
        with Image.open(file_assets.crop.asset_path) as crop:
            assert crop.width < 120 and crop.height < 80
        assert sha256(file_path) == before
        assets = db.scalars(select(DerivedImageAsset)).all()
        assert {asset.asset_type for asset in assets} == {"MASK", "MASKED_CROP"}
        assert any(asset.parent_file_id == item.id for asset in assets)
        assert any(asset.parent_main_image_id == main.id for asset in assets)
