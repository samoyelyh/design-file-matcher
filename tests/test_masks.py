from __future__ import annotations

import base64
import io

from PIL import Image
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.db import Base
from app.models import Batch, MaskProfile, MatchedFile
from app.services.masks import create_default_profile, save_profile_version
from app.services.scanner import scan_batch


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
        profile1 = create_default_profile(db, batch.id, "观赛图")
        reference = db.scalar(select(MatchedFile))
        assert profile1.version == 1
        assert profile1.enabled is True
        assert profile1.product_category == "橄榄球衣"

        mask = Image.new("L", (80, 60), 0)
        for x in range(20, 60):
            for y in range(10, 50):
                mask.putpixel((x, y), 255)
        stream = io.BytesIO()
        mask.save(stream, "PNG")
        data_url = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()
        profile2 = save_profile_version(
            db, batch.id, "观赛图", data_url, reference.id
        )
        db.refresh(profile1)
        assert profile1.enabled is False
        assert profile2.version == 2 and profile2.enabled is True
        assert len(db.scalars(select(MaskProfile)).all()) == 2


def test_psd_never_creates_mask_profile(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'mask.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        batch = Batch(root_path=str(tmp_path), product_category="橄榄球衣")
        db.add(batch)
        db.commit()
        try:
            create_default_profile(db, batch.id, "PSD")
        except ValueError as exc:
            assert "PSD" in str(exc)
        else:
            raise AssertionError("PSD must not call the mask processor")
