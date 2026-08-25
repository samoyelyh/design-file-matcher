from __future__ import annotations

import base64
import io

from PIL import Image
from sqlalchemy import select

from app.models import MatchedFile
from app.services.masks import save_profile_version


def mask_data_url(size=(32, 32), box=None) -> str:
    mask = Image.new("L", size, 0)
    if box is None:
        box = (0, 0, size[0], size[1])
    for x in range(box[0], box[2]):
        for y in range(box[1], box[3]):
            mask.putpixel((x, y), 255)
    stream = io.BytesIO()
    mask.save(stream, "PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def configure_manual_masks(db, batch_id: int) -> None:
    categories = list(db.scalars(select(MatchedFile.category).where(
        MatchedFile.batch_id == batch_id,
        MatchedFile.category != "PSD",
    ).distinct()))
    for category in categories:
        reference = db.scalar(select(MatchedFile).where(
            MatchedFile.batch_id == batch_id,
            MatchedFile.category == category,
        ).order_by(MatchedFile.id))
        save_profile_version(
            db,
            batch_id,
            category,
            mask_data_url(),
            reference.id,
        )
