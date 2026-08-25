from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageOps

from app.config import settings


def open_visual(path: Path) -> Image.Image:
    """Open a normal image or the flattened composite of a PSD."""
    if path.suffix.casefold() == ".psd":
        try:
            from psd_tools import PSDImage
        except ImportError as exc:
            raise RuntimeError("读取 PSD 需要安装 psd-tools") from exc
        image = PSDImage.open(path).composite()
        if image is None:
            raise ValueError(f"PSD 没有可用的合成预览: {path.name}")
        return image.convert("RGB")

    with Image.open(path) as source:
        source.load()
        return ImageOps.exif_transpose(source).convert("RGB")


def thumbnail_cache_path(root: Path, batch_id: int, source: Path) -> Path:
    digest = hashlib.sha1(str(source.resolve()).encode("utf-8")).hexdigest()
    return root / ".design_matcher" / str(batch_id) / "thumbnails" / f"{digest}.jpg"


def embedding_cache_path(root: Path, batch_id: int, source: Path) -> Path:
    digest = hashlib.sha1(str(source.resolve()).encode("utf-8")).hexdigest()
    return root / ".design_matcher" / str(batch_id) / "embeddings" / f"{digest}.npy"


def preview_cache_path(root: Path, batch_id: int, source: Path) -> Path:
    digest = hashlib.sha1(str(source.resolve()).encode("utf-8")).hexdigest()
    return root / ".design_matcher" / str(batch_id) / "previews" / f"{digest}.jpg"


def ensure_thumbnail(source: Path, target: Path, size: int | None = None) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        return target
    image = open_visual(source)
    image.thumbnail((size or settings.thumbnail_size,) * 2, Image.Resampling.LANCZOS)
    image.save(target, "JPEG", quality=88, optimize=True)
    return target


def ensure_preview(source: Path, target: Path, size: int = 2048) -> Path:
    return ensure_thumbnail(source, target, size=size)
