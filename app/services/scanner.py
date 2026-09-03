from __future__ import annotations

from collections import Counter
from pathlib import Path

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.config import (
    IMAGE_EXTENSIONS,
    MAIN_DIR_ALIASES,
    PSD_DIR_ALIASES,
    is_ignored_image_category,
)
from app.models import Batch, BatchProgress, MainImage, MatchedFile

from .media import thumbnail_cache_path


def _canonical_category(directory_name: str) -> str:
    folded = directory_name.casefold().replace(" ", "")
    if folded in PSD_DIR_ALIASES:
        return "PSD"
    return directory_name


def _normalized_directory_name(name: str) -> str:
    return name.casefold().replace(" ", "")


def _is_ignored_directory(path: Path) -> bool:
    return any(part == ".design_matcher" for part in path.parts)


def _find_main_directory(root: Path, directories: list[Path] | None = None) -> Path:
    """Find one main-image directory at any depth under the selected project root."""
    directories = directories if directories is not None else [
        item for item in root.rglob("*")
        if item.is_dir() and not _is_ignored_directory(item.relative_to(root))
    ]
    matches = [
        item for item in directories
        if _normalized_directory_name(item.name) in MAIN_DIR_ALIASES
    ]
    if len(matches) != 1:
        raise ValueError(
            "所选项目目录下必须且只能识别到一个“主图”目录，"
            f"当前递归识别到 {len(matches)} 个"
        )
    return matches[0]


def discover_layout(root: Path) -> tuple[Path, list[Path], list[Path]]:
    """Discover the real project layout, including PSD beside the effect-image folder."""
    directories = [
        item for item in root.rglob("*")
        if item.is_dir() and not _is_ignored_directory(item.relative_to(root))
    ]
    main_dir = _find_main_directory(root, directories)
    effect_root = main_dir.parent

    visual_category_dirs = [
        item for item in effect_root.iterdir()
        if (
            item.is_dir()
            and item != main_dir
            and item.name != ".design_matcher"
            and not is_ignored_image_category(item.name)
        )
    ]
    psd_dirs = [
        item for item in directories
        if _normalized_directory_name(item.name) in PSD_DIR_ALIASES
    ]

    category_dirs: list[Path] = []
    seen: set[Path] = set()
    for directory in [*visual_category_dirs, *psd_dirs]:
        resolved = directory.resolve()
        if resolved not in seen:
            category_dirs.append(directory)
            seen.add(resolved)
    return main_dir, effect_root, category_dirs


def inspect_layout(root: Path) -> dict | None:
    try:
        main_dir, effect_root, category_dirs = discover_layout(root)
    except (OSError, ValueError):
        return None
    psd_dirs = [d for d in category_dirs if _canonical_category(d.name) == "PSD"]
    return {
        "effect_root": str(effect_root.relative_to(root)) if effect_root != root else ".",
        "main_directory": str(main_dir.relative_to(root)),
        "psd_directories": [str(path.relative_to(root)) for path in psd_dirs],
        "category_directories": [str(path.relative_to(root)) for path in category_dirs],
    }


def scan_batch(db: Session, batch_id: int) -> dict:
    batch = db.get(Batch, batch_id)
    if not batch:
        raise ValueError("批次不存在")
    if batch.status in {"QUEUED", "MATCHING"}:
        raise ValueError("匹配进行中，不能重新扫描")
    root = Path(batch.root_path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"设计根目录不存在: {root}")

    main_dir, effect_root, category_dirs = discover_layout(root)

    db.execute(delete(MatchedFile).where(MatchedFile.batch_id == batch.id))
    db.execute(delete(MainImage).where(MainImage.batch_id == batch.id))
    db.flush()

    warnings: list[str] = []
    basename_seen: set[str] = set()
    main_count = 0
    for path in sorted(main_dir.iterdir(), key=lambda p: p.name.casefold()):
        if not path.is_file() or path.suffix.casefold() not in IMAGE_EXTENSIONS:
            continue
        key = path.stem.casefold()
        if key in basename_seen:
            raise ValueError(f"主图 basename 重复（忽略大小写）: {path.stem}")
        basename_seen.add(key)
        db.add(MainImage(
            batch_id=batch.id,
            basename=path.stem,
            filename=path.name,
            file_path=str(path),
            thumbnail_path=str(thumbnail_cache_path(root, batch.id, path)),
        ))
        main_count += 1

    if not main_count:
        raise ValueError("主图目录中没有可读取的图片")

    db.flush()
    counts: Counter[str] = Counter()
    for directory in sorted(category_dirs, key=lambda p: p.name.casefold()):
        category = _canonical_category(directory.name)
        counts.setdefault(category, 0)
        for path in sorted(directory.iterdir(), key=lambda p: p.name.casefold()):
            if not path.is_file():
                continue
            extension = path.suffix.casefold()
            allowed = extension == ".psd" if category == "PSD" else extension in IMAGE_EXTENSIONS
            if not allowed:
                continue
            db.add(MatchedFile(
                batch_id=batch.id,
                category=category,
                original_filename=path.name,
                original_path=str(path),
                current_filename=path.name,
                current_path=str(path),
                extension=path.suffix,
                thumbnail_path=str(thumbnail_cache_path(root, batch.id, path)),
            ))
            counts[category] += 1

    batch.status = "SCANNED"
    batch.error_message = "\n".join(warnings) if warnings else None
    progress = db.get(BatchProgress, batch.id)
    if not progress:
        progress = BatchProgress(batch_id=batch.id)
        db.add(progress)
    progress.stage = "READY"
    progress.current = 0
    progress.total = sum(
        count for category, count in counts.items() if category != "PSD"
    )
    progress.percent = 0
    progress.message = "扫描完成，等待开始匹配"
    progress.device = None
    db.commit()
    return {
        "batch_id": batch.id,
        "root_path": str(root),
        "effect_root": str(effect_root.relative_to(root)) if effect_root != root else ".",
        "main_directory": str(main_dir.relative_to(root)),
        "psd_directories": [
            str(path.relative_to(root))
            for path in category_dirs
            if _canonical_category(path.name) == "PSD"
        ],
        "main_count": main_count,
        "categories": dict(counts),
        "warnings": warnings,
    }
