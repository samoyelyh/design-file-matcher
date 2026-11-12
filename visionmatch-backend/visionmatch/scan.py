from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .provider import executable, file_hash, safe_environment
from .schemas import PAIR_PROMPT, REVIEW_PROMPT, PAIR_SCHEMA, REVIEW_SCHEMA, schema_hash
from .settings import Settings

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def engine_hash() -> str:
    package = Path(__file__).resolve().parent
    parts = {name: hashlib.sha256((package / name).read_bytes()).hexdigest()
             for name in ("provider.py", "schemas.py", "matching.py", "runner.py", "scan.py")}
    return canonical_hash(parts)


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def cli_version() -> str:
    result = subprocess.run([executable(), "--version"], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=15, env=safe_environment())
    if result.returncode:
        raise RuntimeError("无法读取 Codex 版本。")
    return result.stdout.strip()


def scan_folder(root: Path, recursive: bool, limit: int | None = None) -> list[dict]:
    files = sorted((p for p in (root.rglob("*") if recursive else root.iterdir())
                    if p.is_file() and p.suffix.lower() in EXTENSIONS), key=lambda p: str(p).casefold())
    if limit is not None:
        if limit <= 0:
            raise ValueError("图片上限必须大于 0。")
        files = files[:limit]
    images = []
    for path in files:
        try:
            with Image.open(path) as picture:
                width, height, image_format = *picture.size, picture.format
                picture.verify()
            digest = file_hash(path)
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            raise ValueError(f"图片无法读取，请先处理该文件: {path} ({exc})") from exc
        images.append({"path": str(path.resolve()), "name": path.name,
                       "relative_path": str(path.relative_to(root)), "sha256": digest,
                       "width": width, "height": height, "format": image_format})
    return images


def make_plan(settings: Settings, limit_a: int | None = None, limit_b: int | None = None,
              *, version: str | None = None, fresh: bool = False) -> dict:
    settings.validate()
    a = scan_folder(Path(settings.folder_a), settings.recursive, limit_a)
    b = scan_folder(Path(settings.folder_b), settings.recursive, limit_b)
    if not a or not b:
        raise ValueError("A 和 B 均需要至少一张可读图片。")
    version = version or cli_version()
    image_pairs = len(a) * len(b)
    reviews_max = len(a) if len(b) >= 2 else 0
    config = settings.snapshot()
    prompt_hash = canonical_hash({"pair": PAIR_PROMPT, "review": REVIEW_PROMPT})
    core = {"settings": config, "images_a": a, "images_b": b, "cli_version": version,
            "pair_schema_hash": schema_hash(PAIR_SCHEMA), "review_schema_hash": schema_hash(REVIEW_SCHEMA),
            "prompt_hash": prompt_hash, "engine_hash": engine_hash(), "fresh": fresh, "limit_a": limit_a, "limit_b": limit_b}
    return {**core, "plan_id": canonical_hash(core), "pair_count": image_pairs,
            "review_max": reviews_max, "base_calls_max": image_pairs + reviews_max,
            "max_cli_attempts_with_retries": (image_pairs + reviews_max) * settings.max_attempts_per_item,
            "notes": ["调用数为 Codex CLI 启动次数；CLI 内部网络重连可能无法逐次观测。",
                      "复核按条件触发，实际次数可能小于上限。", "不同运行使用独立输出目录。"]}


def validate_sources(plan: dict) -> None:
    for image in plan["images_a"] + plan["images_b"]:
        path = Path(image["path"])
        if not path.is_file() or file_hash(path) != image["sha256"]:
            raise ValueError(f"输入图片缺失或发生变化，请重新生成计划: {path}")


def cache_key(plan: dict, kind: str, images: list[dict]) -> str:
    return canonical_hash({"kind": kind, "images": [x["sha256"] for x in images],
                           "model": plan["settings"]["model"], "effort": plan["settings"]["reasoning_effort"],
                           "rule_version": plan["settings"]["rule_version"], "prompt_version": plan["settings"]["prompt_version"],
                           "prompt_hash": plan["prompt_hash"], "cli_version": plan["cli_version"],
                           "engine_hash": plan["engine_hash"],
                           "schema": plan["pair_schema_hash"] if kind == "PAIR" else plan["review_schema_hash"]})
