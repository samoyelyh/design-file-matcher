"""Run a read-only DINO Top-5 quality check without touching source files."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from app.config import IMAGE_EXTENSIONS
from app.services.features import DinoFeatureProvider
from app.services.matcher import _encode_in_batches
from app.services.scanner import discover_layout


def image_files(directory: Path) -> list[Path]:
    return sorted(
        [p for p in directory.iterdir() if p.is_file() and p.suffix.casefold() in IMAGE_EXTENSIONS],
        key=lambda p: p.name.casefold(),
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="只读验证 DINO Top-5 匹配效果")
    parser.add_argument("root", type=Path, help="设计根目录")
    parser.add_argument("--category", help="只验证某个一级分类目录")
    parser.add_argument("--limit", type=int, default=0, help="每类最多验证多少个文件，0 表示不限")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    try:
        main_dir, _effect_root, category_dirs = discover_layout(root)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    mains = image_files(main_dir)
    if not mains:
        raise SystemExit("没有主图")

    categories = category_dirs
    if args.category:
        categories = [p for p in categories if p.name == args.category]
    queries: list[Path] = []
    for directory in categories:
        files = sorted(
            [p for p in directory.iterdir() if p.is_file() and (p.suffix.casefold() in IMAGE_EXTENSIONS or p.suffix.casefold() == ".psd")],
            key=lambda p: p.name.casefold(),
        )
        queries.extend(files[:args.limit or None])
    if not queries:
        raise SystemExit("没有待验证文件")

    model = DinoFeatureProvider()
    main_vectors = _encode_in_batches(model, mains)
    main_vectors /= np.clip(np.linalg.norm(main_vectors, axis=1, keepdims=True), 1e-12, None)
    print(
        f"模型 {model.model_name}，设备 {model.device}；"
        f"主图 {len(mains)} 张，待验证 {len(queries)} 个（只读，不改名）"
    )
    print("原文件\tTop1\tscore\tTop2\tscore\tTop3\tscore\tTop4\tscore\tTop5\tscore\tTop1-Top2")
    for query in queries:
        vector = _encode_in_batches(model, [query])[0]
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        scores = main_vectors @ vector
        order = np.argsort(-scores)[:min(5, len(mains))]
        columns = [str(query.relative_to(root))]
        for index in order:
            columns.extend([mains[int(index)].stem, f"{float(scores[int(index)]):.4f}"])
        gap = float(scores[int(order[0])] - scores[int(order[1])]) if len(order) > 1 else 0.0
        columns.append(f"{gap:.4f}")
        print("\t".join(columns))


if __name__ == "__main__":
    main()
