"""Run a read-only OpenCLIP Top-3 quality check before enabling auto rename."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from app.config import IMAGE_EXTENSIONS, MAIN_DIR_ALIASES
from app.services.matcher import OpenClipEmbedder


def image_files(directory: Path) -> list[Path]:
    return sorted(
        [p for p in directory.iterdir() if p.is_file() and p.suffix.casefold() in IMAGE_EXTENSIONS],
        key=lambda p: p.name.casefold(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="只读验证 OpenCLIP Top-3 匹配效果")
    parser.add_argument("root", type=Path, help="设计根目录")
    parser.add_argument("--category", help="只验证某个一级分类目录")
    parser.add_argument("--limit", type=int, default=0, help="每类最多验证多少个文件，0 表示不限")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    main_dirs = [p for p in root.iterdir() if p.is_dir() and p.name.casefold() in MAIN_DIR_ALIASES]
    if len(main_dirs) != 1:
        raise SystemExit("根目录必须且只能包含一个主图目录")
    mains = image_files(main_dirs[0])
    if not mains:
        raise SystemExit("没有主图")

    categories = [p for p in root.iterdir() if p.is_dir() and p != main_dirs[0] and p.name != ".design_matcher"]
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

    model = OpenClipEmbedder()
    main_vectors = model.encode(mains)
    main_vectors /= np.clip(np.linalg.norm(main_vectors, axis=1, keepdims=True), 1e-12, None)
    print(f"主图 {len(mains)} 张，待验证 {len(queries)} 个（只读，不改名）")
    print("原文件\tTop1\tscore\tTop2\tscore\tTop3\tscore")
    for query in queries:
        vector = model.encode([query])[0]
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        scores = main_vectors @ vector
        order = np.argsort(-scores)[:min(3, len(mains))]
        columns = [str(query.relative_to(root))]
        for index in order:
            columns.extend([mains[int(index)].stem, f"{float(scores[int(index)]):.4f}"])
        print("\t".join(columns))


if __name__ == "__main__":
    main()
