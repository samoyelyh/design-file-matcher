from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError


MODEL_ID = os.getenv(
    "DESIGN_MATCHER_DINOV3_MODEL_ID",
    "facebook/dinov3-vitb16-pretrain-lvd1689m",
)
MODEL_DIR = Path(os.getenv(
    "DESIGN_MATCHER_DINOV3_MODEL_DIR",
    r"G:\models\DINOv3-ViT-B-16",
))
CACHE_DIR = Path(os.getenv(
    "DESIGN_MATCHER_DINOV3_CACHE_DIR",
    r"G:\hf-cache",
))


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"正在下载 {MODEL_ID}")
    print(f"模型目录: {MODEL_DIR}")
    print(f"缓存目录: {CACHE_DIR}")
    try:
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=MODEL_DIR,
            cache_dir=CACHE_DIR,
        )
    except GatedRepoError as exc:
        raise SystemExit(
            "DINOv3 权重受官方许可保护，当前账号未授权。\n"
            "1. 打开 https://huggingface.co/facebook/"
            "dinov3-vitb16-pretrain-lvd1689m 并接受许可\n"
            "2. 在本机运行: .venv\\Scripts\\hf.exe auth login\n"
            "3. 再次运行本脚本"
        ) from exc
    print("下载完成。请重启匹配服务。")


if __name__ == "__main__":
    main()
