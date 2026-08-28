from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv(
        "DESIGN_MATCHER_DB_URL",
        f"sqlite:///{(BASE_DIR / 'data' / 'matcher.db').as_posix()}",
    )
    low_confidence_threshold: float = float(
        os.getenv("DESIGN_MATCHER_LOW_CONFIDENCE", "0.80")
    )
    match_min_score: float = float(
        os.getenv("DESIGN_MATCHER_MIN_SCORE", "0.80")
    )
    match_min_margin: float = float(
        os.getenv("DESIGN_MATCHER_MIN_MARGIN", "0.0005")
    )
    unmatched_score: float = float(
        os.getenv("DESIGN_MATCHER_UNMATCHED_SCORE", "0.50")
    )
    algorithm_version: str = os.getenv(
        "DESIGN_MATCHER_ALGORITHM_VERSION", "dino_cosine_hungarian_v1"
    )
    masked_algorithm_version: str = os.getenv(
        "DESIGN_MATCHER_MASKED_ALGORITHM_VERSION",
        "dino_masked_cosine_hungarian_v1",
    )
    psd_algorithm_version: str = os.getenv(
        "DESIGN_MATCHER_PSD_ALGORITHM_VERSION", "dino_psd_preview_v1"
    )
    mask_crop_padding_ratio: float = float(
        os.getenv("DESIGN_MATCHER_MASK_PADDING", "0.08")
    )
    phash_max_distance: int = int(
        os.getenv("DESIGN_MATCHER_PHASH_MAX_DISTANCE", "8")
    )
    phash_min_margin: int = int(
        os.getenv("DESIGN_MATCHER_PHASH_MIN_MARGIN", "4")
    )
    default_product_category: str = os.getenv(
        "DESIGN_MATCHER_PRODUCT_CATEGORY", "橄榄球衣"
    )
    upload_root: str = os.getenv(
        "DESIGN_MATCHER_UPLOAD_ROOT",
        r"O:\1-亚马逊数据库\设计库\文件包上传测试",
    )
    thumbnail_size: int = int(os.getenv("DESIGN_MATCHER_THUMBNAIL_SIZE", "512"))
    clip_model: str = os.getenv("DESIGN_MATCHER_CLIP_MODEL", "ViT-B-32")
    # The official OpenAI checkpoint avoids Hugging Face authentication/rate limits
    # on the first local run while keeping the same OpenCLIP inference API.
    clip_pretrained: str = os.getenv("DESIGN_MATCHER_CLIP_PRETRAINED", "openai")
    embedding_backend: str = os.getenv(
        "EMBEDDING_BACKEND",
        os.getenv("DESIGN_MATCHER_EMBEDDING_BACKEND", "dinov2"),
    ).casefold()
    dinov3_model: str = os.getenv(
        "DESIGN_MATCHER_DINOV3_MODEL",
        "facebook/dinov3-vitb16-pretrain-lvd1689m",
    )
    dinov2_model: str = os.getenv(
        "DESIGN_MATCHER_DINOV2_MODEL", "facebook/dinov2-base"
    )
    dinov2_local_dir: str = os.getenv(
        "DESIGN_MATCHER_DINOV2_LOCAL_DIR",
        str(BASE_DIR / "models" / "dinov2-base"),
    )
    dino_allow_fallback: bool = os.getenv(
        "DESIGN_MATCHER_DINO_ALLOW_FALLBACK", "1"
    ).casefold() not in {"0", "false", "no"}
    hf_endpoint: str = os.getenv(
        "HF_ENDPOINT", "https://hf-mirror.com"
    ).rstrip("/")
    match_batch_size: int = int(os.getenv("DESIGN_MATCHER_BATCH_SIZE", "8"))


settings = Settings()

MAIN_DIR_ALIASES = {"主图", "main", "main_images", "main-images"}
PSD_DIR_ALIASES = {"psd", "源文件", "源文件psd"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
