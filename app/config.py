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
        os.getenv("DESIGN_MATCHER_MIN_MARGIN", "0.01")
    )
    unmatched_score: float = float(
        os.getenv("DESIGN_MATCHER_UNMATCHED_SCORE", "0.50")
    )
    algorithm_version: str = os.getenv(
        "DESIGN_MATCHER_ALGORITHM_VERSION", "full_phash_hungarian_v1"
    )
    masked_algorithm_version: str = os.getenv(
        "DESIGN_MATCHER_MASKED_ALGORITHM_VERSION",
        "manual_mask_phash_hungarian_v2",
    )
    psd_algorithm_version: str = os.getenv(
        "DESIGN_MATCHER_PSD_ALGORITHM_VERSION", "psd_preview_phash_v1"
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
    body_crop_padding: float = float(
        os.getenv("DESIGN_MATCHER_BODY_CROP_PADDING", "0.05")
    )
    body_background_threshold: int = int(
        os.getenv("DESIGN_MATCHER_BODY_BACKGROUND_THRESHOLD", "245")
    )
    body_min_component_ratio: float = float(
        os.getenv("DESIGN_MATCHER_BODY_MIN_COMPONENT_RATIO", "0.002")
    )
    detail_category_keywords: tuple[str, ...] = tuple(
        value.strip().casefold()
        for value in os.getenv(
            "DESIGN_MATCHER_DETAIL_CATEGORY_KEYWORDS", "细节,detail"
        ).split(",")
        if value.strip()
    )
    body_phash_version: str = os.getenv(
        "DESIGN_MATCHER_BODY_PHASH_VERSION", "main_body_phash_v2_full_image"
    )
    detail_region_algorithm_version: str = os.getenv(
        "DESIGN_MATCHER_DETAIL_REGION_VERSION", "detail_region_phash_v1"
    )
    default_product_category: str = os.getenv(
        "DESIGN_MATCHER_PRODUCT_CATEGORY", "橄榄球衣"
    )
    upload_root: str = os.getenv(
        "DESIGN_MATCHER_UPLOAD_ROOT",
        r"O:\1-亚马逊数据库\设计库\文件包上传测试",
    )
    thumbnail_size: int = int(os.getenv("DESIGN_MATCHER_THUMBNAIL_SIZE", "512"))


settings = Settings()

MAIN_DIR_ALIASES = {"主图", "main", "main_images", "main-images"}
PSD_DIR_ALIASES = {"psd", "源文件", "源文件psd"}
IGNORED_IMAGE_CATEGORY_KEYWORDS = tuple(
    value.strip().casefold().replace(" ", "")
    for value in os.getenv("DESIGN_MATCHER_IGNORED_IMAGE_CATEGORIES", "白底").split(",")
    if value.strip()
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def is_ignored_image_category(category: str) -> bool:
    folded = category.casefold().replace(" ", "")
    return any(keyword in folded for keyword in IGNORED_IMAGE_CATEGORY_KEYWORDS)
