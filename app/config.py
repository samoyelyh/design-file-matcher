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
    max_references_per_category: int = int(
        os.getenv("DESIGN_MATCHER_MAX_REFERENCES_PER_CATEGORY", "5")
    )
    use_product_reference_scoring: bool = os.getenv(
        "DESIGN_MATCHER_USE_PRODUCT_REFERENCES", "1"
    ).strip().lower() in {"1", "true", "yes", "on"}
    reference_weight_1: float = float(
        os.getenv("DESIGN_MATCHER_REFERENCE_WEIGHT_1", "0.60")
    )
    reference_weight_2: float = float(
        os.getenv("DESIGN_MATCHER_REFERENCE_WEIGHT_2", "0.25")
    )
    reference_weight_3: float = float(
        os.getenv("DESIGN_MATCHER_REFERENCE_WEIGHT_3", "0.15")
    )
    reference_support_max_distance: int = int(
        os.getenv("DESIGN_MATCHER_REFERENCE_SUPPORT_MAX_DISTANCE", "12")
    )
    masked_algorithm_version: str = os.getenv(
        "DESIGN_MATCHER_MASKED_ALGORITHM_VERSION",
        "manual_mask_phash_hungarian_v2",
    )
    product_reference_algorithm_version: str = os.getenv(
        "DESIGN_MATCHER_PRODUCT_REFERENCE_ALGORITHM_VERSION",
        "manual_mask_product_reference_hungarian_v1",
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
    default_matching_model: str = os.getenv(
        "DESIGN_MATCHER_DEFAULT_MODEL", "VISIONMATCH"
    ).strip().upper()
    dinov2_model_id: str = os.getenv(
        "DESIGN_MATCHER_DINOV2_MODEL_ID", "facebook/dinov2-base"
    )
    dinov2_model_dir: str = os.getenv(
        "DESIGN_MATCHER_DINOV2_MODEL_DIR",
        r"G:\models\DINOv2-ViT-B-14",
    )
    dinov2_algorithm_version: str = os.getenv(
        "DESIGN_MATCHER_DINOV2_ALGORITHM_VERSION",
        "manual_mask_dinov2_vitb14_hungarian_v1",
    )
    dinov3_model_id: str = os.getenv(
        "DESIGN_MATCHER_DINOV3_MODEL_ID",
        "facebook/dinov3-vitb16-pretrain-lvd1689m",
    )
    dinov3_model_dir: str = os.getenv(
        "DESIGN_MATCHER_DINOV3_MODEL_DIR",
        r"G:\models\DINOv3-ViT-B-16",
    )
    dinov3_cache_dir: str = os.getenv(
        "DESIGN_MATCHER_DINOV3_CACHE_DIR", r"G:\hf-cache"
    )
    dinov3_batch_size: int = int(
        os.getenv("DESIGN_MATCHER_DINOV3_BATCH_SIZE", "16")
    )
    dinov3_algorithm_version: str = os.getenv(
        "DESIGN_MATCHER_DINOV3_ALGORITHM_VERSION",
        "manual_mask_dinov3_vitb16_hungarian_v1",
    )
    dinov3_min_score: float = float(
        os.getenv("DESIGN_MATCHER_DINOV3_MIN_SCORE", "0.65")
    )
    dinov3_min_margin: float = float(
        os.getenv("DESIGN_MATCHER_DINOV3_MIN_MARGIN", "0.01")
    )
    color_feature_version: str = os.getenv(
        "DESIGN_MATCHER_COLOR_FEATURE_VERSION", "manual_mask_hsv_lab_v1"
    )
    color_hsv_weight: float = float(
        os.getenv("DESIGN_MATCHER_COLOR_HSV_WEIGHT", "0.55")
    )
    color_lab_hist_weight: float = float(
        os.getenv("DESIGN_MATCHER_COLOR_LAB_HIST_WEIGHT", "0.35")
    )
    color_lab_mean_weight: float = float(
        os.getenv("DESIGN_MATCHER_COLOR_LAB_MEAN_WEIGHT", "0.10")
    )
    lightglue_prefilter_top_n: int = int(
        os.getenv("DESIGN_MATCHER_LIGHTGLUE_PREFILTER_TOP_N", "50")
    )
    lightglue_model_dir: str = os.getenv(
        "DESIGN_MATCHER_LIGHTGLUE_MODEL_DIR", r"G:\models\LightGlue"
    )
    lightglue_extractor: str = os.getenv(
        "DESIGN_MATCHER_LIGHTGLUE_EXTRACTOR", "ALIKED"
    ).strip().upper()
    lightglue_max_keypoints: int = int(
        os.getenv("DESIGN_MATCHER_LIGHTGLUE_MAX_KEYPOINTS", "2048")
    )
    lightglue_resize: int = int(
        os.getenv("DESIGN_MATCHER_LIGHTGLUE_RESIZE", "1024")
    )
    lightglue_ransac_threshold: float = float(
        os.getenv("DESIGN_MATCHER_LIGHTGLUE_RANSAC_THRESHOLD", "4.0")
    )
    local_inlier_ratio_weight: float = float(
        os.getenv("DESIGN_MATCHER_LOCAL_INLIER_RATIO_WEIGHT", "0.35")
    )
    local_confidence_weight: float = float(
        os.getenv("DESIGN_MATCHER_LOCAL_CONFIDENCE_WEIGHT", "0.20")
    )
    local_coverage_weight: float = float(
        os.getenv("DESIGN_MATCHER_LOCAL_COVERAGE_WEIGHT", "0.25")
    )
    local_count_weight: float = float(
        os.getenv("DESIGN_MATCHER_LOCAL_COUNT_WEIGHT", "0.20")
    )
    lightglue_color_weight: float = float(
        os.getenv("DESIGN_MATCHER_LIGHTGLUE_COLOR_WEIGHT", "0.30")
    )
    lightglue_local_weight: float = float(
        os.getenv("DESIGN_MATCHER_LIGHTGLUE_LOCAL_WEIGHT", "0.70")
    )
    qwen_fusion_qwen_weight: float = float(
        os.getenv("DESIGN_MATCHER_QWEN_FUSION_QWEN_WEIGHT", "0.70")
    )
    qwen_fusion_lightglue_color_weight: float = float(
        os.getenv("DESIGN_MATCHER_QWEN_FUSION_LIGHTGLUE_COLOR_WEIGHT", "0.30")
    )
    lightglue_algorithm_version: str = os.getenv(
        "DESIGN_MATCHER_LIGHTGLUE_ALGORITHM_VERSION",
        "manual_mask_aliked_lightglue_color_v1",
    )
    lightglue_min_score: float = float(
        os.getenv("DESIGN_MATCHER_LIGHTGLUE_MIN_SCORE", "0.42")
    )
    lightglue_min_margin: float = float(
        os.getenv("DESIGN_MATCHER_LIGHTGLUE_MIN_MARGIN", "0.01")
    )


settings = Settings()

MAIN_DIR_ALIASES = {"主图", "main", "main_images", "main-images"}
PSD_DIR_ALIASES = {"psd", "源文件", "源文件psd"}
IGNORED_IMAGE_CATEGORY_KEYWORDS = tuple(
    value.strip().casefold().replace(" ", "")
    for value in os.getenv("DESIGN_MATCHER_IGNORED_IMAGE_CATEGORIES", "").split(",")
    if value.strip()
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def is_ignored_image_category(category: str) -> bool:
    folded = category.casefold().replace(" ", "")
    return any(keyword in folded for keyword in IGNORED_IMAGE_CATEGORY_KEYWORDS)


def is_excluded_file_category(category: str) -> bool:
    """Folders kept untouched and excluded from image processing/UI."""
    return is_ignored_image_category(category)
