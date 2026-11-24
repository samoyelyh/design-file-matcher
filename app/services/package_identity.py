from __future__ import annotations

import re
from pathlib import Path

# Product tokens found in existing design-pack names. These are literal labels
# for parsing, with no mapping to any design-library category or code.
PRODUCTS = (
    "露脐圆领毛衣", "圆领毛衣", "儿童足球套装", "足球套装", "卫衣套装",
    "橄榄球衣", "短袖T恤", "棒球衣", "刀锋鞋", "毛线帽", "帽衫", "卫衣", "T恤",
)
AUDIENCE = {"成人款", "儿童款", "男款", "女款"}
IDENTITY_FIELDS = ("product_name", "target_category", "designer")


def package_root(root: Path) -> Path:
    # Some batches were opened from inside the package's 效果图 directory.
    parent = re.sub(r"^（已完成）", "", root.parent.name)
    if root.name == "效果图" or (root.name.endswith("效果图") and root.name.startswith(parent + "-")):
        return root.parent
    return root


def parse_package(root: Path) -> tuple[str, dict]:
    original = package_root(root).name
    name = re.sub(r"^（已完成）", "", original)
    parts = [part.strip() for part in re.split(r"[-－–—]+", name) if part.strip()]
    parts = [part for part in parts if part not in {"效果图", "附图", "（成品）"}]
    explicit = set(parts) & set(PRODUCTS)
    product = next(iter(explicit)) if len(explicit) == 1 else None
    if not explicit:
        embedded = {token for part in parts for token in PRODUCTS if token in part}
        longest = {token for token in embedded if not any(token != other and token in other for other in embedded)}
        product = next(iter(longest)) if len(longest) == 1 else None
    audience = set(parts) & AUDIENCE
    if not audience and product:
        if any("儿童" + product in part for part in parts) or product.startswith("儿童"):
            audience.add("儿童款")
    category = next(iter(audience)) if len(audience) == 1 else None
    # The designer is the final token in the observed template. Require a
    # recognized product plus a name-shaped token with no design descriptors.
    candidate = parts[-1] if parts else ""
    name_pattern = r"[王刘何李陈黄张林吴杨赵周郑谢蔡许郭曾徐孙罗梁宋朱蒋唐高马邓韩洪吕彭方][\u4e00-\u9fff]{1,2}"
    designer = candidate if product and len(parts) >= 3 and re.fullmatch(name_pattern, candidate) else None
    fields = dict(zip(IDENTITY_FIELDS, (product, category, designer)))
    uncertain = [key for key, value in fields.items() if not value]
    return original, {
        "fields": fields, "confidence": "HIGH" if not uncertain else "PARTIAL" if len(uncertain) < 3 else "UNKNOWN",
        "uncertain_fields": uncertain,
    }
