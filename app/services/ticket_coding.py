"""Catalog and claim calls using the design library's ticket service contract."""

from __future__ import annotations

import json
import os
import re
from urllib.parse import urlsplit

import httpx
from sqlalchemy import select

from app.models import DesignCode, DesignPackageInfo, MainImageContent
from app.services import design_coding
from app.services.design_content import content_signature, dump


def api_url():
    return os.getenv("DESIGN_MATCHER_TICKET_API_URL", "").strip().rstrip("/")


def operator_id():
    value = os.getenv("DESIGN_MATCHER_TICKET_USER_ID", "").strip()
    return int(value) if value.isascii() and value.isdecimal() and len(value) <= 10 and int(value) > 0 else None


def request(method, base, path, **kwargs):
    with httpx.Client(timeout=httpx.Timeout(20, connect=5), trust_env=False) as client:
        response = client.request(method, f"{base}{path}", **kwargs)
        response.raise_for_status()
        return response.json() if response.content else None


def catalog_rows(base, path, **params):
    value = request("GET", base, path, params=params)
    if not isinstance(value, list) or not all(isinstance(entry, dict) for entry in value):
        raise ValueError("领号系统的映射返回格式不正确")
    return value


def name_key(value):
    return design_coding.key(value) if isinstance(value, str) else ""


def new_theme_name(data, fan_theme):
    if data.get("uncertain_items"):
        return None
    fields = data.get("team_names" if fan_theme else "main_subject", [])
    excluded = {"个性化姓名文字", "个性化名字文字", "个性化文字", "定制文字",
                *(name_key(value) for value in design_coding.IGNORED_WORDS)}
    names = {name_key(value): value.strip() for value in fields
             if name_key(value) and name_key(value) not in excluded}
    if len(names) != 1:
        return None
    name = next(iter(names.values()))
    return name if len(name) <= 100 and design_coding.safe_stem(name) else None


def mapping(db, batch, mains):
    base = api_url()
    tokens = design_coding.package_tokens(batch.root_path)
    rows = [{"main_image_id": main.id, "original_filename": main.filename} for main in mains]
    work = {"api_url": base, "user_id": operator_id(), "mains": [], "receipts": {}, "pending": None,
            "new_themes": []}
    result = {"mode": "TICKET_API", "tokens": tokens, "rows": rows, "ticket": work}
    url = urlsplit(base)
    if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.query or url.fragment:
        return {**result, "error": "请核对正式领号接口地址。"}
    profile = db.get(DesignPackageInfo, batch.id)
    product_name = ((json.loads(profile.manual_json or "{}").get("product_name") if profile else None)
                    or tokens["category"])
    try:
        products = catalog_rows(base, "/products")
        products = [p for p in products if name_key(p.get("productName")) == name_key(product_name)]
        if len(products) != 1:
            return {**result, "error": f"领号系统未能唯一匹配品类「{product_name}」，请在设计包信息中填写领号产品全名。"}
        product = products[0]
        name = product["productName"]
        types = catalog_rows(base, "/themes/by-product", productName=name)
        term = name_key(tokens["subtheme"])
        names = {term}
        fan_theme = tokens["theme"] == "球迷主题"
        if fan_theme:
            names.add(f'{term.removesuffix("球队").strip()}球队')
        matching = [entry for entry in types if term and name_key(entry.get("themeType")) in names]
        # In the current ticket catalog, holiday names such as 圣诞节 are
        # child themes under 节日. Resolve them by exact child name.
        by_holiday = not matching and tokens["theme"] == "经典主题"
        if by_holiday:
            matching = [entry for entry in types if name_key(entry.get("themeType")) in {"节日", "节日主题"}]
        if len(matching) != 1:
            return {**result, "error": f"领号系统未能唯一匹配主题「{tokens['subtheme']}」。"}
        theme = matching[0]
        children = catalog_rows(base, "/themes/by-product-and-type-name",
                                productName=name, themeType=theme["themeType"])
        if by_holiday and not any(name_key(entry.get("childThemeName")) == term for entry in children):
            return {**result, "error": f"领号系统的节日主题没有「{tokens['subtheme']}」映射。"}
        category_prefix = product.get("prefix") or ""
        theme_prefix = theme.get("themeTypePrefix") or ""
        if (type(product.get("id")) is not int or product["id"] <= 0
                or type(theme.get("id")) is not int or theme["id"] <= 0
                or not all(design_coding.OFFICIAL_TEAM_PREFIX.fullmatch(p) for p in (category_prefix, theme_prefix))):
            return {**result, "error": "领号系统返回的产品或主题前缀无效。"}
    except (httpx.HTTPError, KeyError, TypeError, ValueError):
        return {**result, "error": "无法读取领号系统的映射，请核对接口后重试。"}

    prefixes = {"category": category_prefix, "theme": theme_prefix}
    result.update(prefixes=prefixes, source_code_prefix=design_coding.number_prefix(prefixes),
                  ticket_catalog={"product_name": name, "theme_name": theme["themeType"]})
    work["product_id"] = product["id"]
    work["theme_type_id"] = theme["id"]
    if work["user_id"] is None:
        result["error"] = "请配置已授权的领号用户，再确认编码改名。"
    subjects = {
        (entry["id"], entry["childThemeName"].strip(), entry["childThemePrefix"])
        for entry in children
        if type(entry.get("id")) is int and entry["id"] > 0
        and name_key(entry.get("childThemeName")) and isinstance(entry.get("childThemePrefix"), str)
        and design_coding.OFFICIAL_TEAM_PREFIX.fullmatch(entry["childThemePrefix"])
    }
    ids = [main.id for main in mains]
    contents = {entry.main_image_id: entry for entry in db.scalars(select(MainImageContent).where(
        MainImageContent.main_image_id.in_(ids)
    ))}
    existing = {entry.main_image_id: entry for entry in db.scalars(select(DesignCode).where(
        DesignCode.main_image_id.in_(ids), DesignCode.applied.is_(True)
    ))}
    for main, row in zip(mains, rows):
        if main.id in existing:
            row.update(design_coding.proposal(existing[main.id]))
            continue
        content = contents.get(main.id)
        data, error = design_coding.identification_content(content)
        if error:
            row["error"] = error
            continue
        if tokens["theme"] == "球迷主题":
            candidates = design_coding.official_team_candidates(main.filename, data, tokens)
        else:
            candidates = {name_key(subject) for subject in data.get("main_subject", [])}
        choices = {(theme_id, subject, prefix) for theme_id, subject, prefix in subjects
                   if name_key(subject) in candidates}
        if len(choices) != 1:
            row["error"] = "需人工核对领号系统的小主题全名。"
            missing = new_theme_name(data, fan_theme) if not choices else None
            if missing and not any(name_key(entry.get("childThemeName")) == name_key(missing) for entry in children):
                try:
                    sha = design_coding.file_hash(main.file_path)
                except OSError:
                    row["error"] = "主图不可读取。"
                    continue
                if content.source_sha256 and content.source_sha256 != sha:
                    row["error"] = "主图已经变化，请先重新识别。"
                    continue
                work["new_themes"].append({"main_image_id": main.id, "name": missing,
                                           "source_sha256": sha, "signature": content_signature(content)})
                row.update(subject_name=missing, new_theme_name=missing, error="可自动补全缩写。")
            continue
        theme_id, subject, abbreviation = choices.pop()
        if len({name_key(other) for _, other, prefix in subjects if prefix == abbreviation}) != 1:
            row["error"] = "该缩写对应多个小主题，需人工核对。"
            continue
        try:
            sha = design_coding.file_hash(main.file_path)
        except OSError:
            row["error"] = "主图不可读取。"
            continue
        if content.source_sha256 and content.source_sha256 != sha:
            row["error"] = "主图已经变化，请先重新识别。"
            continue
        if not design_coding.safe_stem(subject + result["source_code_prefix"] + "0" * 12):
            row["error"] = "小主题名称无法用作文件名，请核对映射。"
            continue
        selection = {"main_image_id": main.id, "theme_id": theme_id, "subject_name": subject,
                     "abbreviation": abbreviation, "source_path": main.file_path, "source_sha256": sha,
                     "content_signature": content_signature(content), "original_filename": main.filename}
        work["mains"].append(selection)
        row.update(subject_name=subject, abbreviation=abbreviation, decision="OFFICIAL")
    return result


def claim_one(work, item, prefix):
    response = request("POST", work["api_url"], "/ticket/claim", json={
        "productId": work["product_id"], "userId": work["user_id"],
        "themeId": item["theme_id"], "quantity": 1,
    })
    code = response.get("startCode") if isinstance(response, dict) else None
    if (not isinstance(code, str) or code != response.get("endCode")
            or not re.fullmatch(re.escape(prefix) + r"[0-9]{1,12}", code)):
        raise ValueError("领号系统返回的起止编码与预览前缀或数量不符")
    return code


def apply_receipt(db, work, item):
    code = work["receipts"][str(item["main_image_id"])]
    row = db.get(DesignCode, item["main_image_id"])
    if row is None:
        row = DesignCode(main_image_id=item["main_image_id"], applied=False)
        db.add(row)
    row.context_key = dump({"product_id": work["product_id"], "theme_id": item["theme_id"]})
    row.subject_key = name_key(item["subject_name"])
    row.abbreviation = item["abbreviation"]
    row.source_code = code
    row.operation_code = f"{row.abbreviation}-{code}"
    row.original_filename = item["original_filename"]
    row.target_stem = item["subject_name"] + code
    row.source_sha256 = item["source_sha256"]
    row.content_signature = item["content_signature"]
    row.decision = "TICKET_ALLOCATED"
    return row
