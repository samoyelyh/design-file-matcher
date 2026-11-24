"""Complete missing small themes through the ticket catalog's add-only API."""

from __future__ import annotations

import os
import re
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import httpx

from app.services import design_coding, ticket_coding


def enabled():
    return os.getenv("DESIGN_MATCHER_TICKET_THEME_CREATE", "").strip() == "1"


def name_key(name):
    # MySQL's catalog collation treats common case and accent variants alike.
    text = unicodedata.normalize("NFKD", ticket_coding.name_key(name))
    return "".join(char for char in text if not unicodedata.combining(char))


def choose_prefix(name, occupied):
    try:
        from pypinyin import lazy_pinyin

        name = " ".join(lazy_pinyin(name))
    except ImportError:
        pass
    letters = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    used = defaultdict(set, {prefix: {"occupied"} for prefix in occupied})
    suggestion = design_coding.recommend(letters, used)
    if suggestion:
        return suggestion
    seed = ("".join(re.findall(r"[A-Z]+", letters.upper())) or "ZT")[:8]
    for number in range(2, 10000):
        prefix = f"{seed}{number}"
        if prefix not in occupied:
            return prefix
    raise ValueError("当前目录没有找到可用缩写。")


def fill(base, product_id, parent_id, names):
    """Check each product that shares this parent group, then append new rows only."""
    types = ticket_coding.catalog_rows(base, "/themes/types")
    parent = next((row for row in types if row.get("id") == parent_id), None)
    if not parent or product_id not in parent["productIds"]:
        raise ValueError("该品类的主题归属已更新，请重新预览。")
    affected = set(parent["productIds"])
    related = [row["id"] for row in types if affected.intersection(row["productIds"])]

    def read(group):
        return ticket_coding.catalog_rows(base, f"/themes/types/{group}/children")

    with ThreadPoolExecutor(max_workers=5) as executor:
        by_type = dict(zip(related, executor.map(read, related)))
    occupied = {row["prefix"].strip().upper() for children in by_type.values() for row in children}
    parent_rows = by_type[parent_id]
    path = f"/themes/types/{parent_id}/children"
    results = []
    for name in dict((name_key(name), name) for name in names).values():
        existing = next((row for row in parent_rows if name_key(row["name"]) == name_key(name)), None)
        if existing:
            results.append({"name": existing["name"], "prefix": existing["prefix"], "status": "EXISTS"})
            continue
        prefix = choose_prefix(name, occupied)
        acknowledged = True
        try:
            ticket_coding.request("POST", base, path, json={"name": name, "prefix": prefix})
        except (httpx.HTTPError, ValueError):
            acknowledged = False
        try:
            parent_rows = read(parent_id)
        except (httpx.HTTPError, ValueError):
            parent_rows = []
        present = next((row for row in parent_rows if name_key(row["name"]) == name_key(name)), None)
        if not present:
            results.append({"name": name, "prefix": prefix, "status": "NEEDS_CHECK",
                            "message": "新增结果未能核实，请先刷新预览。"})
            break
        occupied.update(row["prefix"].strip().upper() for row in parent_rows)
        status = "CREATED" if acknowledged and present["prefix"].upper() == prefix else "EXISTS"
        results.append({"name": present["name"], "prefix": present["prefix"], "status": status})
    return results
