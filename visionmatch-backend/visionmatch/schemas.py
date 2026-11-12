from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

OBSERVATION_NAMES = ("body_color", "color_blocks", "neck_and_cuffs", "sleeves",
                     "print_and_layout", "text_and_numbers", "other_details")
OBSERVATION_SCHEMA = {
    "type": "object",
    "properties": {
        "state": {"type": "string", "enum": ["CONSISTENT", "CONFLICT", "NOT_VISIBLE"]},
        "detail": {"type": "string"},
    },
    "required": ["state", "detail"],
    "additionalProperties": False,
}

PAIR_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["MATCH", "DIFFERENT", "UNCERTAIN"]},
        "same_design": {"type": ["boolean", "null"]},
        "color_match": {"type": ["boolean", "null"]},
        "pattern_match": {"type": ["boolean", "null"]},
        "local_details_match": {"type": ["boolean", "null"]},
        "match_score": {"type": "integer", "minimum": 0, "maximum": 100},
        "observations": {"type": "object", "properties": {name: OBSERVATION_SCHEMA for name in OBSERVATION_NAMES},
                         "required": list(OBSERVATION_NAMES), "additionalProperties": False},
        "matched_features": {"type": "array", "items": {"type": "string"}},
        "differences": {"type": "array", "items": {"type": "string"}},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
        "evidence_sufficient": {"type": "boolean"},
        "reason": {"type": "string"},
        "needs_review": {"type": "boolean"},
    },
    "required": ["verdict", "same_design", "color_match", "pattern_match", "local_details_match", "match_score", "observations", "matched_features", "differences", "uncertainties", "evidence_sufficient", "reason", "needs_review"],
    "additionalProperties": False,
}

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "winner": {"type": "string", "enum": ["CANDIDATE_1", "CANDIDATE_2", "NONE", "UNCERTAIN"]},
        "candidate_1": PAIR_SCHEMA,
        "candidate_2": PAIR_SCHEMA,
        "reason": {"type": "string"},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["winner", "candidate_1", "candidate_2", "reason", "uncertainties"],
    "additionalProperties": False,
}

SHARED_RULES = """Compare the clothing product identity, not the image identity.
目标是判断两张不同图片展示的衣服是否属于同一商品设计、同一配色，不要求同一件实物或相同照片。
忽略模特身份、性别、肤色、体型、姿势、背景、拍摄角度、正常褶皱、拉伸变形、合理光照曝光白平衡、裁剪及分辨率差异；这些不能单独构成商品冲突。
同一球队或主题不代表同款；同印花但不同商品配色也不是同款。比较衣身颜色及颜色分区、领口袖口及左右袖、印花及布局、文字数字和其他可辨别商品的细节。
observations 的每个类别区分 CONSISTENT（对应可见特征一致）、CONFLICT（可见且足以区分商品的明确设计或配色冲突）、NOT_VISIBLE（被遮挡、裁剪、模糊或不可辨认）。
遮挡是证据缺失，不是差异。禁止补猜不可见特征；不确定的布尔字段可用 null。matched_features 列出可见的一致特征，differences 列出真实可见且足以区分商品的冲突，uncertainties 列出未观察到的内容。
MATCH：现有可见证据充分支持同款同配色且不存在明确商品冲突。即使有局部遮挡，只要剩余关键特征足够也可 MATCH，不要仅因遮挡标记 needs_review。
DIFFERENT：存在足以区分商品的明确设计或配色冲突；图片不同、人物不同及看不到内容不能构成 DIFFERENT。
UNCERTAIN：无法得到足够证据支持 MATCH 或 DIFFERENT。evidence_sufficient 为 true 表示可见证据足以支持当前 MATCH 或 DIFFERENT，UNCERTAIN 时为 false。
match_score 为 0–100 排序分值，不是统计置信概率；不能仅由分数决定商品结论。needs_review 仅用于证据不足、无法裁决或实际存在需人工确认的风险。
不得依据图片文件名、路径或外部信息。请直接观察实际图片并只返回符合指定 JSON Schema 的结果，不要运行命令、使用工具或访问文件。"""

PAIR_PROMPT = """你是服装商品身份比对员。随消息附带两张实际图片：第一张为商品主图 A，第二张为候选 B。
""" + SHARED_RULES

REVIEW_PROMPT = """你是独立服装商品身份裁决员。随消息附带三张实际图片，顺序为主图 A、候选 1、候选 2。
分别重新观察 A 和每个候选并填写各自完整判断，不使用先前分数或分析。若现有证据足以支持唯一候选，选择 CANDIDATE_1 或 CANDIDATE_2；两件均有明确商品冲突则选择 NONE；不能可靠区分或证据不足则选择 UNCERTAIN。
""" + SHARED_RULES


def validate_output(value: object, schema: dict) -> None:
    errors = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda e: str(list(e.path)))
    if errors:
        raise ValueError("输出不符合 JSON Schema: " + errors[0].message)
    pairs = [value] if schema is PAIR_SCHEMA else [value["candidate_1"], value["candidate_2"]]
    fields = ("same_design", "color_match", "pattern_match", "local_details_match")
    for pair in pairs:
        conflicts = [x for x in pair["observations"].values() if x["state"] == "CONFLICT"]
        if pair["verdict"] == "MATCH" and (any(pair[f] is False for f in fields) or conflicts or pair["differences"]):
            raise ValueError("输出矛盾：MATCH 同时包含明确不一致的关键字段。")
        has_observed_match = any(item["state"] == "CONSISTENT" for item in pair["observations"].values())
        if pair["verdict"] == "MATCH" and (not pair["evidence_sufficient"] or not pair["matched_features"] or not has_observed_match):
            raise ValueError("输出矛盾：MATCH 缺少充分的可见一致证据。")
        if pair["verdict"] == "DIFFERENT" and (not pair["evidence_sufficient"] or not pair["differences"] or not conflicts):
            raise ValueError("输出矛盾：DIFFERENT 缺少足以区分商品的明确可见冲突。")
        if pair["verdict"] == "UNCERTAIN" and pair["evidence_sufficient"]:
            raise ValueError("输出矛盾：UNCERTAIN 不能声称证据足以作出确定判断。")
    if schema is REVIEW_SCHEMA and value["winner"] in ("CANDIDATE_1", "CANDIDATE_2"):
        winner = value[value["winner"].lower()]
        if winner["verdict"] != "MATCH":
            raise ValueError("输出矛盾：选择的候选未被判断为 MATCH。")


def schema_hash(schema: dict) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(schema, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def write_schema(path: Path, schema: dict) -> None:
    path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
