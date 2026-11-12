from __future__ import annotations

import json
from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment


def auto_eligible(result: dict) -> bool:
    fields = ("same_design", "color_match", "pattern_match", "local_details_match")
    return (result["verdict"] == "MATCH" and result["evidence_sufficient"]
            and bool(result["matched_features"]) and all(result[field] is not False for field in fields)
            and not result["needs_review"] and not result["differences"]
            and all(item["state"] != "CONFLICT" for item in result["observations"].values())
            and any(item["state"] == "CONSISTENT" for item in result["observations"].values()))


def solve_assignment(n_a: int, n_b: int, edges: dict[tuple[int, int], int]) -> dict[int, int]:
    """Maximum-weight assignment with zero-value dummy columns for unmatched A images."""
    if not n_a or not n_b:
        return {}
    weights = np.full((n_a, n_b + n_a), -1_000_000, dtype=np.int64)
    weights[:, n_b:] = 0
    for (a, b), score in edges.items():
        if 0 <= a < n_a and 0 <= b < n_b and 0 <= score <= 100:
            weights[a, b] = score + 1
    rows, columns = linear_sum_assignment(weights, maximize=True)
    return {int(a): int(b) for a, b in zip(rows, columns)
            if b < n_b and weights[a, b] > 0}


def ranked_rows(pairs: list, n_a: int, n_b: int) -> dict[int, list[tuple[int, dict]]]:
    grouped = {a: [] for a in range(n_a)}
    for row in pairs:
        if row["status"] == "DONE" and row["result"]:
            grouped[row["a"]].append((row["b"], json.loads(row["result"])))
    for choices in grouped.values():
        choices.sort(key=lambda item: (-item[1]["match_score"], item[0]))
    return grouped


def reviews_needed(pairs: list, n_a: int, n_b: int, config: dict) -> list[tuple[int, int, int]]:
    if n_b < 2:
        return []
    grouped = ranked_rows(pairs, n_a, n_b)
    # A failed or pending comparison means the actual Top2 is unknown.
    complete = {a: choices for a, choices in grouped.items() if len(choices) == n_b}
    leading = Counter(choices[0][0] for choices in complete.values()
                      if choices[0][1]["verdict"] != "DIFFERENT")
    requested = []
    for a, choices in complete.items():
        (b1, first), (b2, second) = choices[:2]
        plausible = any(r["verdict"] in ("MATCH", "UNCERTAIN") for _, r in choices[:2])
        close = plausible and first["match_score"] - second["match_score"] < config["review_margin"]
        unclear = first["verdict"] == "UNCERTAIN" or first["needs_review"]
        similar_variant = (first["pattern_match"] is True and second["pattern_match"] is True
                           and (first["color_match"] != second["color_match"]
                                or first["local_details_match"] != second["local_details_match"]))
        multiple = sum(auto_eligible(r) for _, r in choices) > 1
        if config["review_mode"] == "all" or close or unclear or similar_variant or multiple or leading[b1] > 1:
            requested.append((a, b1, b2))
    return requested


def build_results(plan: dict, pairs: list, reviews: list) -> dict:
    n_a, n_b = len(plan["images_a"]), len(plan["images_b"])
    config = plan["settings"]
    grouped = ranked_rows(pairs, n_a, n_b)
    pair_status = {(r["a"], r["b"]): r["status"] for r in pairs}
    review_by_a = {r["a"]: r for r in reviews}
    edges: dict[tuple[int, int], int] = {}
    details: dict[tuple[int, int], dict] = {}
    incomplete = {a for a in range(n_a) if len(grouped[a]) != n_b}
    for a in range(n_a):
        if a in incomplete:
            continue
        review = review_by_a.get(a)
        if review:
            if review["status"] != "DONE":
                continue
            decision = json.loads(review["result"])
            if decision["winner"] not in {"CANDIDATE_1", "CANDIDATE_2"}:
                continue
            key = decision["winner"].lower()
            b = review["b1"] if key == "candidate_1" else review["b2"]
            revised = decision[key]
            original = next(result for index, result in grouped[a] if index == b)
            if original["verdict"] == "DIFFERENT" or any(original[k] is False for k in
                    ("same_design", "color_match", "pattern_match", "local_details_match")):
                continue
            if auto_eligible(revised):
                edges[a, b] = revised["match_score"]
                details[a, b] = revised
        else:
            for b, result in grouped[a]:
                if auto_eligible(result):
                    edges[a, b] = result["match_score"]
                    details[a, b] = result
    # A pending comparison or review can change the global optimum.
    reviews_incomplete = any(review["status"] != "DONE" for review in reviews)
    global_ready = not incomplete and not reviews_incomplete
    assignment = solve_assignment(n_a, n_b, edges) if global_ready else {}
    rows = []
    for a in range(n_a):
        choices = grouped[a]
        review = review_by_a.get(a)
        if a in incomplete:
            failed = any(pair_status.get((a, b)) == "ERROR" for b in range(n_b))
            status, reason = ("ERROR", "候选比较失败，无法确定完整排名。") if failed else ("REVIEW", "候选比较尚未完成。")
        elif incomplete:
            status, reason = "REVIEW", "完整比较矩阵尚未完成，暂不生成全局匹配。"
        elif reviews_incomplete:
            if review and review["status"] == "ERROR":
                status, reason = "ERROR", "Top2 复核失败。"
            else:
                status, reason = "REVIEW", "Top2 复核尚未全部完成，暂不生成全局匹配。"
        elif a in assignment:
            status, reason = "MATCHED", "模型建议匹配，尚未经过人工确认。"
        elif review and json.loads(review["result"])["winner"] == "NONE":
            if all(r["verdict"] == "DIFFERENT" for _, r in choices):
                status, reason = "UNMATCHED", "所有候选被判断为不匹配。"
            else:
                status, reason = "REVIEW", "Top2 均未通过复核，其他候选需要人工检查。"
        elif any(r["verdict"] != "DIFFERENT" for _, r in choices):
            status, reason = "REVIEW", "证据不足、复核不确定或候选存在一对一竞争。"
        else:
            status, reason = "UNMATCHED", "所有候选被判断为不匹配。"
        selected = assignment.get(a)
        rows.append({"a_index": a, "a": plan["images_a"][a], "status": status, "status_reason": reason,
                     "b_index": selected, "b": plan["images_b"][selected] if selected is not None else None,
                     "selected_result": details.get((a, selected)) if selected is not None else None,
                     "top2": [{"b_index": b, "b": plan["images_b"][b], "result": result} for b, result in choices[:2]],
                     "review": dict(review) if review else None})
    used = set(assignment.values())
    return {"plan_id": plan["plan_id"], "rows": rows,
            "unmatched_b": [plan["images_b"][b] for b in range(n_b) if b not in used],
            "counts": dict(Counter(row["status"] for row in rows)),
            "matrix_complete": len(incomplete) == 0, "global_ready": global_ready}
