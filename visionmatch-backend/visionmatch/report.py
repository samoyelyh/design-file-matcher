from __future__ import annotations

import csv
import html
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .matching import build_results
from .provider import file_hash
from .store import RunLock, connect, read_plan


def human_decision(run_dir: Path, a: int, b: int | None, decision: str, note: str) -> None:
    plan = read_plan(run_dir)
    if not 0 <= a < len(plan["images_a"]):
        raise ValueError("A 图片编号无效。")
    if decision not in {"MATCH", "UNMATCHED"}:
        raise ValueError("人工决定只能是 MATCH 或 UNMATCHED。")
    if decision == "MATCH" and (b is None or not 0 <= b < len(plan["images_b"])):
        raise ValueError("确认匹配时必须提供有效的 B 图片编号。")
    if decision == "UNMATCHED" and b is not None:
        raise ValueError("确认未匹配时不能提供 B 图片编号。")
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with RunLock(run_dir), connect(run_dir / "run.sqlite3") as db:
        if b is not None and db.execute("SELECT a FROM human_decisions WHERE b=? AND decision='MATCH' AND a<>?", (b, a)).fetchone():
            raise ValueError("该 B 图片已经被其他人工确认结果占用。")
        db.execute("INSERT OR REPLACE INTO human_decisions VALUES (?,?,?,?,?)", (a, b, decision, note, timestamp))
        db.execute("INSERT INTO human_history(a,b,decision,note,updated_at) VALUES (?,?,?,?,?)", (a, b, decision, note, timestamp))


def apply_human(automatic: dict, plan: dict, decisions: list) -> dict:
    rows = json.loads(json.dumps(automatic["rows"], ensure_ascii=False))
    for decision in decisions:
        a, b = decision["a"], decision["b"]
        if decision["decision"] == "MATCH":
            for other in rows:
                if other["a_index"] != a and other["b_index"] == b:
                    other.update(status="REVIEW", status_reason="候选已被人工确认分配给其他主图。",
                                 b_index=None, b=None, selected_result=None)
            rows[a].update(status="MATCHED", status_reason="人工确认匹配。", b_index=b,
                           b=plan["images_b"][b], selected_result=None)
        else:
            rows[a].update(status="UNMATCHED", status_reason="人工确认未匹配。", b_index=None,
                           b=None, selected_result=None)
        rows[a]["human_decision"] = dict(decision)
    used = {row["b_index"] for row in rows if row["b_index"] is not None}
    return {"rows": rows, "counts": dict(Counter(row["status"] for row in rows)),
            "unmatched_b": [image for index, image in enumerate(plan["images_b"]) if index not in used]}


def safe_assets(report_dir: Path, plan: dict) -> dict[str, str]:
    folder = report_dir / "assets"
    folder.mkdir()
    assets = {}
    for image in plan["images_a"] + plan["images_b"]:
        source = Path(image["path"])
        if not source.is_file() or file_hash(source) != image["sha256"]:
            continue
        destination = folder / (image["sha256"] + source.suffix.lower())
        if not destination.exists():
            shutil.copyfile(source, destination)
        assets[image["path"]] = "assets/" + destination.name
    return assets


def html_report(result: dict, assets: dict[str, str], statistics: dict, plan: dict) -> str:
    esc = lambda value: html.escape(str(value), quote=True)
    def image_block(image: dict | None, label: str) -> str:
        if not image:
            return f'<div class="image"><b>{esc(label)}</b><p>无</p></div>'
        source = assets.get(image["path"])
        rendered = f'<img loading="lazy" src="{esc(source)}" alt="{esc(label)}">' if source else '<p>图片不存在或已变化</p>'
        return f'<div class="image"><b>{esc(label)}</b>{rendered}<small>{esc(image["name"])}</small></div>'
    cards = []
    for row in result["rows"]:
        top = row["top2"]
        backup_item = next((item for item in top if item["b_index"] != row["b_index"]), None)
        backup = backup_item["b"] if backup_item else None
        selected_label = f"匹配结果 B{row['b_index'] + 1:03d}" if row["b_index"] is not None else "匹配结果"
        backup_label = f"Top2 备选 B{backup_item['b_index'] + 1:03d}" if backup_item else "Top2 备选"
        evidence = row.get("selected_result") or (top[0]["result"] if top else {})
        notes = "；".join(evidence.get("differences", []) + evidence.get("uncertainties", [])) or evidence.get("reason", "无")
        cards.append(f'''<section class="card"><div class="heading"><strong>A{row['a_index'] + 1:03d} · {esc(row['a']['name'])}</strong>
            <span class="badge {esc(row['status'])}">{esc(row['status'])}</span></div>
            <p>{esc(row['status_reason'])}</p><div class="images">{image_block(row['a'], '商品主图')}
            {image_block(row['b'], selected_label)}{image_block(backup, backup_label)}</div>
            <p><b>视觉说明：</b>{esc(notes)}</p></section>''')
    unassigned = {item["path"] for item in result["unmatched_b"]}
    unassigned_list = "".join(f"<li>B{index+1:03d} · {esc(item['name'])}</li>" for index, item in enumerate(plan["images_b"]) if item["path"] in unassigned)
    return '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>VisionMatch AI 配对报告</title><style>
    body{font-family:system-ui,'Microsoft YaHei',sans-serif;background:#f1f5f9;color:#172b42;margin:0}
    main{max-width:1250px;margin:auto;padding:28px} h1{margin:0 0 8px} .sub{color:#52677e}
    .card{background:white;border:1px solid #dce5ee;border-radius:14px;margin:18px 0;padding:20px}
    .heading{display:flex;justify-content:space-between;gap:10px;align-items:center}
    .badge{font-size:12px;border-radius:99px;padding:5px 10px;background:#e9eff8}.MATCHED{background:#d6f4e3;color:#126537}
    .REVIEW{background:#fff0cd;color:#825600}.ERROR{background:#ffe0e0;color:#912929}.UNMATCHED{background:#e9eff8}
    .images{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.image{background:#f7f9fc;padding:10px;border-radius:10px}
    .image img{display:block;width:100%;height:280px;object-fit:contain;background:#fff;margin:8px 0}
    small{overflow-wrap:anywhere;color:#617286}@media(max-width:700px){.images{grid-template-columns:1fr}}
    </style></head><body><main><h1>VisionMatch AI 配对报告</h1>''' + f'''<p class="sub">模型建议与人工确认分别标识；结果可为待复核或未匹配。</p>
    <p>比较完成：{statistics['pairs_done']}/{statistics['pairs_total']} · 复核完成：{statistics['reviews_done']}/{statistics['reviews_total']} · CLI 调用：{statistics['cli_attempts']}</p>''' + "".join(cards) + f'<section class="card"><h2>未分配候选图片（{len(unassigned)}）</h2><ul>{unassigned_list}</ul></section></main></body></html>'


def export_report(run_dir: Path) -> Path:
    plan = read_plan(run_dir)
    with RunLock(run_dir), connect(run_dir / "run.sqlite3") as db:
        pairs = db.execute("SELECT * FROM pairs ORDER BY a,b").fetchall()
        reviews = db.execute("SELECT * FROM reviews ORDER BY a").fetchall()
        attempts = [dict(row) for row in db.execute("SELECT * FROM attempts ORDER BY id")]
        decisions = db.execute("SELECT * FROM human_decisions ORDER BY a").fetchall()
        history = [dict(row) for row in db.execute("SELECT * FROM human_history ORDER BY id")]
        automatic = build_results(plan, pairs, reviews)
        result = apply_human(automatic, plan, decisions)
        queue_exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='queue_jobs'").fetchone() is not None
        queue_jobs = [dict(row) for row in db.execute('SELECT * FROM queue_jobs ORDER BY kind,a,b1')] if queue_exists else []
        queue_events = [dict(row) for row in db.execute('SELECT * FROM queue_events ORDER BY id')] if queue_exists else []
        sessions = [dict(row) for row in db.execute('SELECT * FROM queue_sessions ORDER BY started')] if queue_exists else []
        controls = [dict(row) for row in db.execute('SELECT * FROM queue_control_events ORDER BY id')] if queue_exists else []
        complete = automatic['global_ready'] and all(row['state']=='SUCCESS' for row in queue_jobs)
        usage_total = Counter()
        missing_usage = 0
        for attempt in attempts:
            usage = json.loads(attempt["usage"]) if attempt["usage"] else None
            if isinstance(usage, dict):
                usage_total.update({k: v for k, v in usage.items() if isinstance(v, int) and not isinstance(v, bool)})
            else:
                missing_usage += 1
        durations = [attempt["elapsed_seconds"] for attempt in attempts if attempt["elapsed_seconds"] is not None]
        statistics = {"pairs_done": sum(row["status"] == "DONE" for row in pairs), "pairs_total": len(pairs),
                      "complete": complete,
                      "reviews_done": sum(row["status"] == "DONE" for row in reviews), "reviews_total": len(reviews),
                      "cli_attempts": len(attempts), "failed_attempts": sum(row["status"] != "SUCCESS" for row in attempts),
                      "total_recorded_request_seconds": round(sum(durations), 3), "usage_total": dict(usage_total), "attempts_without_usage": missing_usage,
                      "subscription_usage": "UNKNOWN: 订阅额度为账号级共享指标，不能从 token 精确换算",
                      "api_cost": "NOT_APPLICABLE: 使用 ChatGPT 登录的 Codex CLI"}
        final_name = ("report_" if complete else "partial_report_") + datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:6]
        report_dir = run_dir / ('.building_' + final_name)
        report_dir.mkdir(exist_ok=False)
        assets = safe_assets(report_dir, plan)
        payload = {"plan": plan, "automatic": automatic, "final": result, "statistics": statistics,
                   "pairs": [dict(row) for row in pairs], "reviews": [dict(row) for row in reviews],
                   "attempts": attempts, "human_history": history, "queue_jobs": queue_jobs,
                   "queue_events": queue_events, "execution_sessions": sessions, "queue_control_events": controls}
        (report_dir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        with (report_dir / "results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            fields = ["主图编号", "主图文件名", "主图路径", "白底图编号", "白底图文件名", "白底图路径", "匹配分数",
                      "颜色判断", "图案判断", "局部判断", "Top2候选", "复核结果", "最终状态", "差异说明", "需要人工审核"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in result["rows"]:
                relevant = next((item["result"] for item in row["top2"] if item["b_index"] == row["b_index"]), None)
                visual = row["selected_result"] or relevant or (row["top2"][0]["result"] if row["top2"] and row["b_index"] is None else {})
                writer.writerow({"主图编号": row["a_index"] + 1, "主图文件名": row["a"]["name"], "主图路径": row["a"]["path"],
                                 "白底图编号": row["b_index"] + 1 if row["b_index"] is not None else "",
                                 "白底图文件名": row["b"]["name"] if row["b"] else "", "白底图路径": row["b"]["path"] if row["b"] else "",
                                 "匹配分数": visual.get("match_score", ""), "颜色判断": visual.get("color_match", ""),
                                 "图案判断": visual.get("pattern_match", ""), "局部判断": visual.get("local_details_match", ""),
                                 "Top2候选": " | ".join(x["b"]["name"] for x in row["top2"]),
                                 "复核结果": row["review"]["result"] if row["review"] else "",
                                 "最终状态": row["status"], "差异说明": "；".join(visual.get("differences", []) + visual.get("uncertainties", [])) or row["status_reason"],
                                 "需要人工审核": row["status"] in ("REVIEW", "ERROR")})
            used_b = {row["b_index"] for row in result["rows"] if row["b_index"] is not None}
            unassigned_status = "REVIEW" if any(row["status"] in ("REVIEW", "ERROR") for row in result["rows"]) else "UNMATCHED"
            for index, image in enumerate(plan["images_b"]):
                if index not in used_b:
                    writer.writerow({"白底图编号": index + 1, "白底图文件名": image["name"], "白底图路径": image["path"],
                                     "最终状态": unassigned_status, "差异说明": "候选图片尚未分配", "需要人工审核": unassigned_status == "REVIEW"})
        (report_dir / "index.html").write_text(html_report(result, assets, statistics, plan), encoding="utf-8")
        if not complete:
            html_path=report_dir/'index.html'
            html_path.write_text(html_path.read_text(encoding='utf-8').replace('<h1>VisionMatch AI 配对报告</h1>',
                '<h1>任务尚未完整完成：阶段报告</h1>'),encoding='utf-8')
        final_dir = run_dir / final_name
        report_dir.rename(final_dir)
    return final_dir
