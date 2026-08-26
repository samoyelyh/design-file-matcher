from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")
APP_JS = (ROOT / "app" / "static" / "app.js").read_text(encoding="utf-8")
STATE_JS = ROOT / "app" / "static" / "workspace_state.js"


def derive(batch: dict, review: dict | None = None, plan: dict | None = None) -> dict:
    script = (
        f"const x=require({json.dumps(str(STATE_JS))});"
        f"console.log(JSON.stringify(x.deriveWorkspaceState({json.dumps(batch)},"
        f"{json.dumps(review or {})},{json.dumps(plan or {})})));"
    )
    result = subprocess.run(
        ["node", "-e", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_stepper_state_follows_real_batch_state():
    batch = {
        "main_count": 32,
        "categories": {"观赛图": 32},
        "status": "SCANNED",
        "mask_readiness": {
            "ready": False,
            "required_categories": ["观赛图"],
            "configured_categories": [],
        },
    }
    state = derive(batch)
    assert state["currentStep"] == "regions"
    assert state["steps"] == {
        "import": "complete",
        "regions": "current",
        "match": "upcoming",
        "review": "upcoming",
    }


def test_unfinished_regions_cannot_enter_matching():
    batch = {
        "main_count": 1,
        "categories": {"观赛图": 1},
        "status": "SCANNED",
        "mask_readiness": {
            "ready": False,
            "required_categories": ["观赛图"],
            "configured_categories": [],
        },
    }
    assert derive(batch)["canEnter"]["process"] is False
    assert "请先完成识别区域设置" in APP_JS


def test_psd_is_clearly_marked_as_no_setup_required():
    assert "PSD" in HTML
    assert "无需设置识别区域" in APP_JS


def test_product_category_setting_is_not_exposed():
    assert 'id="productCategory"' not in HTML
    assert 'aria-label="品类"' not in HTML
    assert "product_category" not in APP_JS


def test_review_defaults_to_pending_filter():
    assert 'filter: "pending"' in APP_JS
    assert 'pending: "待确认"' in APP_JS


def test_technical_information_is_collapsed_by_default():
    assert HTML.count('class="advanced-details"') >= 4
    assert '<details open' not in HTML
    assert "高级设置与处理详情" in HTML


def test_refresh_restores_batch_and_view_from_hash():
    assert "function hashState()" in APP_JS
    assert "requested.batchId" in APP_JS
    assert "selectBatch(requested.batchId, requested.view)" in APP_JS


def test_rename_requires_business_confirmation_modal():
    assert 'id="renameConfirmModal"' in HTML
    assert "不会覆盖已有同名文件" in HTML
    assert "confirmExecuteRename" in APP_JS


def test_unmatched_copy_is_not_file_missing():
    assert 'UNMATCHED: "未匹配"' in APP_JS
    assert 'UNMATCHED: "文件缺失"' not in APP_JS
