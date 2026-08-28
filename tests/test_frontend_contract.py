from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")
APP_JS = (ROOT / "app" / "static" / "app.js").read_text(encoding="utf-8")
CSS = (ROOT / "app" / "static" / "styles.css").read_text(encoding="utf-8")
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
        encoding="utf-8",
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


def test_dino_matching_requires_manual_regions():
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
    state = derive(batch)
    assert state["canEnter"]["mask"] is True
    assert state["canEnter"]["process"] is False
    assert "开始遮罩 DINO 自动匹配" in APP_JS


def test_psd_is_clearly_marked_as_dino_preview():
    assert "DINO Preview" in HTML
    assert "PSD Preview" in APP_JS


def test_product_category_setting_is_not_exposed():
    assert 'id="productCategory"' not in HTML
    assert 'aria-label="品类"' not in HTML
    assert "product_category" not in APP_JS


def test_public_upload_root_is_loaded_before_batch_selection():
    assert "/api/workspace-config" in APP_JS
    assert "/api/upload-folders" in APP_JS
    assert 'id="folderPickerModal"' in HTML
    assert 'id="rootPath" autocomplete="off" readonly' in HTML
    assert "先将文件包上传到公共盘" in HTML
    assert "请在公共盘路径后填写具体文件包子目录" in HTML


def test_review_defaults_to_visual_all_groups_overview():
    assert 'filter: "all"' in APP_JS
    assert 'class="category-check visual-category main-category"' in APP_JS
    assert 'class="category-check visual-category ${status.className}"' in APP_JS
    assert "slot.name.replace" not in APP_JS
    assert "${esc(slot.name)}" in APP_JS
    assert "group.categories.flatMap" in APP_JS
    assert "slot.files.map((file, index)" in APP_JS
    assert "命名冲突" in APP_JS
    assert "按主图聚合显示关联图片和 PSD" in HTML


def test_review_keeps_pending_filter_available():
    assert 'pending: "待确认"' in APP_JS


def test_reassign_loads_all_teams_and_uses_large_visual_grid():
    assert "DINO Top5 候选" in HTML
    assert "其他未审核通过的球队" in HTML
    assert "main-images/search?q=&exclude_confirmed=true" in APP_JS
    assert "exclude_confirmed=true" in APP_JS
    assert "result.swapped_file_id" in APP_JS
    assert "repeat(4,minmax(220px,1fr))" in CSS
    assert "reassign-split" in HTML
    assert '$("#reassignImage").src = file.preview_url' in APP_JS


def test_completed_batch_can_rerun_dino_safely():
    assert 'id="resetMaskForRematch"' in HTML
    assert "重新选择遮罩与置信度" in HTML
    assert "已整组确认和人工调换的结果会保留" in APP_JS
    assert '$("#matchBtn").classList.remove("hidden")' in APP_JS


def test_technical_information_is_collapsed_by_default():
    assert HTML.count('class="advanced-details"') >= 4
    assert '<details open' not in HTML
    assert "模型与处理详情" in HTML


def test_confidence_controls_are_posted_with_each_match():
    assert 'id="unmatchedScore"' in HTML
    assert 'id="matchMinScore"' in HTML
    assert 'id="matchMinMargin"' in HTML
    assert "JSON.stringify(matchingConfig)" in APP_JS
    assert "自动归属最低分不能高于低置信度分数" in APP_JS


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


def test_extra_files_do_not_block_batch_rename():
    assert 'IGNORED: "多余文件（保持原名）"' in APP_JS
    assert '["READY", "NOOP", "RENAMED", "IGNORED"]' in APP_JS
    assert '["READY", "NOOP", "RENAMED", "IGNORED"]' in STATE_JS.read_text(encoding="utf-8")
