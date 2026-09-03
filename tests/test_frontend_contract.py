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
    assert "按同名主图直接关联，不参与匹配计算" in APP_JS
    assert "PSD 同名直接关联" in APP_JS
    assert "PSD Preview pHash" not in APP_JS


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
    assert "默认展示全部商品组" in HTML


def test_review_keeps_pending_filter_available():
    assert 'pending: "待确认"' in APP_JS


def test_reassign_loads_all_teams_and_uses_large_visual_grid():
    assert "未审核通过的球队（点击即可调换）" in HTML
    assert "main-images/search?q=&exclude_confirmed=true" in APP_JS
    assert "exclude_confirmed=true" in APP_JS
    assert "result.swapped_file_id" in APP_JS
    assert "repeat(4,minmax(220px,1fr))" in CSS
    assert "reassign-split" in HTML
    assert '$("#reassignImage").src = file.preview_url' in APP_JS


def test_completed_batch_can_return_to_masks_and_rematch_safely():
    assert 'id="resetMaskForRematch"' in HTML
    assert "使用当前识别区域重新匹配" in APP_JS
    assert "已整组确认和人工调换的结果会保留" in APP_JS
    assert '$("#matchBtn").classList.remove("hidden")' in APP_JS


def test_technical_information_is_collapsed_by_default():
    assert HTML.count('class="advanced-details"') >= 4
    assert '<details open' not in HTML
    assert "高级设置与处理详情" in HTML


def test_main_images_share_one_category_mask_applied_to_the_batch():
    assert 'id="maskEditor"' in HTML
    assert 'id="maskCanvas"' in HTML
    assert "设置一次，按比例应用到全部主图" in APP_JS
    assert "保存并批量应用" in APP_JS
    assert "reference_main_id" in APP_JS
    assert "逐张裁剪" not in APP_JS
    assert 'id="mainBodyRoiModal"' not in HTML
    assert "自动去除白底" not in HTML
    assert "mask_data_url" in APP_JS


def test_matching_uses_only_manual_masks_and_hides_generated_ai_view():
    assert "附图手动遮罩 vs 主图手动遮罩" in APP_JS
    assert "手动遮罩 pHash" in APP_JS
    assert "phash_best_region_type" in APP_JS
    assert "pHash Hamming Distance" in APP_JS
    assert "AI 实际识别画面" not in HTML
    assert "AI 识别画面" not in APP_JS
    assert "buildAiPreview" not in APP_JS


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
