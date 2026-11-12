const state = {
  batchId: null,
  batch: null,
  workspace: deriveWorkspaceState(null),
  filter: "all",
  reviewData: null,
  renamePlan: null,
  reviewSaving: false,
  renameSaving: false,
  conflictMarkSaving: false,
  selectedMains: new Set(),
  reviewFile: null,
  reassignFile: null,
  reassignCandidates: [],
  poll: null,
  maskData: null,
  maskProfile: null,
  maskCategory: null,
  maskHistory: [],
  maskRedo: [],
  maskMode: "crop",
  cropBox: null,
  cropStart: null,
  folderPicker: { currentPath: null, parentPath: null, isRoot: true },
  lightbox: { scale: 1, x: 0, y: 0, dragging: false, startX: 0, startY: 0 },
};

const $ = selector => document.querySelector(selector);
const $$ = selector => document.querySelectorAll(selector);
const esc = value => String(value ?? "").replace(/[&<>'"]/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[char]);
const pct = value => value == null ? "—" : `${(value * 100).toFixed(1)}%`;

async function api(url, options = {}) {
  const reviewWrite = options.method?.toUpperCase() === "POST" &&
    /\/(?:confirm|confirm-groups|unconfirm|unconfirm-groups|reassign|cancel-match|restore|review-categories|rename-conflict-marks)$/.test(url);
  if (reviewWrite && state.reviewSaving) throw new Error("正在保存审核，请等待完成后再操作");
  if (reviewWrite) state.reviewSaving = true;
  try {
    const response = await fetch(url, { headers: { "Content-Type": "application/json", ...(options.headers || {}) }, ...options });
    // A Response body is a stream: read it once, then parse the saved text.
    const body = await response.text();
    let payload;
    try { payload = body.trim() ? JSON.parse(body) : null; }
    catch {
      if (response.ok) throw new Error("服务返回的数据格式异常，请刷新页面后重试");
    }
    if (!response.ok) {
      let detail = payload?.detail;
      if (Array.isArray(detail)) detail = detail.map(item => item.msg || "参数无效").join("；");
      const plainText = !/^\s*</.test(body) ? body.trim().slice(0, 500) : "";
      const message = typeof detail === "string" && detail ? detail :
        (plainText && plainText !== "Internal Server Error" ? plainText :
          `服务请求失败（HTTP ${response.status}），请稍后刷新核对状态后重试`);
      throw new Error(message);
    }
    return payload;
  } finally {
    if (reviewWrite) state.reviewSaving = false;
  }
}

function toast(message, bad = false) {
  const element = $("#toast");
  element.textContent = message;
  element.style.background = bad ? "#8d302a" : "#16312e";
  element.classList.remove("hidden");
  setTimeout(() => element.classList.add("hidden"), 3000);
}

function hashState() {
  const params = new URLSearchParams(location.hash.replace(/^#/, ""));
  return { batchId: Number(params.get("batch")) || null, view: params.get("view") || "create" };
}

function updateHash(view) {
  const params = new URLSearchParams();
  if (state.batchId) params.set("batch", state.batchId);
  params.set("view", view);
  history.replaceState(null, "", `#${params}`);
}

const viewStep = { create: "import", mask: "regions", process: "match", review: "review" };
const stepStatusText = { complete: "✓ 已完成", current: "● 当前步骤", upcoming: "○ 未开始", needs: "! 需要处理" };

function renderStepper(activeView) {
  $$(".nav-item").forEach(button => {
    const key = button.dataset.step;
    const status = state.workspace.steps[key] || "upcoming";
    button.dataset.status = status;
    button.classList.toggle("active", button.dataset.view === activeView);
    button.classList.toggle("locked", !state.workspace.canEnter[button.dataset.view]);
    button.querySelector(".step-status").textContent = stepStatusText[status];
  });
}

function guardMessage(view) {
  if (view === "mask") return "请先导入图片";
  if (view === "process") return "请先导入图片";
  if (view === "review") return "请先完成自动匹配";
  return "请先完成上一步";
}

function showView(name, force = false) {
  if (!force && !state.workspace.canEnter[name]) {
    toast(guardMessage(name), true);
    return;
  }
  $$(".view").forEach(view => view.classList.toggle("active", view.id === `view-${name}`));
  renderStepper(name);
  updateHash(name);
  if (name === "mask" && state.batchId) loadMasks();
  if (name === "review" && state.batchId) loadReview();
  if (name === "process") renderProcess();
}

$$(".nav-item").forEach(button => button.onclick = () => showView(button.dataset.view));

async function refreshWorkspace(includeReview = true) {
  if (!state.batchId) return;
  state.batch = await api(`/api/batches/${state.batchId}`);
  if (includeReview && state.batch.status === "COMPLETED") {
    const [review, plan] = await Promise.all([
      api(`/api/batches/${state.batchId}/review?filter=all`),
      api(`/api/batches/${state.batchId}/rename-plan`),
    ]);
    state.reviewData = review;
    state.renamePlan = plan;
  }
  state.workspace = deriveWorkspaceState(state.batch, state.reviewData || {}, state.renamePlan || {});
  renderWorkspaceHome();
  renderImportSummary();
  renderProcess();
  renderStepper(viewStep[$(".nav-item.active")?.dataset.view] ? $(".nav-item.active").dataset.view : "create");
}

const batchStatusNames = { CREATED: "等待导入", SCANNED: "识别区域待设置", QUEUED: "等待匹配", MATCHING: "正在自动匹配", COMPLETED: "匹配完成", FAILED: "需要重新处理" };
function batchStatusLabel(batch) {
  if (batch?.error_message?.includes("服务重启中断")) return "上次中断，可重新开始";
  if (batch?.status === "MATCHING" && batch.progress?.percent != null) return `正在匹配 ${batch.progress.percent}%`;
  return batchStatusNames[batch?.status] || batch?.status || "未知状态";
}

async function loadBatches() {
  const rows = await api("/api/batches");
  const list = $("#batchList");
  list.classList.toggle("empty", !rows.length);
  list.innerHTML = rows.length ? rows.map(batch => `
    <div class="batch-row">
      <div class="batch-root"><b>批次 #${batch.id}</b><small>${batch.main_count || 0} 个商品 · ${esc(batch.match_engine || "PHASH")}</small></div>
      <span class="status-pill">${esc(batchStatusLabel(batch))}</span>
      <button class="secondary" onclick="selectBatch(${batch.id})">打开</button>
    </div>`).join("") : "暂无批次";
  const requested = hashState();
  if (!state.batchId && requested.batchId && rows.some(row => row.id === requested.batchId)) {
    await selectBatch(requested.batchId, requested.view);
  }
}

window.selectBatch = async (id, requestedView = "create") => {
  state.batchId = id;
  state.reviewData = null;
  state.renamePlan = null;
  state.maskData = null;
  state.maskProfile = null;
  state.maskCategory = null;
  state.selectedMains.clear();
  state.filter = "all";
  await refreshWorkspace(true);
  const safeView = state.workspace.canEnter[requestedView] ? requestedView : "create";
  showView(safeView, true);
  if (["QUEUED", "MATCHING"].includes(state.batch.status)) startPolling();
};

function renderWorkspaceHome() {
  const batch = state.batch;
  $("#workspaceHome").classList.toggle("hidden", !batch);
  if (!batch) return;
  const totals = state.workspace.totals;
  $("#workspaceBatchTitle").textContent = `批次 #${batch.id}`;
  $("#workspaceProductSummary").textContent = `${batch.main_count || 0} 个商品`;
  $("#workspaceBatchStatus").textContent = batchStatusNames[batch.status] || batch.status;
  $("#homeImportIcon").textContent = totals.files ? "✓" : "●";
  $("#homeImportStatus").textContent = totals.files ? `${totals.files} 个文件` : "等待导入";
  $("#homeRegionIcon").textContent = state.workspace.steps.regions === "complete" ? "✓" : state.workspace.steps.regions === "current" ? "●" : "○";
  $("#homeRegionStatus").textContent = `${totals.regionsDone} / ${totals.regionsTotal} 已完成`;
  $("#homeMatchIcon").textContent = state.workspace.steps.match === "complete" ? "✓" : state.workspace.steps.match === "needs" ? "!" : state.workspace.steps.match === "current" ? "●" : "○";
  $("#homeMatchStatus").textContent = state.workspace.matched ? `${Math.max(0, totals.files - totals.unmatched)} 已匹配 · ${totals.unmatched} 未匹配` : state.workspace.matching ? "正在处理" : "尚未开始";
  $("#homeReviewIcon").textContent = state.workspace.steps.review === "needs" ? "!" : state.workspace.matched && !state.workspace.needsReview ? "✓" : "○";
  $("#homeReviewStatus").textContent = !state.workspace.matched ? "等待自动匹配" : state.workspace.needsReview ? `还需处理 ${totals.pending + totals.unmatched} 项` : "审核完成，可以预览命名";
  $("#continueBtn").textContent = state.workspace.nextLabel;
  const config = batch.matching_config || {};
  $("#workspaceAdvanced").innerHTML = `<p><b>处理设备：</b>${esc(batch.progress?.device || "将在匹配时自动选择")}</p><p><b>算法版本：</b>${esc(config.algorithm_version || "—")}</p><p><b>PSD：</b>读取合成预览匹配，确认后只改PSD文件名</p><p><b>当前目录：</b><code>${esc(batch.root_path)}</code></p>`;
}

$("#continueBtn").onclick = () => showView(state.workspace.nextView);

function renderImportSummary() {
  const batch = state.batch;
  const visible = batch && state.workspace.totals.files > 0;
  $("#currentImportSummary").classList.toggle("hidden", !visible);
  if (!visible) return;
  const rows = [["主图", batch.main_count || 0], ...Object.entries(batch.categories || {})];
  $("#importCounts").innerHTML = rows.map(([name, count]) => `<div><span>${esc(name)}</span><b>${count}</b><i>✓</i></div>`).join("");
  const associations = batch.association_counts || {};
  $("#importAnomalies").textContent = `✓ 文件名直接关联 ${associations.exact_filename || 0} · 视觉匹配 ${associations.visual || 0} · 待匹配 ${associations.pending || 0} · 匹配冲突 ${associations.conflict || 0}`;
  $("#importRootPath").textContent = batch.root_path;
}

$("#nextToRegions").onclick = () => showView("mask");
$("#refreshBatches").onclick = loadBatches;

async function importPublicFolder(root) {
  if (!root) return toast("请选择公共盘文件包", true);
  try {
    $("#folderPickerConfirm").disabled = true;
    $("#folderPickerConfirm").textContent = "正在导入…";
    const batch = await api("/api/batches", { method: "POST", body: JSON.stringify({
      root_path: root,
    }) });
    state.batchId = batch.id;
    await api(`/api/batches/${batch.id}/scan`, { method: "POST" });
    await refreshWorkspace(false);
    await loadBatches();
    $("#folderPickerModal").classList.add("hidden");
    showView("create", true);
    toast("图片导入完成");
  } catch (error) {
    toast(error.message, true);
  } finally {
    $("#folderPickerConfirm").textContent = "选择并导入";
    $("#folderPickerConfirm").disabled = state.folderPicker.isRoot;
  }
}

async function openUploadFolder(path = null) {
  try {
    const query = path ? `?path=${encodeURIComponent(path)}` : "";
    const data = await api(`/api/upload-folders${query}`);
    state.folderPicker = {
      currentPath: data.current_path,
      parentPath: data.parent_path,
      isRoot: data.is_root,
    };
    $("#rootPath").value = data.current_path;
    $("#folderPickerPath").textContent = data.current_path;
    $("#folderPickerBack").disabled = data.is_root;
    $("#folderPickerConfirm").disabled = data.is_root;
    $("#folderPickerList").innerHTML = data.folders.length
      ? data.folders.map((folder, index) => `<button class="folder-picker-row" data-folder-index="${index}"><span>📁</span><b>${esc(folder.name)}</b></button>`).join("")
      : '<div class="folder-picker-empty">当前文件夹内没有子文件夹，可以直接选择当前文件夹。</div>';
    $$('[data-folder-index]').forEach(button => {
      button.onclick = () => openUploadFolder(data.folders[Number(button.dataset.folderIndex)].path);
    });
    $("#folderPickerModal").classList.remove("hidden");
  } catch (error) {
    toast(error.message, true);
  }
}

$("#createBtn").onclick = () => openUploadFolder();
$("#rootPath").onclick = () => openUploadFolder();
$("#folderPickerBack").onclick = () => {
  if (state.folderPicker.parentPath) openUploadFolder(state.folderPicker.parentPath);
};
$("#folderPickerConfirm").onclick = () => importPublicFolder(state.folderPicker.currentPath);

function renderProcess() {
  const batch = state.batch;
  if (!batch) return;
  $("#processRoot").textContent = `${batch.main_count || 0} 个商品`;
  const progress = batch.progress || { percent: 0, current: 0, total: 0, message: "等待开始" };
  const percent = Math.max(0, Math.min(100, Number(progress.percent) || 0));
  const title = batch.status === "MATCHING" ? "正在自动匹配" : batch.status === "QUEUED" ? "正在准备" : batch.status === "COMPLETED" ? "匹配完成" : batch.status === "FAILED" ? "处理未完成" : "准备完成";
  $("#processStatus h2").textContent = title;
  $("#processStatus .status-icon").textContent = batch.status === "COMPLETED" ? "✓" : batch.status === "FAILED" ? "!" : ["MATCHING", "QUEUED"].includes(batch.status) ? "↻" : "●";
  $("#progressPercent").textContent = `${percent.toFixed(percent % 1 ? 1 : 0)}%`;
  $("#progressBar").style.width = `${percent}%`;
  $("#progressMessage").textContent = progress.message || "等待开始";
  $("#progressCount").textContent = progress.total ? `${progress.current} / ${progress.total}` : "";
  $("#mainCount").textContent = batch.main_count || 0;
  $("#fileCount").textContent = Object.values(batch.categories || {}).reduce((sum, count) => sum + Number(count || 0), 0);
  $("#regionReadyCount").textContent = `${state.workspace.totals.regionsDone} / ${state.workspace.totals.regionsTotal}`;
  const busy = ["QUEUED", "MATCHING"].includes(batch.status);
  const terra = batch.matching_config?.available_engines?.find(engine => engine.id === "VISIONMATCH");
  $$("[data-visionmatch-control]").forEach(button => { button.disabled = !busy; });
  const matchButton = $("#matchBtn");
  matchButton.disabled = !["SCANNED", "COMPLETED", "FAILED"].includes(batch.status);
  matchButton.classList.remove("hidden");
  matchButton.textContent = batch.status === "COMPLETED" ? "重新匹配" : "开始匹配";
  const error = batch.error_message || terra?.error;
  $("#processError").classList.toggle("hidden", !error);
  $("#processError").textContent = error || "";
  $("#technicalConfig").textContent = progress.device || terra?.device || "";
  const layout = batch.layout || {};
  $("#layoutInfo").classList.toggle("hidden", !batch.layout);
  $("#layoutInfo").innerHTML = batch.layout ? `<div><b>图片目录</b><code>${esc(layout.effect_root || batch.root_path)}</code></div><div><b>主图目录</b><code>${esc(layout.main_directory || "—")}</code></div>` : "";
  $("#categoryStats").innerHTML = Object.entries(batch.categories || {}).map(([name, count]) => `<div class="category-stat"><span>${esc(name)}</span><b>${count}</b></div>`).join("");
  renderMatchResult();
}

function renderMatchResult() {
  const complete = state.batch?.status === "COMPLETED" && state.reviewData;
  $("#matchResultOverview").classList.toggle("hidden", !complete);
  if (!complete) return;
  const summary = state.reviewData.summary || {};
  const manualSelectionCount = Number(summary.manual_selection_files ?? summary.unmatched_files ?? 0);
  $("#resultMatched").textContent = Math.max(0, state.workspace.totals.files - manualSelectionCount);
  $("#resultPending").textContent = summary.pending || 0;
  $("#resultUnmatched").textContent = manualSelectionCount;
}

$$('[data-visionmatch-control]').forEach(button => button.onclick = async () => {
  try {
    await api(`/api/batches/${state.batchId}/visionmatch-control`, {method: "POST", body: JSON.stringify({action: button.dataset.visionmatchControl})});
    toast("操作已提交；已发出的请求不会被撤销");
  } catch (error) { toast(error.message, true); }
});

$("#reviewPendingBtn").onclick = () => { state.filter = "pending"; showView("review"); };
$("#viewAllResultsBtn").onclick = () => { state.filter = "all"; showView("review"); };

$("#matchBtn").onclick = async () => {
  try {
    const folderEngines = Object.fromEntries(
      Object.keys(state.batch.categories || {}).map(category => [category, { engine: "VISIONMATCH" }])
    );
    const payload = { match_engine: "VISIONMATCH", folder_engines: folderEngines };
    const plan = await api(`/api/batches/${state.batchId}/visionmatch-plan`, { method: "POST", body: JSON.stringify(payload) });
    const calls = plan.calls_upper_bound;
    if (calls && !confirm(`VisionMatch AI · Terra 本次最多调用 ${calls} 次，确认开始？`)) return;
    await api(`/api/batches/${state.batchId}/match`, {
      method: "POST",
      body: JSON.stringify({ ...payload, visionmatch_confirmed_budget: Boolean(calls), visionmatch_max_calls: calls }),
    });
    await refreshWorkspace(false);
    startPolling();
    toast("自动匹配已开始");
  } catch (error) { toast(error.message, true); }
};

function startPolling() {
  clearInterval(state.poll);
  state.poll = setInterval(async () => {
    await refreshWorkspace(false);
    if (!["QUEUED", "MATCHING"].includes(state.batch.status)) {
      clearInterval(state.poll);
      state.poll = null;
      await refreshWorkspace(true);
      if (state.batch.status === "COMPLETED") toast("自动匹配完成");
      else toast("自动匹配未完成，请查看处理详情", true);
    }
  }, 1200);
}

function renderMaskProfileList(data) {
  const active = new Map(data.profiles.filter(item => item.enabled && item.creation_method === "MANUAL" && item.storage_valid !== false).map(item => [item.image_category, item]));
  $("#maskProgressText").textContent = `${data.readiness.configured_categories.length} / ${data.readiness.required_categories.length} 已完成`;
  $("#nextToMatch").classList.toggle("hidden", !data.readiness.ready);
  $("#maskProfileList").innerHTML = data.categories.map(category => {
    if (category === "PSD") return `<div class="mask-profile-row disabled"><div><b>PSD</b><small class="hidden helper-note">使用合成预览参与匹配，无需遮罩</small></div><span>✓</span></div>`;
    const profile = active.get(category);
    const complete = Array.isArray(profile?.crop_box);
    const completeText = category === "主图" ? "✓ 已批量应用" : "✓ 已完成";
    if (complete) return `<div class="mask-profile-row ${state.maskCategory === category ? "active" : ""}"><button onclick="openMaskCategory('${esc(category)}')"><div><b>${esc(category)}</b><small>${completeText}</small></div><span>重新调整</span></button></div>`;
    if (profile) return `<div class="mask-profile-row needs"><button onclick="openMaskCategory('${esc(category)}')"><div><b>${esc(category)}</b><small>! 该旧设置需要补充识别范围</small></div><span>补充识别范围</span></button></div>`;
    const pendingText = category === "主图" ? "! 设置一次，批量应用" : "! 待设置";
    return `<div class="mask-profile-row needs ${state.maskCategory === category ? "active" : ""}"><button onclick="openMaskCategory('${esc(category)}')"><div><b>${esc(category)}</b><small>${pendingText}</small></div><span>去设置</span></button></div>`;
  }).join("");
}

async function loadMasks() {
  if (!state.batchId) return;
  try {
    const data = await api(`/api/batches/${state.batchId}/mask-profiles`);
    state.maskData = data;
    renderMaskProfileList(data);
    if (!state.maskCategory) {
      const first = data.readiness.missing_categories[0];
      if (first) await openMaskCategory(first);
    }
  } catch (error) { toast(error.message, true); }
}

window.openMaskCategory = async category => {
  state.maskCategory = category;
  let profile = (state.maskData?.profiles || []).find(item => item.image_category === category && item.enabled && item.creation_method === "MANUAL" && item.storage_valid !== false);
  try {
    const files = await api(`/api/batches/${state.batchId}/category-files/${encodeURIComponent(category)}`);
    if (!files.length) throw new Error("这个分类没有可用图片");
    const isMain = category === "主图";
    if (!profile) profile = { name: category, version: 0, reference_file_id: isMain ? null : files[0].id, reference_main_id: isMain ? files[0].id : null, mask_url: null, crop_box: null, draft: true };
    const referenceId = isMain ? profile.reference_main_id : profile.reference_file_id;
    state.maskProfile = profile;
    state.maskHistory = [];
    state.maskRedo = [];
    setMaskMode("crop");
    $("#fineTunePanel").classList.add("hidden");
    $("#maskEmpty").classList.add("hidden");
    $("#maskEditor").classList.remove("hidden");
    $("#maskEditorTitle").textContent = category;
    $("#maskVersion").textContent = profile.draft ? isMain ? "设置一次，按比例应用到全部主图" : "尚未设置" : profile.crop_box ? isMain ? "已批量应用到全部主图" : "已设置" : "该旧设置需要补充识别范围";
    $("#maskSave").textContent = isMain ? "保存并批量应用" : "保存并下一类";
    $("#maskReference").innerHTML = files.map(file => `<option value="${file.id}" data-url="${file.preview_url}" ${file.id === referenceId ? "selected" : ""}>${esc(file.filename)}</option>`).join("");
    await renderMaskCanvas($("#maskReference").selectedOptions[0]?.dataset.url, profile.mask_url, profile.crop_box, isMain && profile.draft);
    renderMaskProfileList(state.maskData);
  } catch (error) {
    $("#maskEditor").classList.add("hidden");
    $("#maskEmpty").classList.remove("hidden");
    $("#maskEmpty").innerHTML = "<b>参考图片无法打开</b><p>图片可能已移动，请返回导入页重新扫描当前文件夹。</p>";
    toast(error.message === "参考图片无法打开" ? "参考图片已移动，请重新扫描当前文件夹" : error.message, true);
  }
};

function loadImage(url) { return new Promise((resolve, reject) => { const image = new Image(); image.onload = () => resolve(image); image.onerror = () => reject(new Error("参考图片无法打开")); image.src = url; }); }

function fitMaskCanvasToPane(reference = $("#maskReferenceImage")) {
  if (!reference?.naturalWidth || !reference?.naturalHeight) return;
  const pane = $(".mask-edit-pane");
  const wrap = $("#maskCanvasWrap");
  const availableWidth = Math.max(240, pane.clientWidth);
  const availableHeight = Math.max(240, window.innerHeight * .65);
  const scale = Math.min(1, availableWidth / reference.naturalWidth, availableHeight / reference.naturalHeight);
  wrap.style.width = `${Math.max(1, Math.round(reference.naturalWidth * scale))}px`;
  wrap.style.height = `${Math.max(1, Math.round(reference.naturalHeight * scale))}px`;
}

async function renderMaskCanvas(referenceUrl, maskUrl, cropBox, defaultWhite = false) {
  if (!referenceUrl) return;
  const [reference, mask] = await Promise.all([loadImage(referenceUrl), maskUrl ? loadImage(maskUrl) : Promise.resolve(null)]);
  const image = $("#maskReferenceImage");
  const canvas = $("#maskCanvas");
  const context = canvas.getContext("2d", { willReadFrequently: true });
  image.src = referenceUrl;
  canvas.width = reference.naturalWidth;
  canvas.height = reference.naturalHeight;
  fitMaskCanvasToPane(reference);
  context.imageSmoothingEnabled = false;
  context.fillStyle = defaultWhite ? "white" : "black";
  context.fillRect(0, 0, canvas.width, canvas.height);
  if (mask) context.drawImage(mask, 0, 0, canvas.width, canvas.height);
  state.maskHistory = [context.getImageData(0, 0, canvas.width, canvas.height)];
  state.maskRedo = [];
  state.cropBox = Array.isArray(cropBox) ? [...cropBox] : null;
  updateCropOverlay();
}

let maskPreviewFrame = null;
function scheduleRecognitionPreview() {
  if (maskPreviewFrame !== null) cancelAnimationFrame(maskPreviewFrame);
  maskPreviewFrame = requestAnimationFrame(() => {
    maskPreviewFrame = null;
    renderRecognitionPreview();
  });
}

function renderRecognitionPreview() {
  const output = $("#maskResultCanvas");
  const empty = $("#maskResultEmpty");
  const info = $("#maskResultInfo");
  const source = $("#maskReferenceImage");
  const mask = $("#maskCanvas");
  if (!state.cropBox || !source.complete || !source.naturalWidth || !mask.width) {
    output.classList.add("hidden");
    empty.classList.remove("hidden");
    info.textContent = "黑色遮罩区域会变为灰白色，不参与识别。";
    return;
  }

  const [left, top, right, bottom] = state.cropBox;
  const sx = Math.max(0, Math.round(left * mask.width));
  const sy = Math.max(0, Math.round(top * mask.height));
  const sourceWidth = Math.max(1, Math.round((right - left) * mask.width));
  const sourceHeight = Math.max(1, Math.round((bottom - top) * mask.height));
  const scale = Math.min(1, 720 / sourceWidth, 560 / sourceHeight);
  const width = Math.max(1, Math.round(sourceWidth * scale));
  const height = Math.max(1, Math.round(sourceHeight * scale));

  output.width = width;
  output.height = height;
  const context = output.getContext("2d", { willReadFrequently: true });
  context.fillStyle = "rgb(245, 245, 245)";
  context.fillRect(0, 0, width, height);
  context.drawImage(source, sx, sy, sourceWidth, sourceHeight, 0, 0, width, height);

  const scaledMask = document.createElement("canvas");
  scaledMask.width = width;
  scaledMask.height = height;
  const maskContext = scaledMask.getContext("2d", { willReadFrequently: true });
  maskContext.imageSmoothingEnabled = false;
  maskContext.drawImage(mask, sx, sy, sourceWidth, sourceHeight, 0, 0, width, height);
  const maskPixels = maskContext.getImageData(0, 0, width, height).data;
  const preview = context.getImageData(0, 0, width, height);
  let keptPixels = 0;
  for (let index = 0; index < preview.data.length; index += 4) {
    if (maskPixels[index] < 128) {
      preview.data[index] = 245;
      preview.data[index + 1] = 245;
      preview.data[index + 2] = 245;
      preview.data[index + 3] = 255;
    } else keptPixels += 1;
  }
  context.putImageData(preview, 0, 0);
  const coverage = Math.round(keptPixels / (width * height) * 100);
  info.textContent = `裁剪 ${sourceWidth} × ${sourceHeight}px · 有效识别区域约 ${coverage}% · 灰白色部分不参与匹配`;
  empty.classList.add("hidden");
  output.classList.remove("hidden");
}

function pushMaskHistory() {
  const canvas = $("#maskCanvas");
  state.maskHistory.push(canvas.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, canvas.width, canvas.height));
  if (state.maskHistory.length > 20) state.maskHistory.shift();
  state.maskRedo = [];
  scheduleRecognitionPreview();
}

function restoreMask(snapshot) { $("#maskCanvas").getContext("2d").putImageData(snapshot, 0, 0); scheduleRecognitionPreview(); }

window.addEventListener("resize", () => {
  fitMaskCanvasToPane();
  scheduleRecognitionPreview();
});

function canvasPoint(event) {
  const canvas = $("#maskCanvas");
  const rect = canvas.getBoundingClientRect();
  const nx = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
  const ny = Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height));
  return { x: nx * canvas.width, y: ny * canvas.height, nx, ny };
}

function updateCropOverlay() {
  const overlay = $("#cropBoxOverlay");
  const box = state.cropBox;
  const canvas = $("#maskCanvas");
  if (!box) {
    overlay.classList.add("hidden");
    $("#cropBoxInfo").textContent = "尚未框选识别范围";
    scheduleRecognitionPreview();
    return;
  }
  const [left, top, right, bottom] = box;
  overlay.classList.remove("hidden");
  overlay.style.left = `${left * 100}%`;
  overlay.style.top = `${top * 100}%`;
  overlay.style.width = `${(right - left) * 100}%`;
  overlay.style.height = `${(bottom - top) * 100}%`;
  const width = Math.round((right - left) * canvas.width);
  const height = Math.round((bottom - top) * canvas.height);
  $("#cropBoxSize").textContent = `${width} × ${height}`;
  $("#cropBoxInfo").textContent = `已框选匹配区域 · ${width} × ${height}px`;
  scheduleRecognitionPreview();
}

function setMaskMode(mode) {
  state.maskMode = mode;
  $$("[data-mask-mode]").forEach(button => button.classList.toggle("active", button.dataset.maskMode === mode));
  $("#maskCanvasWrap").classList.toggle("crop-mode", mode === "crop");
  $("#maskCanvasWrap").classList.toggle("paint-mode", mode !== "crop");
  if (mode === "add") $("#currentMaskTool").textContent = "补充识别区域";
  if (mode === "erase") $("#currentMaskTool").textContent = "排除识别区域";
}

let drawing = false;
function paintMask(event) {
  if (!drawing || state.maskMode === "crop") return;
  const canvas = $("#maskCanvas");
  const context = canvas.getContext("2d");
  const point = canvasPoint(event);
  context.fillStyle = state.maskMode === "add" ? "white" : "black";
  context.beginPath();
  context.arc(point.x, point.y, Number($("#maskBrush").value) / 2, 0, Math.PI * 2);
  context.fill();
  scheduleRecognitionPreview();
}

function updateCropFromEvent(event) {
  if (!state.cropStart) return;
  const point = canvasPoint(event);
  state.cropBox = [Math.min(state.cropStart.nx, point.nx), Math.min(state.cropStart.ny, point.ny), Math.max(state.cropStart.nx, point.nx), Math.max(state.cropStart.ny, point.ny)];
  updateCropOverlay();
}

function applyCropToMask(showMessage = false) {
  if (!state.cropBox) return;
  const canvas = $("#maskCanvas");
  const context = canvas.getContext("2d");
  const [left, top, right, bottom] = state.cropBox;
  context.fillStyle = "black";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "white";
  context.fillRect(
    Math.round(left * canvas.width),
    Math.round(top * canvas.height),
    Math.round((right - left) * canvas.width),
    Math.round((bottom - top) * canvas.height),
  );
  pushMaskHistory();
  if (showMessage) toast("已应用当前框选区域，可直接保存或精细调整");
}

$("#maskCanvas").onpointerdown = event => {
  $("#maskCanvas").setPointerCapture(event.pointerId);
  if (state.maskMode === "crop") {
    const point = canvasPoint(event);
    state.cropStart = { nx: point.nx, ny: point.ny };
    state.cropBox = [point.nx, point.ny, point.nx, point.ny];
    updateCropOverlay();
  } else { drawing = true; paintMask(event); }
};
$("#maskCanvas").onpointermove = event => state.maskMode === "crop" ? updateCropFromEvent(event) : paintMask(event);
$("#maskCanvas").onpointerup = event => {
  if (state.maskMode === "crop" && state.cropStart) {
    updateCropFromEvent(event);
    state.cropStart = null;
    const [left, top, right, bottom] = state.cropBox;
    if (right - left < .02 || bottom - top < .02) { state.cropBox = null; updateCropOverlay(); toast("框选范围太小，请重新拖动", true); }
    else applyCropToMask();
  } else if (drawing) { drawing = false; pushMaskHistory(); }
};

$$("[data-mask-mode]").forEach(button => button.onclick = () => setMaskMode(button.dataset.maskMode));
$("#maskFineTuneToggle").onclick = () => { $("#fineTunePanel").classList.remove("hidden"); setMaskMode("add"); };
$("#maskFineTuneClose").onclick = () => { $("#fineTunePanel").classList.add("hidden"); setMaskMode("crop"); };
$("#maskFillCrop").onclick = () => {
  if (!state.cropBox) return toast("请先在图片上拖出识别范围", true);
  applyCropToMask(true);
};
$("#maskClearCrop").onclick = () => { state.cropBox = null; updateCropOverlay(); setMaskMode("crop"); };
$("#maskUndo").onclick = () => { if (state.maskHistory.length < 2) return; state.maskRedo.push(state.maskHistory.pop()); restoreMask(state.maskHistory.at(-1)); };
$("#maskRedo").onclick = () => { const next = state.maskRedo.pop(); if (!next) return; state.maskHistory.push(next); restoreMask(next); };
$("#maskClear").onclick = () => { const canvas = $("#maskCanvas"); const context = canvas.getContext("2d"); context.fillStyle = "black"; context.fillRect(0, 0, canvas.width, canvas.height); pushMaskHistory(); };
$("#maskPreviewToggle").onclick = () => { const canvas = $("#maskCanvas"); const hidden = canvas.style.opacity === "0"; canvas.style.opacity = hidden ? ".48" : "0"; $("#maskPreviewToggle").textContent = hidden ? "隐藏识别区域" : "查看识别区域"; };

$("#maskReference").onchange = async () => renderMaskCanvas($("#maskReference").selectedOptions[0].dataset.url, state.maskProfile.draft ? null : state.maskProfile.mask_url, state.maskProfile.draft ? null : state.maskProfile.crop_box, state.maskCategory === "主图" && state.maskProfile.draft);
$("#maskSave").onclick = async () => {
  const button = $("#maskSave");
  const category = state.maskCategory;
  const startedAt = performance.now();
  try {
    if (!state.cropBox) throw new Error("请先框选识别范围");
    button.disabled = true;
    button.textContent = category === "主图" ? "正在批量应用…" : "正在保存…";
    const referenceId = Number($("#maskReference").value);
    const reference = category === "主图" ? { reference_main_id: referenceId } : { reference_file_id: referenceId };
    const profile = await api(`/api/batches/${state.batchId}/mask-profiles/${encodeURIComponent(category)}`, { method: "POST", body: JSON.stringify({ mask_data_url: $("#maskCanvas").toDataURL("image/png"), ...reference, crop_box: state.cropBox, name: state.maskProfile.name }) });
    state.maskProfile = profile;
    const elapsed = ((performance.now() - startedAt) / 1000).toFixed(1);
    toast(category === "主图" ? `主图遮罩已保存，匹配时将按比例应用到全部主图（${elapsed} 秒）` : state.batch?.status === "COMPLETED" ? "新识别区域已保存，可以进入自动匹配重新计算" : "识别区域已保存");
    await refreshWorkspace(false);
    const data = await api(`/api/batches/${state.batchId}/mask-profiles`);
    state.maskData = data;
    const next = data.readiness.missing_categories.find(category => category !== state.maskCategory);
    state.maskCategory = null;
    renderMaskProfileList(data);
    if (next) await openMaskCategory(next);
    else { $("#maskEditor").classList.add("hidden"); $("#maskEmpty").classList.remove("hidden"); $("#maskEmpty").textContent = "✓ 所有图片分类均已完成，可以进入自动匹配"; }
  } catch (error) { toast(error.message, true); }
  finally {
    button.disabled = false;
    button.textContent = state.maskCategory === "主图" ? "保存并批量应用" : "保存并下一类";
  }
};
$("#nextToMatch").onclick = () => showView("process");
$("#rematchBtn").onclick = () => showView("process");

const filterLabels = { pending: "待确认", unmatched: "未匹配 / 需人工选择", confirmed: "已完成", all: "全部" };
const userStatus = { MATCHED: "系统已匹配", PENDING: "待确认", AMBIGUOUS: "需要确认", LOW_CONFIDENCE: "需要确认", UNMATCHED: "未匹配", MANUAL_CONFIRMED: "已确认" };

async function loadReview() {
  if (!state.batchId) return;
  try {
    const [data, plan] = await Promise.all([api(`/api/batches/${state.batchId}/review?filter=${state.filter}`), api(`/api/batches/${state.batchId}/rename-plan`)]);
    state.reviewData = data;
    state.renamePlan = plan;
    state.workspace = deriveWorkspaceState(state.batch, data, plan);
    const engineLabels = { PHASH: "pHash", DINO: "DINOv2 ViT-B/14 GPU", LIGHTGLUE_COLOR: "LightGlue + Color", VISIONMATCH: "VisionMatch AI · Terra", QWEN: `AI视觉匹配（${state.batch?.match_config_snapshot?.model || state.batch?.matching_model || "Qwen-VL"}）` };
    const folderLabels = [...new Set(Object.values(state.batch?.matching_config?.folder_engines || {}).map(config => engineLabels[config.engine] || config.engine))];
    $("#reviewEngineBadge").textContent = `当前匹配算法：${state.batch?.match_engine === "FOLDER_SHARDED" ? folderLabels.join(" / ") : engineLabels[state.batch?.match_engine] || state.batch?.match_engine || "—"}`;
    const visible = new Set(data.groups.map(group => group.main.id));
    state.selectedMains = new Set([...state.selectedMains].filter(id => visible.has(id)));
    renderReviewOverview(data.summary);
    renderReviewCategories(data.review_categories || []);
    renderFilters(data.summary);
    renderUnmatched(data.manual_selection_files || data.unmatched_files || []);
    renderGroups(data.groups);
    renderBulkToolbar();
    renderRenamePlan(plan);
    renderStepper("review");
  } catch (error) { toast(error.message, true); }
}

function renderReviewOverview(summary) {
  $("#overviewAll").textContent = summary.all || 0;
  $("#overviewPending").textContent = summary.pending || 0;
  $("#overviewUnmatched").textContent = summary.manual_selection_files ?? summary.unmatched_files ?? 0;
  $("#overviewConfirmed").textContent = summary.confirmed || 0;
}

function renderReviewCategories(categories) {
  $("#reviewCategoryScope").innerHTML = categories.map(item => `<button class="secondary" data-review-category="${esc(item.category)}" data-ignored="${item.ignored}" ${state.reviewSaving || state.renameSaving ? "disabled" : ""}>${esc(item.category)} · ${item.count} 张 · ${item.ignored ? "已忽略，点击恢复" : "点击忽略"}</button>`).join("");
  $$('[data-review-category]').forEach(button => {
    button.onclick = async () => {
      button.disabled = true;
      try {
        const ignored = button.dataset.ignored !== "true";
        await api(`/api/batches/${state.batchId}/review-categories`, { method: "POST", body: JSON.stringify({ category: button.dataset.reviewCategory, ignored }) });
        toast(ignored ? "该类型已忽略，原图和匹配保留；其他项目可继续确认并改名" : "该类型已恢复，原匹配结果可继续审核");
        await loadReview();
      } catch (error) { toast(error.message, true); button.disabled = false; }
    };
  });
}

function renderFilters(summary) {
  const manualSelectionCount = summary.manual_selection_files ?? summary.unmatched_files ?? 0;
  $("#filters").innerHTML = Object.entries(filterLabels).map(([key, label]) => `<button class="filter-btn ${state.filter === key ? "active" : ""}" data-filter="${key}">${label}<b>${key === "unmatched" ? manualSelectionCount : summary[key] || 0}</b></button>`).join("");
  $$(".filter-btn").forEach(button => button.onclick = () => { state.filter = button.dataset.filter; loadReview(); });
}

function categoryStatus(slot) {
  if (slot.missing) return { icon: "○", label: "未导入", className: "muted" };
  if (!slot.files.length) return { icon: "!", label: "未匹配", className: "needs" };
    if (slot.files.every(file => file.match_method === "EXACT_FILENAME")) return { icon: "✓", label: "文件名一致，已直接关联", className: "auto" };
  if (slot.files.every(file => file.review_status === "CONFIRMED" || file.review_status === "CORRECTED")) return { icon: "✓", label: "已确认", className: "complete" };
  if (slot.files.some(file => file.match_details?.duplicate_note)) return { icon: "!", label: "内容重复，需复核", className: "needs" };
  if (slot.files.some(file => file.match_details?.auto_filled)) return { icon: "!", label: "自动补位，需复核", className: "needs" };
  if (slot.files.some(file => ["PENDING", "AMBIGUOUS", "LOW_CONFIDENCE", "UNMATCHED"].includes(file.status))) return { icon: "!", label: "需要确认", className: "needs" };
  return { icon: "✓", label: "系统已匹配", className: "auto" };
}

function renderGroups(groups) {
  $("#reviewGrid").innerHTML = groups.length ? groups.map(group => `
    <article class="review-product ${state.selectedMains.has(group.main.id) ? "selected" : ""}">
      <header><label class="group-select"><input type="checkbox" data-group-select="${group.main.id}" ${state.selectedMains.has(group.main.id) ? "checked" : ""} onchange="toggleGroupSelection(${group.main.id},this.checked)"> 选择</label><div><h2>${esc(group.main.basename)}</h2><small>${group.flags.confirmed ? "✓ 已完成" : "还有项目需要确认"}</small></div><button class="secondary" onclick="${group.flags.confirmed ? `unconfirmGroup(${group.main.id})` : `confirmGroup(${group.main.id})`}">${group.flags.confirmed ? "撤回确认" : "确认整组"}</button></header>
      <div class="category-checklist visual-category-grid">
        <button class="category-check visual-category main-category" onclick="preview('/api/main-images/${group.main.id}/preview','主图')"><img src="${group.main.thumbnail_url}" loading="lazy" alt="${esc(group.main.basename)} 主图"><span>✓</span><b>主图</b><small>匹配基准</small></button>
        ${group.categories.map(slot => { const status = categoryStatus(slot); const file = slot.files[0]; if (slot.files.length > 1) return `<div class="category-check visual-category needs duplicate-position"><b>${esc(slot.name)}</b><small>同一位置 ${slot.files.length} 张，请处理冲突</small><div class="duplicate-position-thumbs">${slot.files.map(row => `<button type="button" onclick="openMatchReview(${row.id})" title="${esc(row.original_filename)}"><img src="${row.thumbnail_url}" loading="lazy" alt="${esc(row.original_filename)}"><small>${esc(row.original_filename)}</small></button>`).join("")}</div></div>`; return `<button class="category-check visual-category ${status.className}" ${file ? `onclick="openMatchReview(${file.id})"` : slot.unmatched ? `data-unmatched-category="${esc(slot.name)}"` : "disabled"}>${file ? `<img src="${file.thumbnail_url}" loading="lazy" alt="${esc(file.original_filename)}">` : `<div class="category-image-placeholder">${slot.unmatched ? "!" : "—"}</div>`}<span>${status.icon}</span><b>${esc(slot.name)}</b><small>${status.label}</small></button>`; }).join("")}
      </div>
    </article>`).join("") : `<div class="empty-state panel">当前没有${filterLabels[state.filter] || ""}项目。${state.filter !== "all" ? '<button class="text-btn" onclick="showAllReview()">查看全部结果</button>' : ""}</div>`;
  $$('[data-unmatched-category]').forEach(button => {
    button.onclick = () => showUnmatchedCategory(button.dataset.unmatchedCategory);
  });
}

window.showAllReview = () => { state.filter = "all"; loadReview(); };
window.showUnmatchedCategory = category => { state.filter = "unmatched"; toast(`${category}需要人工归位，请在“未匹配 / 需人工选择”区选择主图`); loadReview(); };
window.toggleGroupSelection = (id, checked) => { checked ? state.selectedMains.add(id) : state.selectedMains.delete(id); renderBulkToolbar(); };
function renderBulkToolbar() { const ids = (state.reviewData?.groups || []).map(group => group.main.id); const selected = ids.filter(id => state.selectedMains.has(id)); const confirmedSelected = (state.reviewData?.groups || []).filter(group => selected.includes(group.main.id) && group.flags.confirmed).map(group => group.main.id); $("#selectedGroupCount").textContent = `已选 ${selected.length} 组`; $("#selectAllGroups").checked = ids.length > 0 && selected.length === ids.length; $("#selectAllGroups").indeterminate = selected.length > 0 && selected.length < ids.length; $("#confirmSelectedGroups").disabled = !selected.length; $("#unconfirmSelectedGroups").disabled = !confirmedSelected.length; $("#confirmAllVisibleGroups").disabled = !ids.length; }
$("#selectAllGroups").onchange = event => { for (const group of state.reviewData?.groups || []) event.target.checked ? state.selectedMains.add(group.main.id) : state.selectedMains.delete(group.main.id); renderGroups(state.reviewData?.groups || []); renderBulkToolbar(); };

async function bulkConfirmGroups(ids) {
  if (!ids.length || !confirm(`确认通过这 ${ids.length} 个商品组？`)) return;
  try { await api(`/api/batches/${state.batchId}/confirm-groups`, { method: "POST", body: JSON.stringify({ main_image_ids: ids }) }); state.selectedMains.clear(); toast("已确认，命名预览已同步更新"); await loadReview(); } catch (error) { toast(error.message, true); }
}
$("#confirmSelectedGroups").onclick = () => bulkConfirmGroups([...state.selectedMains]);
$("#confirmAllVisibleGroups").onclick = () => bulkConfirmGroups((state.reviewData?.groups || []).map(group => group.main.id));
$("#unconfirmSelectedGroups").onclick = async () => { const ids = (state.reviewData?.groups || []).filter(group => state.selectedMains.has(group.main.id) && group.flags.confirmed).map(group => group.main.id); if (!ids.length || !confirm(`撤回这 ${ids.length} 个商品组的审核确认？`)) return; try { await api(`/api/batches/${state.batchId}/unconfirm-groups`, { method: "POST", body: JSON.stringify({ main_image_ids: ids }) }); state.selectedMains.clear(); toast("已撤回确认，命名预览已更新"); await loadReview(); } catch (error) { toast(error.message, true); } };

function renderUnmatched(files) {
  const panel = $("#unmatchedPanel");
  panel.classList.toggle("hidden", !files.length);
  panel.innerHTML = files.length ? `<div class="panel-title"><h2>未匹配 / 需人工归位</h2><span class="muted">点击图片后选择正确主图；同分类冲突会自动交换原归属</span></div><div class="unmatched-grid">${files.map(file => `<button class="unmatched-item" onclick="openReassign(${file.id})"><img src="${file.thumbnail_url}" alt="${esc(file.original_filename)}"><div><b>${esc(file.folder_name || file.category)}${file.is_conflict ? " · 分类冲突" : ""}</b><small>${esc(file.original_filename)}${file.matched_main_basename ? ` · 当前：${esc(file.matched_main_basename)}` : " · 当前未匹配"}</small><span>点击选择主图</span></div></button>`).join("")}</div>` : "";
}

function allReviewFiles() {
  if (!state.reviewData) return [];
  const files = [...(state.reviewData.manual_selection_files || state.reviewData.unmatched_files || [])];
  for (const group of state.reviewData.groups || []) for (const category of group.categories || []) files.push(...(category.files || []));
  return [...new Map(files.map(file => [file.id, file])).values()];
}

function confidenceLabel(file) {
  if (file.match_method === "VISIONMATCH" && file.match_details?.duplicate_note) return file.match_details.duplicate_main_ids?.length ? "主图内容相同，需人工选择" : "图片内容重复，需人工复核";
  if (file.match_method === "VISIONMATCH" && file.match_details?.auto_filled) return "自动补位，需人工复核";
  if (file.match_method === "VISIONMATCH") return file.status === "MATCHED" ? "视觉证据一致，待审核" : "需人工复核";
  if (["EXACT_FILENAME", "FILENAME_EXACT"].includes(file.match_method) && file.matched_main_id) return "✓ 文件名一致，已直接关联";
  if (["PENDING", "AMBIGUOUS", "LOW_CONFIDENCE", "UNMATCHED"].includes(file.status) || file.assigned_score == null) return "不确定";
  if (file.assigned_score >= .9) return "高";
  return "一般";
}

window.openMatchReview = id => {
  const file = allReviewFiles().find(item => item.id === id);
  if (!file) return;
  state.reviewFile = file;
  $("#reviewModalTitle").textContent = `${file.folder_name || file.category} · ${file.original_filename}`;
  $("#reviewSourceLabel").textContent = "待匹配原图";
  $("#reviewSourceImage").src = file.preview_url;
  const mainPreviewUrl = file.matched_main_id ? `/api/main-images/${file.matched_main_id}/preview` : "";
  $("#reviewMainImage").src = mainPreviewUrl;
  $("#reviewMainImage").dataset.previewUrl = mainPreviewUrl;
  $("#reviewMainImage").dataset.previewTitle = `${file.matched_main_basename || "球队"} 主图`;
  $("#reviewConfidence").textContent = `匹配度：${confidenceLabel(file)}`;
  $("#reviewConfidence").dataset.level = confidenceLabel(file);
  const tabs = [{ label: "原图", url: file.preview_url }];
  if (file.derived?.mask) tabs.push({ label: "识别区域", url: file.derived.mask.url });
  $("#reviewSourceTabs").innerHTML = tabs.map((tab, index) => `<button class="${index === 0 ? "active" : ""}" onclick="switchReviewSource('${tab.url}',this)">${tab.label}</button>`).join("");
  const usesPhash = String(file.match_method || "").startsWith("PHASH");
  const isQwen = file.match_method === "QWEN_VISUAL";
  const methodLabel = file.match_method === "EXACT_FILENAME" ? "文件名直接关联" : file.match_method === "FILENAME_MATCH_AMBIGUOUS" ? "文件名匹配冲突，待确认" : isQwen ? "AI视觉匹配（Qwen）" : file.match_method === "LIGHTGLUE_COLOR" ? "LightGlue + HSV / Lab Color" : String(file.match_method || "").startsWith("DINO") ? "DINO 视觉特征" : "手动遮罩 pHash";
  const phashLine = usesPhash ? `<p><b>pHash Hamming Distance：</b>${file.phash_distance ?? "—"}</p><p><b>最佳主图区域：</b>${esc(file.phash_best_region_type || "—")}</p>` : "";
  const local = file.match_details;
  const lightGlueLine = file.match_method === "LIGHTGLUE_COLOR" && local ? `<p><b>Final Score：</b>${pct(local.final_score)}</p><p><b>Color Score：</b>${pct(local.color_score)}</p><p><b>Local Score：</b>${pct(local.local_score)}</p><p><b>Raw Matches：</b>${local.raw_matches ?? "—"}</p><p><b>RANSAC Inliers：</b>${local.ransac_inliers ?? "—"}</p><p><b>Inlier Ratio：</b>${pct(local.inlier_ratio)}</p><p><b>Coverage Score：</b>${pct(local.coverage_score)}</p>` : "";
  const qwenFinal = isQwen ? local?.final : null;
  const qwenEvidence = (qwenFinal?.evidence || []).map(esc).join("；") || "—";
  const qwenConflicts = (qwenFinal?.conflicts || []).map(esc).join("；") || "无明确冲突";
  const qwenPairwise = isQwen ? (local?.pairwise || []) : [];
  const qwenRanked = [...qwenPairwise].filter(row => !row.error).sort((a, b) => (b.fusion_score ?? -1) - (a.fusion_score ?? -1)).slice(0, 5);
  const qwenRankTable = qwenRanked.length ? `<div class="qwen-score-list"><b>Qwen + LightGlue / Color 候选排序（Top 5）</b>${qwenRanked.map((row, index) => `<div class="qwen-score-row"><strong>${index + 1}. ${esc(row.main_basename || `候选 ${row.candidate_index}`)}</strong><span>Qwen：${row.match ? "同款" : "否"} · ${pct(row.confidence)}</span><span>Color：${pct(row.color_score)}</span><span>LightGlue：${row.lightglue_refined ? pct(row.local_score) : "未精排"}</span><span>融合：${pct(row.fusion_score)}</span></div>`).join("")}</div>` : "";
  const fusion = local?.fusion;
  const fusionWeights = fusion?.weights ? `Qwen ${(fusion.weights.qwen * 100).toFixed(0)}% + LightGlue/Color ${(fusion.weights.lightglue_color * 100).toFixed(0)}%` : "—";
  const qwenLine = isQwen && local ? `<p><b>AI最终选择：</b>${esc(qwenFinal?.selected || qwenFinal?.decision_status || "处理中")} · ${esc(file.matched_main_basename || "暂未生成主图")}</p><p><b>两图判断：</b>${qwenPairwise.filter(row => row.match && !row.error).length}/${qwenPairwise.filter(row => !row.error).length} 个有效候选被判为同款</p><p><b>排序权重：</b>${esc(fusionWeights)}；LightGlue Color TopN=${fusion?.color_top_n ?? "—"}；区域=${esc(fusion?.query_roi_mode || fusion?.roi_mode || "—")}</p>${fusion?.query_roi_error ? `<p><b>目标区域回退：</b>${esc(fusion.query_roi_error)}</p>` : ""}${fusion?.lightglue_error ? `<p><b>LightGlue 状态：</b>${esc(fusion.lightglue_error)}（本次回退到 Color/Qwen 分数）</p>` : ""}${qwenRankTable}<p><b>最终 AI 依据：</b>${qwenEvidence}</p><p><b>最终冲突点：</b>${qwenConflicts}</p>${local.decision_note ? `<p><b>人工复核提示：</b>${esc(local.decision_note)}</p>` : ""}<p class="muted">Qwen confidence 是模型自报值；融合分只用于排序和 Top2 候选选择，二者都不是校准概率。最终裁决仍直接比较原图。</p>` : "";
  const scoreDetails = ["EXACT_FILENAME", "FILENAME_EXACT"].includes(file.match_method) ? `<p>✓ 文件名一致，已直接关联</p><p>源文件：${esc(file.source_filename || file.original_filename)}</p><p>主图：${esc(file.matched_main_image_filename || file.matched_main_basename)}</p>` : `<p><b>相似度：</b>${pct(file.assigned_score)}</p><p><b>Top1：</b>${esc(file.original_top1?.basename || "—")} ${pct(file.original_top1?.score)}</p><p><b>Top2：</b>${pct(file.original_top2?.score)}</p><p><b>Margin：</b>${pct(file.top1_top2_margin)}</p><p><b>Assigned Rank：</b>${file.assigned_rank ?? "—"}</p>`;
  $("#matchDetailsBody").innerHTML = `<p><b>匹配方式：</b>${methodLabel}</p>${isQwen ? "" : scoreDetails}<p><b>算法版本：</b>${esc(file.algorithm_version || "—")}</p>${qwenLine}${phashLine}${lightGlueLine}`;
  if (file.match_method === "VISIONMATCH") {
    const result = file.match_details || {};
    const suggestion = result.suggestion || {};
    const label = {MATCH: "同款同配色", DIFFERENT: "存在商品差异", UNCERTAIN: "证据不足", UNMATCHED: "未找到对应"};
    $("#reviewConfidence").textContent = confidenceLabel(file);
    $("#matchDetailsBody").innerHTML = `<p><b>匹配方式：</b>VisionMatch AI · ${esc(result.model || "Terra")}</p><p><b>模型判断：</b>${esc(label[suggestion.decision] || suggestion.decision || "待复核")}</p><p><b>一致证据：</b>${esc((suggestion.evidence || []).join("；") || "未记录")}</p><p><b>冲突：</b>${esc((suggestion.conflicts || []).join("；") || "无明确冲突")}</p><p><b>复核提示：</b>${esc((suggestion.review_reasons || []).join("；") || "请核对商品细节")}</p><p>拼图判断不提供数值相似度；可点击“换一个”查看候选或手动调整。</p>`;
    if (result.auto_filled) $("#matchDetailsBody").insertAdjacentHTML("afterbegin", "<p><b>自动补位，需人工复核：</b>模型未确定此归属，系统将图片补入同分类剩余空位；下方保留原模型判断。</p>");
    if (result.duplicate_note) $("#matchDetailsBody").insertAdjacentHTML("afterbegin", `<p><b>重复图片：</b>${esc(result.duplicate_note)}</p>`);
  }
  $("#matchDetails").open = false;
  $("#reviewCorrect").disabled = !file.matched_main_id;
  $("#reviewModal").classList.remove("hidden");
};
window.switchReviewSource = (url, button) => { $("#reviewSourceImage").src = url; $$("#reviewSourceTabs button").forEach(item => item.classList.toggle("active", item === button)); };
window.openReviewMainPreview = () => {
  const image = $("#reviewMainImage");
  if (image.dataset.previewUrl) preview(image.dataset.previewUrl, image.dataset.previewTitle || "球队主图");
};
$("#reviewCorrect").onclick = async () => { if (!state.reviewFile) return; await confirmFile(state.reviewFile.id); $("#reviewModal").classList.add("hidden"); };
$("#reviewChange").onclick = () => { const id = state.reviewFile?.id; $("#reviewModal").classList.add("hidden"); if (id) openReassign(id); };
$("#reviewLater").onclick = () => $("#reviewModal").classList.add("hidden");

window.confirmGroup = async id => { try { await api(`/api/main-images/${id}/confirm`, { method: "POST" }); toast("当前商品已确认"); await loadReview(); } catch (error) { toast(error.message, true); } };
window.unconfirmGroup = async id => { if (!confirm("撤回这一组的审核确认？人工换图结果会保留。")) return; try { await api(`/api/main-images/${id}/unconfirm`, { method: "POST", body: "{}" }); toast("已撤回确认，可以重新审核"); await loadReview(); } catch (error) { toast(error.message, true); } };
window.confirmFile = async id => { try { await api(`/api/files/${id}/confirm`, { method: "POST", body: "{}" }); toast("匹配已确认"); await loadReview(); } catch (error) { toast(error.message, true); } };

window.openReassign = async id => {
  let file = allReviewFiles().find(item => item.id === id);
  if (!file) {
    const data = await api(`/api/batches/${state.batchId}/review?filter=all`);
    const candidates = [...(data.manual_selection_files || data.unmatched_files || [])];
    for (const group of data.groups) for (const category of group.categories) candidates.push(...category.files);
    file = candidates.find(item => item.id === id);
  }
  if (!file) file = await api(`/api/files/${id}`);
  if (!file) return;
  state.reassignFile = file;
  $("#reassignImage").src = file.preview_url;
  $("#reassignFilename").textContent = file.original_filename;
  $("#reassignCurrent").textContent = file.matched_main_basename ? `当前匹配：${file.matched_main_basename}` : "当前：未匹配";
  $("#mainSearch").value = "";
  $("#reassignModal").classList.remove("hidden");
  $("#candidateList").innerHTML = '<span class="muted">正在加载全部球队…</span>';
  // Manual repair must be able to move an image into a confirmed product
  // group when that group is missing this category.
  state.reassignCandidates = await api(`/api/files/${file.id}/product-candidates?exclude_confirmed=true&manual_selection=true`);
  renderCandidates(state.reassignCandidates, "#candidateList");
};

function renderCandidates(items, target) {
  $("#candidateCount").textContent = `${items.length} 支可选球队 · 已按本次匹配模型排序`;
  $(target).innerHTML = items.map(item => {
    const mainId = item.main_image_id || item.id;
    const current = mainId === state.reassignFile?.matched_main_id;
    const qwenFusion = item.score_source === "QWEN_FUSION_PAIRWISE";
    const lightGlue = item.details?.refined || item.model_name?.includes("LightGlue");
    const model = qwenFusion ? "Qwen + LightGlue / Color" : lightGlue ? "LightGlue + Color" : item.model_name?.includes("dinov2") ? "DINOv2" : item.model_name?.includes("dinov3") ? "DINOv3" : "pHash";
    const detail = qwenFusion ? `${model} · Qwen ${item.details?.match ? "同款" : "否"} · 融合 ${pct(item.details?.fusion_score)} · Color ${pct(item.details?.color_score)}` : lightGlue ? `${model} · 颜色 ${pct(item.details?.color_score)} · 局部 ${pct(item.details?.local_score)}` : item.best_distance == null ? model : `pHash 距离 ${item.best_distance} · ${item.support_count ?? 0} 条支持`;
    const score = item.score_source === "VISIONMATCH_EVIDENCE" ? `${item.decision === "MATCH" ? "模型建议" : item.decision === "UNCERTAIN" ? "备选，需复核" : "未由模型提名，可手动选择"}${item.evidence?.length ? ` · ${esc(item.evidence.join("；"))}` : ""}` : item.aggregated_score == null ? "" : `相似度 ${pct(item.aggregated_score)} · ${detail}`;
    return `<button class="candidate ${current ? "current" : ""}" onclick="chooseMain(${mainId})"><span class="candidate-image-zoom" data-candidate-preview="${mainId}" data-preview-title="${esc(item.basename)}"><img src="${item.thumbnail_url}" loading="lazy" alt="${esc(item.basename)}"><em>🔍 点击放大</em></span><strong>${esc(item.basename)}</strong><small>${current ? "当前匹配 · " : ""}${score || "点击替换"}</small></button>`;
  }).join("") || '<span class="muted">没有可替换球队，其他球队组均已审核通过</span>';
  $$(`${target} [data-candidate-preview]`).forEach(element => {
    element.onclick = event => {
      event.preventDefault();
      event.stopPropagation();
      preview(`/api/main-images/${element.dataset.candidatePreview}/preview`, `${element.dataset.previewTitle} 主图`);
    };
  });
}
window.chooseMain = async mainId => { try { const result = await api(`/api/files/${state.reassignFile.id}/reassign`, { method: "POST", body: JSON.stringify({ main_image_id: mainId }) }); $("#reassignModal").classList.add("hidden"); toast(result.swapped_file_id ? "已替换，并自动交换原有归属" : "已更换并确认主图"); await loadReview(); } catch (error) { toast(error.message, true); } };
$("#searchMainBtn").onclick = () => { const query = $("#mainSearch").value.trim().toLocaleLowerCase(); renderCandidates(state.reassignCandidates.filter(item => !query || item.basename.toLocaleLowerCase().includes(query)), "#candidateList"); };
$("#mainSearch").onkeydown = event => { if (event.key === "Enter") $("#searchMainBtn").click(); };
$("#refreshReview").onclick = loadReview;
$("#reconcileFiles").onclick = async () => {
  const button = $("#reconcileFiles");
  if (!state.batchId) return;
  try {
    button.disabled = true;
    button.textContent = "检查中…";
    const result = await api(`/api/batches/${state.batchId}/scan`, { method: "POST" });
    await refreshWorkspace(true);
    await loadReview();
    toast(`文件核对完成：恢复 ${result.restored_file_count || 0}，新增 ${result.added_file_count || 0}，仍缺失 ${result.missing_file_count || 0}；原匹配已保留`);
  } catch (error) { toast(error.message, true); }
  finally { button.disabled = false; button.textContent = "重新检查文件"; }
};

const renameStatusNames = { READY: "将重命名", SKIPPED: "未确认，暂不改名", BLOCKED: "需要处理", FILE_MISSING: "源文件不可用", NOOP: "名称已正确", RENAME_CONFLICT: "命名冲突", RENAMED: "已完成", IGNORED: "已忽略（保持原名）" };
function renderRenameIssues(issues, busy) {
  if (!issues.length) return "";
  return `<section class="rename-issues" id="renameIssueList">
    <div class="rename-issues-heading"><h3>异常图片与冲突标记</h3><button class="secondary" data-mark-all-conflicts ${busy || !issues.some(item => item.can_mark_conflict && !item.conflict_marked) ? "disabled" : ""}>全部标记为冲突图片</button></div>
    <p>勾选后生成特殊名称，并移出正常匹配位置；点击执行命名后才会修改文件。取消标记后需重新确认归属。</p>
    <div class="rename-issue-grid">${issues.map(item => `<article class="rename-issue ${item.conflict_marked ? "marked" : ""}">
      <div class="rename-issue-head"><strong>${esc(item.current_filename || item.original_filename)}</strong><span>${esc(item.category)}</span></div>
      <p class="rename-issue-reason">${esc(item.category_ignored ? "该类型目前已忽略；历史重复位置仍保留在这里，可恢复该类型后单独标记问题图片" : item.error_message || (item.conflict_marked ? "已标记为冲突图片，不占用正常匹配位置" : item.conflict_peers?.length ? `与 ${item.conflict_peers.map(peer => peer.current_filename).join("、")} 指向同一位置` : "可标记为冲突图片或更正匹配"))}</p>
      <label class="conflict-mark-choice"><input type="checkbox" data-conflict-file="${item.source_file_id}" ${item.conflict_marked ? "checked" : ""} ${busy || (!item.can_mark_conflict && !item.conflict_marked) ? "disabled" : ""}>标记为冲突图片</label>
      <dl><dt>原图位置</dt><dd><code>${esc(item.source_path)}</code></dd><dt>${item.conflict_marked ? "标记后文件名" : "当前计划名称"}</dt><dd><b>${esc(item.target_filename || "—")}</b></dd><dt>目标位置</dt><dd><code>${esc(item.target_path || "尚无有效目标")}</code></dd></dl>
      <div class="rename-issue-actions"><button class="secondary" data-issue-preview="${item.source_file_id}" ${item.rename_status === "FILE_MISSING" ? "disabled" : ""}>查看原图</button>
      ${item.category_ignored ? `<button class="secondary" data-restore-issue-category="${item.source_file_id}" ${busy ? "disabled" : ""}>恢复该类型</button>` : `<button class="secondary" data-issue-reassign="${item.source_file_id}" ${busy ? "disabled" : ""}>更正匹配</button>`}
      <button class="secondary" data-issue-copy="${item.source_file_id}">复制所在目录</button></div>
    </article>`).join("")}</div></section>`;
}
function renderRenamePlan(plan) {
  const plans = plan?.plans || [];
  const blockers = plans.filter(item => !["READY", "NOOP", "RENAMED", "IGNORED", "SKIPPED"].includes(item.rename_status));
  const skipped = plans.filter(item => item.rename_status === "SKIPPED");
  const ready = plans.filter(item => item.rename_status === "READY");
  const alreadyNamed = plans.filter(item => item.rename_status === "NOOP");
  const actionable = ready.length + alreadyNamed.length;
  const marked = plans.filter(item => item.conflict_marked);
  $("#showUnprocessed").classList.toggle("hidden", !blockers.length && !skipped.length);
  $("#renameBusinessStatus").textContent = blockers.length ? `发现 ${blockers.length} 项异常，具体图片和位置已列在下方；可更正归属或勾选冲突标记` : marked.length ? `${marked.length} 张已标记为冲突图片，将按 A-原图片名-冲突图片序号命名；其他图片按预览处理` : ready.length ? `只重命名 ${ready.length} 个已确认文件；其余未确认或已忽略文件保持原名` : alreadyNamed.length ? `${alreadyNamed.length} 个已确认文件名称已正确，可记录完成。` : skipped.length ? `还有 ${skipped.length} 个文件尚未确认，本次保持原名；可随时继续审核` : plans.length ? "已确认项目命名已完成，其余文件保持原名。" : "确认需要处理的项目后即可预览命名结果。";
  $("#executeRenamePlan").disabled = blockers.length > 0 || !actionable || !plan?.plan_group || state.renameSaving || state.conflictMarkSaving;
  $("#executeRenamePlan").textContent = !plans.length ? "请先预览命名结果" : blockers.length ? "请先处理异常图片" : alreadyNamed.length && !ready.length ? "确认完成" : !actionable ? (skipped.length ? "暂无已确认项目" : "已完成") : marked.length ? "执行命名（含冲突标记）" : "仅改名已确认项目";
  if (!plans.length) { $("#renamePlanBody").innerHTML = '<span class="muted">点击“预览命名结果”查看即将修改的文件名。</span>'; return; }
  const chips = Object.entries(plan.counts || {}).map(([status, count]) => `<span class="plan-chip status-${status.toLowerCase()}">${esc(renameStatusNames[status] || status)} <b>${count}</b></span>`).join("");
  const issues = plans.filter(item => blockers.includes(item) || item.can_mark_conflict || item.conflict_marked || item.normal_position_conflict || item.matching_position_conflict);
  const busy = state.renameSaving || state.conflictMarkSaving || state.reviewSaving;
  const issueHtml = renderRenameIssues(issues, busy);
  $("#renamePlanBody").innerHTML = `<div class="rename-summary">${chips}</div>${issueHtml}<details class="advanced-details"><summary>查看命名明细</summary><div class="rename-table-wrap"><table class="rename-table"><thead><tr><th>分类</th><th>当前名称</th><th>新名称</th><th>状态</th></tr></thead><tbody>${plans.map(item => `<tr><td>${esc(item.category)}</td><td>${esc(item.current_filename || item.original_filename)}</td><td>${esc(item.target_filename || "—")}</td><td>${esc(renameStatusNames[item.rename_status] || item.rename_status)}${item.error_message ? `<br><small>${esc(item.error_message)}</small>` : ""}</td></tr>`).join("")}</tbody></table></div></details>`;
  $$('[data-conflict-file]').forEach(input => { input.onchange = () => { const ids = new Set(marked.map(item => item.source_file_id)); if (input.checked) ids.add(Number(input.dataset.conflictFile)); else ids.delete(Number(input.dataset.conflictFile)); saveConflictMarks([...ids]); }; });
  const markAll = $('[data-mark-all-conflicts]');
  if (markAll) markAll.onclick = () => saveConflictMarks([...new Set([...marked, ...issues.filter(item => item.can_mark_conflict)].map(item => item.source_file_id))]);
  $$('[data-restore-issue-category]').forEach(button => { button.onclick = async () => { const item = issues.find(row => row.source_file_id === Number(button.dataset.restoreIssueCategory)); button.disabled = true; try { await api(`/api/batches/${state.batchId}/review-categories`, {method: "POST", body: JSON.stringify({category: item.category, ignored: false})}); await loadReview(); toast("该类型已恢复，可以只标记冲突图片"); } catch (error) { button.disabled = false; toast(error.message, true); } }; });
  $$('[data-issue-preview]').forEach(button => { button.onclick = () => { const item = issues.find(row => row.source_file_id === Number(button.dataset.issuePreview)); preview(`/api/files/${item.source_file_id}/preview`, item.current_filename || item.original_filename); }; });
  $$('[data-issue-reassign]').forEach(button => { button.onclick = () => openReassign(Number(button.dataset.issueReassign)).catch(error => toast(error.message, true)); });
  $$('[data-issue-copy]').forEach(button => { button.onclick = async () => { const item = issues.find(row => row.source_file_id === Number(button.dataset.issueCopy)); try { await copyRenamePath(item.source_directory || item.source_path); toast("图片所在目录已复制"); } catch { toast("复制失败，可直接选中原图位置复制", true); } }; });
}
async function copyRenamePath(path) {
  if (navigator.clipboard?.writeText) { try { await navigator.clipboard.writeText(path); return; } catch {} }
  const input = document.createElement("textarea"); input.value = path; input.style.position = "fixed"; input.style.opacity = "0"; document.body.appendChild(input); input.select();
  try { if (!document.execCommand("copy")) throw new Error("copy failed"); } finally { input.remove(); }
}
async function saveConflictMarks(fileIds) {
  if (state.conflictMarkSaving || state.renameSaving || state.reviewSaving) return;
  state.conflictMarkSaving = true; renderRenamePlan(state.renamePlan);
  try { const result = await api(`/api/batches/${state.batchId}/rename-conflict-marks`, {method: "POST", body: JSON.stringify({file_ids: fileIds})}); state.renamePlan = result.rename_plan; await loadReview(); toast("冲突标记已保存，特殊文件名已生成；尚未改动文件"); }
  catch (error) { toast(error.message, true); }
  finally { state.conflictMarkSaving = false; renderRenamePlan(state.renamePlan); }
}
$("#generateRenamePlan").onclick = async () => { try { state.renamePlan = await api(`/api/batches/${state.batchId}/rename-plan`, { method: "POST" }); renderRenamePlan(state.renamePlan); const recovered = state.renamePlan.recovered_source_count || 0; toast(recovered ? `已找回 ${recovered} 个重新放回的文件，原匹配已保留` : "命名预览已更新，尚未修改文件"); } catch (error) { toast(error.message, true); } };
$("#showUnprocessed").onclick = () => { const issues = $("#renameIssueList"); if (issues) { issues.scrollIntoView({behavior: "smooth", block: "start"}); return; } state.filter = "pending"; loadReview(); window.scrollTo({ top: $("#filters").offsetTop - 90, behavior: "smooth" }); };
$("#executeRenamePlan").onclick = () => {
  if (!state.renamePlan?.plan_group || $("#executeRenamePlan").disabled) return;
  const changesNames = state.renamePlan.plans.some(item => item.rename_status === "READY");
  $("#renameConfirmTitle").textContent = changesNames ? "确认执行重命名" : "确认完成命名";
  $("#renameConfirmMessage").textContent = changesNames ? (state.renamePlan.plans.some(item => item.conflict_marked) ? "已标记的冲突图片按 A-原图片名-冲突图片序号命名，其余已确认图片按商品名称命名；以本次预览为准。" : "只修改本次预览中已确认文件的名称。未确认及已忽略类型保持原名，之后可继续审核改名。") : "已确认文件名称已正确，本次只记录这些项目完成；其他项目保持原状。";
  $("#confirmExecuteRename").textContent = changesNames ? "确认执行" : "确认完成";
  $("#renameConfirmModal").classList.remove("hidden");
};
$("#confirmExecuteRename").onclick = async () => {
  if (state.renameSaving || !state.renamePlan?.plan_group) return;
  const changesNames = state.renamePlan.plans.some(item => item.rename_status === "READY");
  const button = $("#confirmExecuteRename");
  state.renameSaving = true;
  button.disabled = true;
  renderRenamePlan(state.renamePlan);
  try {
    state.renamePlan = await api(`/api/batches/${state.batchId}/rename-plan/execute`, { method: "POST", body: JSON.stringify({ plan_group: state.renamePlan.plan_group }) });
    $("#renameConfirmModal").classList.add("hidden");
    const cleanup = state.renamePlan.workspace_cleanup;
    toast(cleanup && !cleanup.removed ? "已确认项目命名完成，临时文件清理失败" : changesNames ? "已确认项目重命名完成，其余项目可继续审核" : "已确认项目已记录完成，其余项目保持原状");
    await loadReview();
  } catch (error) { toast(error.message, true); }
  finally {
    state.renameSaving = false;
    button.disabled = false;
    renderRenamePlan(state.renamePlan);
  }
};

$$("[data-close]").forEach(button => button.onclick = () => $("#" + button.dataset.close).classList.add("hidden"));
window.preview = (url, title = "图片预览") => { Object.assign(state.lightbox, { scale: 1, x: 0, y: 0 }); $("#lightboxTitle").textContent = title; $("#lightboxImage").src = url; $("#lightbox").classList.remove("hidden"); applyLightboxTransform(); };
function applyLightboxTransform() { const box = state.lightbox; $("#lightboxImage").style.transform = `translate(${box.x}px,${box.y}px) scale(${box.scale})`; $("#zoomLabel").textContent = `${Math.round(box.scale * 100)}%`; }
function setZoom(next) { state.lightbox.scale = Math.max(.1, Math.min(8, next)); applyLightboxTransform(); }
$("#zoomIn").onclick = () => setZoom(state.lightbox.scale * 1.25);
$("#zoomOut").onclick = () => setZoom(state.lightbox.scale / 1.25);
$("#zoomReset").onclick = () => { Object.assign(state.lightbox, { scale: 1, x: 0, y: 0 }); applyLightboxTransform(); };
$("#zoomActual").onclick = () => { const image = $("#lightboxImage"), viewport = $("#lightboxViewport"); const fit = Math.min(viewport.clientWidth / image.naturalWidth, viewport.clientHeight / image.naturalHeight, 1); Object.assign(state.lightbox, { scale: fit ? 1 / fit : 1, x: 0, y: 0 }); applyLightboxTransform(); };
$("#lightboxViewport").onwheel = event => { event.preventDefault(); setZoom(state.lightbox.scale * (event.deltaY < 0 ? 1.12 : .89)); };
$("#lightboxViewport").onpointerdown = event => { const box = state.lightbox; box.dragging = true; box.startX = event.clientX - box.x; box.startY = event.clientY - box.y; $("#lightboxViewport").setPointerCapture(event.pointerId); };
$("#lightboxViewport").onpointermove = event => { const box = state.lightbox; if (!box.dragging) return; box.x = event.clientX - box.startX; box.y = event.clientY - box.startY; applyLightboxTransform(); };
$("#lightboxViewport").onpointerup = () => { state.lightbox.dragging = false; };

async function initializeWorkspace() {
  try {
    const config = await api("/api/workspace-config");
    $("#rootPath").value = config.upload_root || "";
    $("#rootPath").placeholder = config.upload_root || "公共盘文件包目录";
  } catch (error) {
    toast("无法读取公共盘设置", true);
  }
  await loadBatches();
}

setMaskMode("crop");
initializeWorkspace();
