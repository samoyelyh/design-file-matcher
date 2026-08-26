const state = {
  batchId: null,
  batch: null,
  workspace: deriveWorkspaceState(null),
  filter: "all",
  reviewData: null,
  renamePlan: null,
  selectedMains: new Set(),
  reviewFile: null,
  reassignFile: null,
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
  const response = await fetch(url, { headers: { "Content-Type": "application/json", ...(options.headers || {}) }, ...options });
  if (!response.ok) {
    let payload;
    try { payload = await response.json(); } catch { payload = { detail: await response.text() }; }
    throw new Error(payload.detail || "操作失败，请重试");
  }
  return response.json();
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
  if (view === "process") return "请先完成识别区域设置";
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

async function loadBatches() {
  const rows = await api("/api/batches");
  const list = $("#batchList");
  list.classList.toggle("empty", !rows.length);
  list.innerHTML = rows.length ? rows.map(batch => `
    <div class="batch-row">
      <div class="batch-root"><b>批次 #${batch.id}</b><small>${batch.main_count || 0} 个商品</small></div>
      <span class="status-pill">${esc(batchStatusNames[batch.status] || batch.status)}</span>
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
  $("#workspaceAdvanced").innerHTML = `<p><b>处理设备：</b>${esc(batch.progress?.device || "将在匹配时自动选择")}</p><p><b>算法版本：</b>${esc(config.algorithm_version || "—")}</p><p><b>PSD 匹配：</b>${esc(config.psd_algorithm_version || "—")}</p><p><b>当前目录：</b><code>${esc(batch.root_path)}</code></p>`;
}

$("#continueBtn").onclick = () => showView(state.workspace.nextView);

function renderImportSummary() {
  const batch = state.batch;
  const visible = batch && state.workspace.totals.files > 0;
  $("#currentImportSummary").classList.toggle("hidden", !visible);
  if (!visible) return;
  const rows = [["主图", batch.main_count || 0], ...Object.entries(batch.categories || {})];
  $("#importCounts").innerHTML = rows.map(([name, count]) => `<div><span>${esc(name)}</span><b>${count}</b><i>✓</i></div>`).join("");
  $("#importAnomalies").textContent = "异常文件 0";
  $("#importRootPath").textContent = batch.root_path;
}

$("#nextToRegions").onclick = () => showView("mask");
$("#refreshBatches").onclick = loadBatches;

async function importPublicFolder(root) {
  if (!root) return toast("请选择公共盘文件包", true);
  try {
    $("#folderPickerConfirm").disabled = true;
    $("#folderPickerConfirm").textContent = "正在导入…";
    const batch = await api("/api/batches", { method: "POST", body: JSON.stringify({ root_path: root }) });
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
  $("#fileCount").textContent = state.workspace.totals.files;
  $("#regionReadyCount").textContent = `${state.workspace.totals.regionsDone} / ${state.workspace.totals.regionsTotal}`;
  const ready = batch.mask_readiness?.ready === true;
  const missing = batch.mask_readiness?.missing_categories || [];
  $("#maskPreflight").classList.toggle("hidden", ready);
  $("#maskPreflightText").textContent = missing.length ? `还有 ${missing.length} 个图片分类未设置：${missing.join("、")}` : "";
  const matchable = ["SCANNED", "COMPLETED", "FAILED"].includes(batch.status) && ready;
  $("#matchBtn").disabled = !matchable;
  $("#matchBtn").classList.remove("hidden");
  $("#matchBtn").textContent = batch.status === "COMPLETED" ? "使用当前识别区域重新匹配" : batch.status === "FAILED" ? "重新开始自动匹配" : "开始自动匹配";
  $("#processError").classList.toggle("hidden", !batch.error_message);
  $("#processError").textContent = batch.error_message || "";
  const config = batch.matching_config || {};
  $("#technicalConfig").innerHTML = `<b>设备：</b>${esc(progress.device || "自动选择 GPU")}　<b>模型：</b>OpenCLIP　<b>算法版本：</b>${esc(config.algorithm_version || "—")}　<b>阈值：</b>${Math.round((config.match_min_score || 0) * 100)}%`;
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
  $("#resultMatched").textContent = Math.max(0, state.workspace.totals.files - Number(summary.unmatched_files || 0));
  $("#resultPending").textContent = summary.pending || 0;
  $("#resultUnmatched").textContent = summary.unmatched_files || 0;
}

$("#goMaskSetup").onclick = () => showView("mask");
$("#reviewPendingBtn").onclick = () => { state.filter = "pending"; showView("review"); };
$("#viewAllResultsBtn").onclick = () => { state.filter = "all"; showView("review"); };

$("#matchBtn").onclick = async () => {
  try {
    if (state.batch?.status === "COMPLETED" && !confirm("确定重新匹配？系统会使用最新识别区域重新计算未确认结果；已整组确认和人工调换的结果会保留。")) return;
    await api(`/api/batches/${state.batchId}/match`, { method: "POST" });
    await refreshWorkspace(false);
    startPolling();
    toast("自动匹配已开始");
  } catch (error) { toast(error.message.includes("Mask") ? "请先完成识别区域设置" : error.message, true); }
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
  const active = new Map(data.profiles.filter(item => item.enabled && item.creation_method === "MANUAL").map(item => [item.image_category, item]));
  $("#maskProgressText").textContent = `${data.readiness.configured_categories.length} / ${data.readiness.required_categories.length} 已完成`;
  $("#nextToMatch").classList.toggle("hidden", !data.readiness.ready);
  $("#maskProfileList").innerHTML = data.categories.map(category => {
    if (category === "PSD") return `<div class="mask-profile-row disabled"><div><b>PSD</b><small>无需设置识别区域</small></div><span>✓</span></div>`;
    const profile = active.get(category);
    const complete = Array.isArray(profile?.crop_box);
    if (complete) return `<div class="mask-profile-row ${state.maskCategory === category ? "active" : ""}"><button onclick="openMaskCategory('${esc(category)}')"><div><b>${esc(category)}</b><small>✓ 已完成</small></div><span>重新调整</span></button></div>`;
    if (profile) return `<div class="mask-profile-row needs"><button onclick="openMaskCategory('${esc(category)}')"><div><b>${esc(category)}</b><small>! 该旧设置需要补充识别范围</small></div><span>补充识别范围</span></button></div>`;
    return `<div class="mask-profile-row needs"><button onclick="openMaskCategory('${esc(category)}')"><div><b>${esc(category)}</b><small>! 待设置</small></div><span>去设置</span></button></div>`;
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
  let profile = (state.maskData?.profiles || []).find(item => item.image_category === category && item.enabled && item.creation_method === "MANUAL");
  try {
    const files = await api(`/api/batches/${state.batchId}/category-files/${encodeURIComponent(category)}`);
    if (!files.length) throw new Error("这个分类没有可用图片");
    if (!profile) profile = { name: category, version: 0, reference_file_id: files[0].id, mask_url: null, crop_box: null, draft: true };
    state.maskProfile = profile;
    state.maskHistory = [];
    state.maskRedo = [];
    setMaskMode("crop");
    $("#fineTunePanel").classList.add("hidden");
    $("#maskEmpty").classList.add("hidden");
    $("#maskEditor").classList.remove("hidden");
    $("#maskEditorTitle").textContent = category;
    $("#maskVersion").textContent = profile.draft ? "尚未设置" : profile.crop_box ? "已设置" : "该旧设置需要补充识别范围";
    $("#maskReference").innerHTML = files.map(file => `<option value="${file.id}" data-url="${file.preview_url}" ${file.id === profile.reference_file_id ? "selected" : ""}>${esc(file.filename)}</option>`).join("");
    await renderMaskCanvas($("#maskReference").selectedOptions[0]?.dataset.url, profile.mask_url, profile.crop_box);
    renderMaskProfileList(state.maskData);
  } catch (error) {
    $("#maskEditor").classList.add("hidden");
    $("#maskEmpty").classList.remove("hidden");
    $("#maskEmpty").innerHTML = "<b>参考图片无法打开</b><p>图片可能已移动，请返回导入页重新扫描当前文件夹。</p>";
    toast(error.message === "参考图片无法打开" ? "参考图片已移动，请重新扫描当前文件夹" : error.message, true);
  }
};

function loadImage(url) { return new Promise((resolve, reject) => { const image = new Image(); image.onload = () => resolve(image); image.onerror = () => reject(new Error("参考图片无法打开")); image.src = url; }); }

async function renderMaskCanvas(referenceUrl, maskUrl, cropBox) {
  if (!referenceUrl) return;
  const [reference, mask] = await Promise.all([loadImage(referenceUrl), maskUrl ? loadImage(maskUrl) : Promise.resolve(null)]);
  const image = $("#maskReferenceImage");
  const canvas = $("#maskCanvas");
  const context = canvas.getContext("2d", { willReadFrequently: true });
  image.src = referenceUrl;
  canvas.width = reference.naturalWidth;
  canvas.height = reference.naturalHeight;
  context.imageSmoothingEnabled = false;
  context.fillStyle = "black";
  context.fillRect(0, 0, canvas.width, canvas.height);
  if (mask) context.drawImage(mask, 0, 0, canvas.width, canvas.height);
  state.maskHistory = [context.getImageData(0, 0, canvas.width, canvas.height)];
  state.maskRedo = [];
  state.cropBox = Array.isArray(cropBox) ? [...cropBox] : null;
  updateCropOverlay();
}

function pushMaskHistory() {
  const canvas = $("#maskCanvas");
  state.maskHistory.push(canvas.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, canvas.width, canvas.height));
  if (state.maskHistory.length > 20) state.maskHistory.shift();
  state.maskRedo = [];
}

function restoreMask(snapshot) { $("#maskCanvas").getContext("2d").putImageData(snapshot, 0, 0); }

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
  $("#cropBoxInfo").textContent = `已框选 AI 识别画面 · ${width} × ${height}px`;
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
}

function updateCropFromEvent(event) {
  if (!state.cropStart) return;
  const point = canvasPoint(event);
  state.cropBox = [Math.min(state.cropStart.nx, point.nx), Math.min(state.cropStart.ny, point.ny), Math.max(state.cropStart.nx, point.nx), Math.max(state.cropStart.ny, point.ny)];
  updateCropOverlay();
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
  } else if (drawing) { drawing = false; pushMaskHistory(); }
};

$$("[data-mask-mode]").forEach(button => button.onclick = () => setMaskMode(button.dataset.maskMode));
$("#maskFineTuneToggle").onclick = () => { $("#fineTunePanel").classList.remove("hidden"); setMaskMode("add"); };
$("#maskFineTuneClose").onclick = () => { $("#fineTunePanel").classList.add("hidden"); setMaskMode("crop"); };
$("#maskFillCrop").onclick = () => {
  if (!state.cropBox) return toast("请先在图片上拖出识别范围", true);
  const canvas = $("#maskCanvas");
  const context = canvas.getContext("2d");
  const [left, top, right, bottom] = state.cropBox;
  context.fillStyle = "black";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "white";
  context.fillRect(Math.round(left * canvas.width), Math.round(top * canvas.height), Math.round((right - left) * canvas.width), Math.round((bottom - top) * canvas.height));
  pushMaskHistory();
  toast("已使用当前框选区域，可直接保存或精细调整");
};
$("#maskClearCrop").onclick = () => { state.cropBox = null; updateCropOverlay(); setMaskMode("crop"); };
$("#maskUndo").onclick = () => { if (state.maskHistory.length < 2) return; state.maskRedo.push(state.maskHistory.pop()); restoreMask(state.maskHistory.at(-1)); };
$("#maskRedo").onclick = () => { const next = state.maskRedo.pop(); if (!next) return; state.maskHistory.push(next); restoreMask(next); };
$("#maskClear").onclick = () => { const canvas = $("#maskCanvas"); const context = canvas.getContext("2d"); context.fillStyle = "black"; context.fillRect(0, 0, canvas.width, canvas.height); pushMaskHistory(); };
$("#maskPreviewToggle").onclick = () => { const canvas = $("#maskCanvas"); const hidden = canvas.style.opacity === "0"; canvas.style.opacity = hidden ? ".48" : "0"; $("#maskPreviewToggle").textContent = hidden ? "隐藏识别区域" : "查看识别区域"; };

function buildAiPreview() {
  if (!state.cropBox) throw new Error("请先框选识别范围");
  const image = $("#maskReferenceImage");
  const mask = $("#maskCanvas");
  const full = document.createElement("canvas");
  const context = full.getContext("2d", { willReadFrequently: true });
  full.width = mask.width; full.height = mask.height;
  context.fillStyle = "#f5f5f5"; context.fillRect(0, 0, full.width, full.height);
  const source = document.createElement("canvas");
  const sourceContext = source.getContext("2d", { willReadFrequently: true });
  source.width = full.width; source.height = full.height;
  sourceContext.drawImage(image, 0, 0, full.width, full.height);
  const pixels = sourceContext.getImageData(0, 0, full.width, full.height);
  const maskPixels = mask.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, full.width, full.height).data;
  for (let index = 0; index < pixels.data.length; index += 4) pixels.data[index + 3] = maskPixels[index];
  sourceContext.putImageData(pixels, 0, 0);
  context.drawImage(source, 0, 0);
  const [left, top, right, bottom] = state.cropBox;
  const sx = Math.round(left * full.width), sy = Math.round(top * full.height), width = Math.max(1, Math.round((right - left) * full.width)), height = Math.max(1, Math.round((bottom - top) * full.height));
  const output = document.createElement("canvas");
  output.width = width; output.height = height;
  output.getContext("2d").drawImage(full, sx, sy, width, height, 0, 0, width, height);
  return output.toDataURL("image/jpeg", .92);
}

$("#maskCropPreview").onclick = () => {
  try {
    $("#aiPreviewOriginal").src = $("#maskReferenceImage").src;
    $("#aiPreviewResult").src = buildAiPreview();
    $("#aiPreviewModal").classList.remove("hidden");
  } catch (error) { toast(error.message, true); }
};
$("#aiPreviewMask").onclick = () => preview($("#maskCanvas").toDataURL("image/png"), "识别区域");
$("#maskReference").onchange = async () => renderMaskCanvas($("#maskReference").selectedOptions[0].dataset.url, state.maskProfile.draft ? null : state.maskProfile.mask_url, state.maskProfile.draft ? null : state.maskProfile.crop_box);
$("#maskSave").onclick = async () => {
  try {
    if (!state.cropBox) throw new Error("请先框选识别范围");
    const profile = await api(`/api/batches/${state.batchId}/mask-profiles/${encodeURIComponent(state.maskCategory)}`, { method: "POST", body: JSON.stringify({ mask_data_url: $("#maskCanvas").toDataURL("image/png"), reference_file_id: Number($("#maskReference").value), crop_box: state.cropBox, name: state.maskProfile.name }) });
    state.maskProfile = profile;
    toast(state.batch?.status === "COMPLETED" ? "新识别区域已保存，可以进入自动匹配重新计算" : "识别区域已保存");
    await refreshWorkspace(false);
    const data = await api(`/api/batches/${state.batchId}/mask-profiles`);
    state.maskData = data;
    const next = data.readiness.missing_categories.find(category => category !== state.maskCategory);
    state.maskCategory = null;
    renderMaskProfileList(data);
    if (next) await openMaskCategory(next);
    else { $("#maskEditor").classList.add("hidden"); $("#maskEmpty").classList.remove("hidden"); $("#maskEmpty").textContent = "✓ 所有图片分类均已完成，可以进入自动匹配"; }
  } catch (error) { toast(error.message, true); }
};
$("#nextToMatch").onclick = () => showView("process");
$("#resetMaskForRematch").onclick = () => {
  showView("mask");
  toast("请选择问题较多的图片分类，重新调整裁剪框或手绘识别区域");
};

const filterLabels = { pending: "待确认", unmatched: "未匹配", confirmed: "已完成", all: "全部" };
const userStatus = { MATCHED: "系统已匹配", AMBIGUOUS: "需要确认", LOW_CONFIDENCE: "需要确认", UNMATCHED: "未匹配", MANUAL_CONFIRMED: "已确认" };

async function loadReview() {
  if (!state.batchId) return;
  try {
    const [data, plan] = await Promise.all([api(`/api/batches/${state.batchId}/review?filter=${state.filter}`), api(`/api/batches/${state.batchId}/rename-plan`)]);
    state.reviewData = data;
    state.renamePlan = plan;
    state.workspace = deriveWorkspaceState(state.batch, data, plan);
    const visible = new Set(data.groups.map(group => group.main.id));
    state.selectedMains = new Set([...state.selectedMains].filter(id => visible.has(id)));
    renderReviewOverview(data.summary);
    renderFilters(data.summary);
    renderUnmatched(data.unmatched_files || []);
    renderGroups(data.groups);
    renderBulkToolbar();
    renderRenamePlan(plan);
    renderStepper("review");
  } catch (error) { toast(error.message, true); }
}

function renderReviewOverview(summary) {
  $("#overviewAll").textContent = summary.all || 0;
  $("#overviewPending").textContent = summary.pending || 0;
  $("#overviewUnmatched").textContent = summary.unmatched_files || 0;
  $("#overviewConfirmed").textContent = summary.confirmed || 0;
}

function renderFilters(summary) {
  $("#filters").innerHTML = Object.entries(filterLabels).map(([key, label]) => `<button class="filter-btn ${state.filter === key ? "active" : ""}" data-filter="${key}">${label}<b>${key === "unmatched" ? summary.unmatched_files || 0 : summary[key] || 0}</b></button>`).join("");
  $$(".filter-btn").forEach(button => button.onclick = () => { state.filter = button.dataset.filter; loadReview(); });
}

function categoryStatus(slot) {
  if (slot.missing) return { icon: "○", label: "未导入", className: "muted" };
  if (!slot.files.length) return { icon: "!", label: "未匹配", className: "needs" };
  if (slot.files.every(file => file.review_status === "CONFIRMED" || file.review_status === "CORRECTED")) return { icon: "✓", label: "已确认", className: "complete" };
  if (slot.files.some(file => ["AMBIGUOUS", "LOW_CONFIDENCE", "UNMATCHED"].includes(file.status))) return { icon: "!", label: "需要确认", className: "needs" };
  return { icon: "✓", label: "系统已匹配", className: "auto" };
}

function renderGroups(groups) {
  $("#reviewGrid").innerHTML = groups.length ? groups.map(group => `
    <article class="review-product ${state.selectedMains.has(group.main.id) ? "selected" : ""}">
      <header><label class="group-select"><input type="checkbox" data-group-select="${group.main.id}" ${state.selectedMains.has(group.main.id) ? "checked" : ""} onchange="toggleGroupSelection(${group.main.id},this.checked)"> 选择</label><div><h2>${esc(group.main.basename)}</h2><small>${group.flags.confirmed ? "✓ 已完成" : "还有项目需要确认"}</small></div><button class="secondary" onclick="confirmGroup(${group.main.id})">确认整组</button></header>
      <div class="category-checklist visual-category-grid">
        <button class="category-check visual-category main-category" onclick="preview('/api/main-images/${group.main.id}/preview','主图')"><img src="${group.main.thumbnail_url}" loading="lazy" alt="${esc(group.main.basename)} 主图"><span>✓</span><b>主图</b><small>匹配基准</small></button>
        ${group.categories.map(slot => { const status = categoryStatus(slot); const file = slot.files[0]; const shortName = slot.name.replace("白底模特图", "白底图").replace("细节展示图", "细节图"); return `<button class="category-check visual-category ${status.className}" ${file ? `onclick="openMatchReview(${file.id})"` : slot.unmatched ? `onclick="showUnmatchedCategory('${esc(slot.name)}')"` : "disabled"}>${file ? `<img src="${file.thumbnail_url}" loading="lazy" alt="${esc(file.original_filename)}">` : `<div class="category-image-placeholder">${slot.unmatched ? "!" : "—"}</div>`}<span>${status.icon}</span><b>${esc(shortName)}</b><small>${status.label}</small></button>`; }).join("")}
      </div>
    </article>`).join("") : `<div class="empty-state panel">当前没有${filterLabels[state.filter] || ""}项目。${state.filter !== "all" ? '<button class="text-btn" onclick="showAllReview()">查看全部结果</button>' : ""}</div>`;
}

window.showAllReview = () => { state.filter = "all"; loadReview(); };
window.showUnmatchedCategory = category => { state.filter = "unmatched"; toast(`${category}中有图片尚未匹配，请在未匹配区人工选择主图`); loadReview(); };
window.toggleGroupSelection = (id, checked) => { checked ? state.selectedMains.add(id) : state.selectedMains.delete(id); renderBulkToolbar(); };
function renderBulkToolbar() { const ids = (state.reviewData?.groups || []).map(group => group.main.id); const selected = ids.filter(id => state.selectedMains.has(id)); $("#selectedGroupCount").textContent = `已选 ${selected.length} 组`; $("#selectAllGroups").checked = ids.length > 0 && selected.length === ids.length; $("#selectAllGroups").indeterminate = selected.length > 0 && selected.length < ids.length; $("#confirmSelectedGroups").disabled = !selected.length; $("#confirmAllVisibleGroups").disabled = !ids.length; }
$("#selectAllGroups").onchange = event => { for (const group of state.reviewData?.groups || []) event.target.checked ? state.selectedMains.add(group.main.id) : state.selectedMains.delete(group.main.id); renderGroups(state.reviewData?.groups || []); renderBulkToolbar(); };

async function bulkConfirmGroups(ids) {
  if (!ids.length || !confirm(`确认通过这 ${ids.length} 个商品组？`)) return;
  try { await api(`/api/batches/${state.batchId}/confirm-groups`, { method: "POST", body: JSON.stringify({ main_image_ids: ids }) }); state.selectedMains.clear(); toast("已确认，命名预览已同步更新"); await loadReview(); } catch (error) { toast(error.message, true); }
}
$("#confirmSelectedGroups").onclick = () => bulkConfirmGroups([...state.selectedMains]);
$("#confirmAllVisibleGroups").onclick = () => bulkConfirmGroups((state.reviewData?.groups || []).map(group => group.main.id));

function renderUnmatched(files) {
  const panel = $("#unmatchedPanel");
  panel.classList.toggle("hidden", !files.length);
  panel.innerHTML = files.length ? `<div class="panel-title"><h2>未匹配图片</h2><span class="muted">图片仍在原文件夹中，只是尚未找到可靠主图</span></div><div class="unmatched-grid">${files.map(file => `<button class="unmatched-item" onclick="openReassign(${file.id})"><img src="${file.thumbnail_url}" alt="${esc(file.original_filename)}"><div><b>${esc(file.category)}</b><small>${esc(file.original_filename)}</small><span>选择对应主图</span></div></button>`).join("")}</div>` : "";
}

function allReviewFiles() {
  if (!state.reviewData) return [];
  const files = [...(state.reviewData.unmatched_files || [])];
  for (const group of state.reviewData.groups || []) for (const category of group.categories || []) files.push(...(category.files || []));
  return [...new Map(files.map(file => [file.id, file])).values()];
}

function confidenceLabel(file) {
  if (["AMBIGUOUS", "LOW_CONFIDENCE", "UNMATCHED"].includes(file.status) || file.assigned_score == null) return "不确定";
  if (file.assigned_score >= .9) return "高";
  return "一般";
}

window.openMatchReview = id => {
  const file = allReviewFiles().find(item => item.id === id);
  if (!file) return;
  state.reviewFile = file;
  const isPsd = file.category === "PSD";
  $("#reviewModalTitle").textContent = `${file.category} · ${file.original_filename}`;
  $("#reviewSourceLabel").textContent = isPsd ? "PSD 预览" : "待匹配原图";
  $("#reviewSourceImage").src = file.preview_url;
  $("#reviewMainImage").src = file.matched_main_id ? `/api/main-images/${file.matched_main_id}/preview` : "";
  $("#reviewConfidence").textContent = `匹配度：${confidenceLabel(file)}`;
  $("#reviewConfidence").dataset.level = confidenceLabel(file);
  const tabs = [{ label: "原图", url: file.preview_url }];
  if (file.derived?.mask) tabs.push({ label: "识别区域", url: file.derived.mask.url });
  if (file.derived?.masked_crop) tabs.push({ label: "AI 识别画面", url: file.derived.masked_crop.url });
  $("#reviewSourceTabs").innerHTML = tabs.map((tab, index) => `<button class="${index === 0 ? "active" : ""}" onclick="switchReviewSource('${tab.url}',this)">${tab.label}</button>`).join("");
  $("#matchDetailsBody").innerHTML = `<p><b>匹配方式：</b>${isPsd ? "预览图匹配" : "AI 画面匹配"}</p><p><b>相似度：</b>${pct(file.assigned_score)}</p><p><b>Top1：</b>${esc(file.original_top1?.basename || "—")} ${pct(file.original_top1?.score)}</p><p><b>Top2：</b>${pct(file.original_top2?.score)}</p><p><b>Margin：</b>${pct(file.top1_top2_margin)}</p><p><b>Assigned Rank：</b>${file.assigned_rank ?? "—"}</p><p><b>算法版本：</b>${esc(file.algorithm_version || "—")}</p>${isPsd ? `<p><b>pHash 距离：</b>${file.phash_distance ?? "—"}</p>` : ""}`;
  $("#matchDetails").open = false;
  $("#reviewCorrect").disabled = !file.matched_main_id;
  $("#reviewModal").classList.remove("hidden");
};
window.switchReviewSource = (url, button) => { $("#reviewSourceImage").src = url; $$("#reviewSourceTabs button").forEach(item => item.classList.toggle("active", item === button)); };
$("#reviewCorrect").onclick = async () => { if (!state.reviewFile) return; await confirmFile(state.reviewFile.id); $("#reviewModal").classList.add("hidden"); };
$("#reviewChange").onclick = () => { const id = state.reviewFile?.id; $("#reviewModal").classList.add("hidden"); if (id) openReassign(id); };
$("#reviewLater").onclick = () => $("#reviewModal").classList.add("hidden");

window.confirmGroup = async id => { try { await api(`/api/main-images/${id}/confirm`, { method: "POST" }); toast("当前商品已确认"); await loadReview(); } catch (error) { toast(error.message, true); } };
window.confirmFile = async id => { try { await api(`/api/files/${id}/confirm`, { method: "POST", body: "{}" }); toast("匹配已确认"); await loadReview(); } catch (error) { toast(error.message, true); } };

window.openReassign = async id => {
  let file = allReviewFiles().find(item => item.id === id);
  if (!file) {
    const data = await api(`/api/batches/${state.batchId}/review?filter=all`);
    const candidates = [...(data.unmatched_files || [])];
    for (const group of data.groups) for (const category of group.categories) candidates.push(...category.files);
    file = candidates.find(item => item.id === id);
  }
  if (!file) return;
  state.reassignFile = file;
  $("#reassignImage").src = file.preview_url;
  $("#reassignFilename").textContent = file.original_filename;
  $("#reassignCurrent").textContent = file.matched_main_basename ? `当前匹配：${file.matched_main_basename}` : "当前：未匹配";
  $("#mainSearch").value = "";
  $("#reassignModal").classList.remove("hidden");
  $("#candidateList").innerHTML = '<span class="muted">正在加载全部球队…</span>';
  renderCandidates(await api(`/api/batches/${state.batchId}/main-images/search?q=&exclude_confirmed=true`), "#candidateList");
};

function renderCandidates(items, target) { $("#candidateCount").textContent = `${items.length} 支可选球队`; $(target).innerHTML = items.map(item => { const mainId = item.main_image_id || item.id; const current = mainId === state.reassignFile?.matched_main_id; return `<button class="candidate ${current ? "current" : ""}" onclick="chooseMain(${mainId})"><img src="${item.thumbnail_url}" loading="lazy" alt="${esc(item.basename)}"><strong>${esc(item.basename)}</strong><small>${current ? "当前匹配" : "点击替换"}</small></button>`; }).join("") || '<span class="muted">没有可替换球队，其他球队组均已审核通过</span>'; }
window.chooseMain = async mainId => { try { const result = await api(`/api/files/${state.reassignFile.id}/reassign`, { method: "POST", body: JSON.stringify({ main_image_id: mainId }) }); $("#reassignModal").classList.add("hidden"); toast(result.swapped_file_id ? "已替换，并自动交换原有归属" : "已更换并确认主图"); await loadReview(); } catch (error) { toast(error.message, true); } };
$("#searchMainBtn").onclick = async () => renderCandidates(await api(`/api/batches/${state.batchId}/main-images/search?q=${encodeURIComponent($("#mainSearch").value.trim())}&exclude_confirmed=true`), "#candidateList");
$("#mainSearch").onkeydown = event => { if (event.key === "Enter") $("#searchMainBtn").click(); };
$("#refreshReview").onclick = loadReview;

const renameStatusNames = { READY: "将重命名", BLOCKED: "需要确认", FILE_MISSING: "源文件不可用", NOOP: "名称已正确", RENAME_CONFLICT: "命名冲突", RENAMED: "已完成", IGNORED: "多余文件（保持原名）" };
function renderRenamePlan(plan) {
  const plans = plan?.plans || [];
  const blockers = plans.filter(item => !["READY", "NOOP", "RENAMED", "IGNORED"].includes(item.rename_status));
  const ready = plans.filter(item => item.rename_status === "READY");
  $("#showUnprocessed").classList.toggle("hidden", !blockers.length);
  $("#renameBusinessStatus").textContent = blockers.length ? `还有 ${blockers.length} 个文件需要处理` : plans.length ? `审核完成，即将重命名 ${ready.length} 个文件` : "完成必要审核后即可预览命名结果。";
  $("#executeRenamePlan").disabled = blockers.length > 0 || !ready.length;
  if (!plans.length) { $("#renamePlanBody").innerHTML = '<span class="muted">点击“预览命名结果”查看即将修改的文件名。</span>'; return; }
  const chips = Object.entries(plan.counts || {}).map(([status, count]) => `<span class="plan-chip status-${status.toLowerCase()}">${esc(renameStatusNames[status] || status)} <b>${count}</b></span>`).join("");
  $("#renamePlanBody").innerHTML = `<div class="rename-summary">${chips}</div><details class="advanced-details"><summary>查看命名明细</summary><div class="rename-table-wrap"><table class="rename-table"><thead><tr><th>分类</th><th>当前名称</th><th>新名称</th><th>状态</th></tr></thead><tbody>${plans.map(item => `<tr><td>${esc(item.category)}</td><td>${esc(item.original_filename)}</td><td>${esc(item.target_filename || "—")}</td><td>${esc(renameStatusNames[item.rename_status] || item.rename_status)}</td></tr>`).join("")}</tbody></table></div></details>`;
}
$("#generateRenamePlan").onclick = async () => { try { state.renamePlan = await api(`/api/batches/${state.batchId}/rename-plan`, { method: "POST" }); renderRenamePlan(state.renamePlan); toast("命名预览已更新，尚未修改文件"); } catch (error) { toast(error.message, true); } };
$("#showUnprocessed").onclick = () => { state.filter = "pending"; loadReview(); window.scrollTo({ top: $("#filters").offsetTop - 90, behavior: "smooth" }); };
$("#executeRenamePlan").onclick = () => { if (state.renamePlan?.plan_group) $("#renameConfirmModal").classList.remove("hidden"); };
$("#confirmExecuteRename").onclick = async () => { try { state.renamePlan = await api(`/api/batches/${state.batchId}/rename-plan/execute`, { method: "POST", body: JSON.stringify({ plan_group: state.renamePlan.plan_group }) }); $("#renameConfirmModal").classList.add("hidden"); renderRenamePlan(state.renamePlan); const cleanup = state.renamePlan.workspace_cleanup; toast(cleanup && !cleanup.removed ? "文件重命名已完成，但临时文件清理失败，请稍后重试" : "文件重命名已完成，临时文件已清理"); await loadReview(); } catch (error) { toast(error.message, true); } };

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
