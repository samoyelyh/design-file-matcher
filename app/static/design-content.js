const $ = selector => document.querySelector(selector);
const fields = {
  main_subject: "主体元素", secondary_subjects: "辅助元素", visible_text: "可见文字",
  team_names: "球队", league_names: "联盟", animal_names: "动物", sports_categories: "体育项目",
  pattern_types: "图案类型", theme_candidates: "主题候选", theme_types: "主题类型",
  subtheme_candidates: "子主题候选", product_category_hints: "品类线索",
  uncertain_items: "不确定项", evidence: "视觉证据",
};
const packageFields = { product_name: "产品", target_category: "目标品类", designer: "设计师" };
const statusNames = { PENDING: "待识别", FAILED: "识别失败", RECOGNIZED: "已识别", CONFIRMED: "人工已确认",
  QUEUED: "排队", RUNNING: "识别中", COMPLETE: "识别完成", PARTIAL: "部分完成", INTERRUPTED: "已中断" };
const state = { batch: null, data: null, runs: [], selected: new Set(), capability: null, timer: null, submitting: false };
const esc = value => String(value ?? "").replace(/[&<>"']/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]);
const names = list => (list || []).join(" · ") || "—";
const root = () => `/api/design-content/batches/${state.batch}`;
const running = () => ["QUEUED", "RUNNING"].includes(state.runs[0]?.status);

async function api(path, method = "GET", body) {
  const response = await fetch(path, { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(typeof data.detail === "string" ? data.detail : `操作失败（${response.status}）`);
  }
  return response.json();
}

function message(text) {
  $("#contentMessage").textContent = text;
  $("#contentMessage").hidden = !text;
}

function openSection(section) {
  document.querySelectorAll("[data-section]").forEach(button => {
    const active = button.dataset.section === section;
    button.setAttribute("aria-pressed", active);
    $(`#section-${button.dataset.section}`).hidden = !active;
  });
}

function renderPackage() {
  const info = state.data.package;
  $("#currentPackage").textContent = info.package_name;
  for (const key of Object.keys(packageFields)) $("#packageForm").elements[key].value = info[key] || "";
  $("#parseConfidence").textContent = { HIGH: "已解析", PARTIAL: "部分解析", UNKNOWN: "待补充" }[info.parse_confidence];
  $("#uncertainPackage").textContent = info.uncertain_fields.length ? `待补充：${info.uncertain_fields.map(key => packageFields[key]).join("、")}` : "";
}

function visibleGroups() {
  const filter = $("#statusFilter").value;
  return state.data.design_groups.filter(group => filter === "all" || group.recognition_status === filter ||
    (filter === "review" && group.review_status === "PENDING") || (filter === "confirmed" && group.review_status === "CONFIRMED"));
}

function chip(status) {
  return `<span class="content-chip" data-state="${esc(status)}">${esc(statusNames[status] || status)}</span>`;
}

function reviewLabel(group) {
  if (group.review_status === "CONFIRMED") return chip("CONFIRMED");
  if (group.review_status === "PENDING") return `<small class="muted">待人工确认</small>`;
  return "";
}

function renderSelection() {
  const groups = visibleGroups();
  const count = groups.filter(row => state.selected.has(row.main_image_id)).length;
  $("#selectVisible").checked = count > 0 && count === groups.length;
  $("#selectVisible").indeterminate = count > 0 && count < groups.length;
  $("#selectedMains").textContent = state.selected.size ? `已选 ${state.selected.size} 张` : "默认识别待处理主图";
  $("#recognizeBatch").disabled = state.submitting || running() || !state.capability?.ready;
  $("#mainCards").querySelectorAll("[data-rerun]").forEach(button => {
    button.disabled = state.submitting || running() || !state.capability?.ready;
  });
}

function renderMains() {
  $("#mainCards").innerHTML = visibleGroups().map(group => {
    const id = group.main_image_id;
    const content = group.recognized_content;
    const details = content ? [
      ["主体", names(content.main_subject)], ["辅助", names(content.secondary_subjects)], ["文字", names(content.visible_text)],
      ["球队 / 联盟", names([...(content.team_names || []), ...(content.league_names || [])])],
      ["动物 / 图案", names([...(content.animal_names || []), ...(content.pattern_types || [])])],
      ["主题", names(content.theme_candidates)], ["置信度", content.confidence == null ? "—" : `${Math.round(content.confidence * 100)}%`],
      ["不确定项", names(content.uncertain_items)],
    ].map(([label, value]) => `<dt>${label}</dt><dd>${esc(value)}</dd>`).join("") : "";
    return `<article class="content-card">
      <a href="/api/main-images/${id}/preview" target="_blank" rel="noopener"><img src="/api/main-images/${id}/thumbnail" loading="lazy" alt="${esc(group.main_image.filename)}"></a>
      <div class="content-card-body"><div class="content-actions"><input type="checkbox" data-select-main="${id}" aria-label="选择 ${esc(group.main_image.filename)}" ${state.selected.has(id) ? "checked" : ""}>${chip(group.recognition_status)}${reviewLabel(group)}</div>
      <h3>${esc(group.main_image.filename)}</h3><dl>${details}</dl>
      ${group.recognition_error ? `<p role="status">${esc(group.recognition_error)}</p>` : ""}
      <div class="content-actions"><button class="secondary" data-edit="${id}">查看与审核</button><button class="secondary" data-rerun="${id}">${group.recognition_status === "PENDING" ? "单张识别" : "重新识别"}</button></div></div></article>`;
  }).join("") || `<p class="muted">当前筛选没有主图。</p>`;
  renderSelection();
}

function assetTable(assets) {
  if (!assets.length) return "<p class=muted>暂无关联图片。</p>";
  return `<div class="content-table"><table><thead><tr><th>文件</th><th>分类</th><th>匹配方式</th><th>审核状态</th></tr></thead><tbody>${assets.map(item =>
    `<tr><td title="${esc(item.relative_path)}">${esc(item.original_filename)}</td><td>${esc(item.category)}${item.category_ignored ? "（已忽略）" : ""}</td><td>${esc(item.match_method || "—")}</td><td>${item.association_confirmed ? "已确认" : "未确认"} · ${esc(item.review_status || "—")}</td></tr>`).join("")}</tbody></table></div>`;
}

function renderGroups() {
  document.querySelectorAll("[data-format]").forEach(link => { link.href = `${root()}/export/${link.dataset.format}`; });
  const previous = new Set([...$("#groupRows").querySelectorAll("details[open]")].map(row => row.dataset.group));
  $("#groupRows").innerHTML = state.data.design_groups.map(group => {
    const id = group.main_image_id;
    const types = [...new Set(group.assets.map(item => item.category))].join("、") || "—";
    const confirmed = group.assets.filter(item => item.association_confirmed).length;
    return `<details class="content-group" data-group="${id}" ${previous.has(String(id)) ? "open" : ""}>
      <summary><img src="/api/main-images/${id}/thumbnail" loading="lazy" alt=""><div><strong>${esc(group.main_image.filename)}</strong><small>${esc(names(group.recognized_content?.main_subject))}</small></div>
      <div class="group-categories"><strong>${group.assets.length} 张关联 · ${confirmed} 已确认</strong><small>${esc(types)}</small></div><div>${chip(group.recognition_status)} ${reviewLabel(group)}</div></summary>${assetTable(group.assets)}</details>`;
  }).join("");
  if (state.data.unassigned_assets.length) {
    $("#groupRows").insertAdjacentHTML("beforeend", `<details class="content-group"><summary>未归属图片 ${state.data.unassigned_assets.length} 张</summary>${assetTable(state.data.unassigned_assets)}</details>`);
  }
}

function renderRun() {
  const summary = $("#runSummary");
  const run = state.runs[0];
  summary.hidden = !run;
  if (!run) return;
  const resolved = run.succeeded + run.failed;
  const errors = [...new Set(run.sheets.map(sheet => sheet.error).filter(Boolean))];
  summary.innerHTML = `<strong>${esc(statusNames[run.status])}</strong><progress max="${run.main_image_ids.length}" value="${resolved}"></progress>
    <span>成功 ${run.succeeded} · 失败 ${run.failed} · 未处理 ${run.remaining} · Terra 调用 ${run.real_calls} / ${run.max_calls}${run.unknown_usage ? " · 存在用量未知的请求" : ""}</span>
    ${errors.map(text => `<p>${esc(text)}</p>`).join("")}
    ${state.runs.length > 1 ? `<details><summary>历史任务</summary><ul>${state.runs.slice(1).map(item =>
      `<li>${esc(new Date(item.created_at).toLocaleString("zh-CN", {hour12:false}))} · ${esc(statusNames[item.status])} · ${item.succeeded}/${item.main_image_ids.length} 张 · ${item.real_calls} 次调用${item.unknown_usage ? "（部分未知）" : ""}</li>`).join("")}</ul></details>` : ""}`;
}

async function refresh(batch, updatePackage = false) {
  const path = `/api/design-content/batches/${batch}`;
  const [data, runs] = await Promise.all([api(path), api(`${path}/runs`)]);
  if (state.batch !== batch) return;
  state.data = data;
  state.runs = runs;
  if (updatePackage) renderPackage();
  $("#workspacePanels").hidden = false;
  const labels = { main_images: "主图总数", recognized: "识别完成", pending: "待识别", failed: "识别失败", awaiting_review: "待人工确认", confirmed: "内容已确认" };
  $("#summary").innerHTML = Object.entries(labels).map(([key, label]) => `<div><span>${label}</span><strong>${data.counts[key]}</strong></div>`).join("");
  renderMains();
  renderGroups();
  renderRun();
  clearTimeout(state.timer);
  if (running()) state.timer = setTimeout(() => refresh(batch).catch(error => message(error.message)), 2000);
}

async function openBatch(batch) {
  clearTimeout(state.timer);
  state.batch = batch;
  state.selected.clear();
  state.data = null;
  state.runs = [];
  $("#workspacePanels").hidden = true;
  $("#matchingModule").href = `/#batch=${batch}&view=create`;
  history.replaceState(null, "", `?batch=${batch}`);
  await refresh(batch, true);
}

async function start(ids, force = false) {
  if (state.submitting || running()) return;
  state.submitting = true;
  const batch = state.batch;
  renderSelection();
  try {
    await api(`${root()}/runs`, "POST", ids ? { main_image_ids: ids, force } : {});
    message("");
    state.selected.clear();
    await refresh(batch);
  } catch (error) {
    message(error.message);
  } finally {
    state.submitting = false;
    if (state.data) renderSelection();
  }
}

function openEditor(id) {
  const group = state.data.design_groups.find(row => row.main_image_id === id);
  const content = group.recognized_content || {};
  $("#editorTitle").textContent = group.main_image.filename;
  $("#rawDetails").open = false;
  $("#rawOutput").textContent = group.ai_content ? JSON.stringify(group.ai_content, null, 2) : "暂无 AI 结果";
  $("#editorMessage").textContent = group.review_status === "CONFIRMED" ? "正在查看人工确认结果" : "";
  $("#editorFields").innerHTML = `${Object.entries(fields).map(([key, label]) =>
    `<label>${label}<textarea name="${key}" placeholder="每行一项">${esc((content[key] || []).join("\n"))}</textarea></label>`).join("")}
    <label>识别置信度（0–1）<input type="number" name="confidence" min="0" max="1" step="any" value="${esc(content.confidence)}"></label>`;
  $("#editorForm").dataset.id = id;
  $("#contentEditor").showModal();
}

document.querySelectorAll("[data-section]").forEach(button => { button.onclick = () => openSection(button.dataset.section); });
$("#batchSelect").onchange = event => openBatch(Number(event.target.value)).catch(error => message(error.message));
$("#statusFilter").onchange = renderMains;
$("#selectVisible").onchange = event => {
  for (const row of visibleGroups()) {
    if (event.target.checked) state.selected.add(row.main_image_id);
    else state.selected.delete(row.main_image_id);
  }
  renderMains();
};
$("#mainCards").onchange = event => {
  const id = Number(event.target.dataset.selectMain);
  if (!id) return;
  if (event.target.checked) state.selected.add(id);
  else state.selected.delete(id);
  renderSelection();
};
$("#mainCards").onclick = event => {
  if (event.target.dataset.edit) openEditor(Number(event.target.dataset.edit));
  if (event.target.dataset.rerun) start([Number(event.target.dataset.rerun)], true);
};
$("#recognizeBatch").onclick = () => start(state.selected.size ? [...state.selected] : null);
$("#packageForm").onsubmit = async event => {
  event.preventDefault();
  try {
    const values = Object.fromEntries(new FormData(event.target));
    const changes = Object.fromEntries(Object.entries(values).filter(([key, value]) => value.trim() !== (state.data.package[key] || "")));
    if (Object.keys(changes).length) state.data.package = await api(`${root()}/package`, "PATCH", changes);
    renderPackage();
    message("设计包信息已保存。");
  } catch (error) { message(error.message); }
};
$("#closeEditor").onclick = () => $("#contentEditor").close();
$("#editorForm").onsubmit = async event => {
  event.preventDefault();
  const form = new FormData(event.target);
  const payload = Object.fromEntries(Object.keys(fields).map(key => [key, form.get(key).split("\n").map(line => line.trim()).filter(Boolean)]));
  payload.confidence = form.get("confidence") === "" ? null : Number(form.get("confidence"));
  try {
    await api(`${root()}/mains/${event.target.dataset.id}/confirmation`, "PUT", payload);
    $("#contentEditor").close();
    await refresh(state.batch);
    message("人工确认已保存。");
  } catch (error) { $("#editorMessage").textContent = error.message; }
};

async function init() {
  const [rows, capability] = await Promise.all([api("/api/design-content/batches"), api("/api/design-content/capabilities")]);
  state.capability = capability;
  if (!rows.length) {
    message("没有已扫描的设计包。请先到图片匹配与命名导入设计包。");
    return;
  }
  const wanted = Number(new URLSearchParams(location.search).get("batch"));
  const id = rows.some(row => row.id === wanted) ? wanted : rows[0].id;
  const select = $("#batchSelect");
  select.innerHTML = rows.map(row => `<option value="${row.id}">#${row.id} ${esc(row.name)}</option>`).join("");
  select.value = id;
  select.disabled = false;
  await openBatch(id);
  if (!capability.ready) message(capability.error);
}

init().catch(error => message(error.message));
