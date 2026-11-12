'use strict';
const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const labels={PENDING:'未识别',MATCHED:'待确认',CONFIRMED:'已确认',REVIEW:'待复核',UNMATCHED:'未匹配',ERROR:'异常',SCANNED:'已导入',RUNNING:'运行中',COMPLETE:'完成',PARTIAL:'结果不完整',FAILED:'失败',SUCCESS:'成功',PAUSED:'已暂停',BUDGET_PAUSED:'预算已用完',CANCELLED:'已停止',COMPLETE_WITH_ERRORS:'完成，有失败项',PAUSED_UNKNOWN:'中断待恢复',QUOTA_OR_AUTH_PAUSED:'额度或登录异常',EXECUTED:'已执行',EXECUTED_WITH_ERRORS:'部分失败',PREVIEW:'预览',INTERRUPTED:'已中断',ROLLED_BACK:'已回退',MATCHING:'匹配中',DRAFT:'草稿',REVIEWING:'审核中'};
const statusOrder=['','MATCHED','REVIEW','ERROR','UNMATCHED','CONFIRMED','PENDING'];

let batch=null,state=null,plan=null,busy=false,replayMode=false;
let selected=new Set(),active=null,anchor=null,visibleRows=[],pageRows=[],editing=null,mappingDraft=null,pendingAssign=null,cmpMode='pair',runFolders=new Set(),activeSource=null;
let view={screen:'import',layout:'matrix',folder:'',status:'',search:'',sourceSearch:'',page:0,pageSize:16,scope:'filtered',thumb:1,runOpen:true,leftOpen:true,rightOpen:true,namingScope:'all',namingMode:'COPY_AND_RENAME',concurrency:2};

/* ---------- helpers ---------- */
function load(key,fallback){try{return JSON.parse(localStorage.getItem(key))??fallback}catch{return fallback}}
function save(){if(batch)localStorage.setItem('vm3.ux.'+batch,JSON.stringify({view,selected:[...selected],active,runFolders:[...runFolders],editing,mappingDraft:editing||mappingDraft}))}
function toast(text,{error=false,action=null,timeout=error?7000:4200}={}){
  const el=document.createElement('div');el.className='toast'+(error?' error':'');const span=document.createElement('span');span.textContent=text;el.append(span);
  if(action){const b=document.createElement('button');b.textContent=action.label;b.onclick=()=>{el.remove();action.fn()};el.append(b)}
  const close=document.createElement('button');close.textContent='✕';close.onclick=()=>el.remove();el.append(close);
  $('#toasts').append(el);setTimeout(()=>el.remove(),timeout);
}
async function api(path,body){const r=await fetch('/api'+path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});if(!r.ok){let e;try{e=await r.json()}catch{e={detail:r.statusText}}throw Error(typeof e.detail==='string'?e.detail:JSON.stringify(e.detail))}return r.json()}
async function action(fn){if(busy){toast('当前操作正在保存，请稍候。');return}busy=true;$$('[data-mutation]').forEach(x=>x.disabled=true);try{await fn()}catch(e){toast(e.message,{error:true})}finally{busy=false;$$('[data-mutation]').forEach(x=>x.disabled=false);updateBulk()}}
const folders=()=>state?.folders.filter(f=>f.role==='TARGET')||[];
const sources=()=>state?.groups.map(g=>g.source)||[];
const match=id=>state?.matches.find(m=>m.target_id===Number(id));
const image=id=>state.images.find(i=>i.id===Number(id));
const folderName=id=>state.folders.find(f=>f.id===id)?.label||'';
const selectedRows=()=>state.matches.filter(m=>selected.has(m.target_id));
const badge=st=>`<span class="badge ${esc(st)}">${esc(labels[st]||st)}</span>`;
const dot=st=>`<span class="dot ${esc(st)}" title="${esc(labels[st]||st)}"></span>`;
const thumb=(id,extra='')=>`<img loading="lazy" decoding="async" src="/api/images/${id}" ${extra}>`;
const fmtTime=s=>{s=Math.max(0,Math.round(s));return s<60?s+' 秒':Math.floor(s/60)+' 分 '+(s%60)+' 秒'};

/* ---------- boot & batch ---------- */
async function listBatches(){const bs=await api('/batches');$('#batchPicker').innerHTML='<option value="">选择批次…</option>'+bs.map(b=>`<option value="${b.id}">${esc(b.name)} · ${esc(labels[b.status]||b.status)}</option>`).join('');$('#batchPicker').value=batch||'';
  $('#recentBatches').innerHTML=bs.slice(0,6).map(b=>`<button data-open-batch="${b.id}">${esc(b.name)} <span class="muted">${esc(labels[b.status]||b.status)}</span></button>`).join('')}
async function createBatch(){const b=await api('/batches',{name:$('#batchDraft').value.trim()||'新素材批次'});await openBatch(b.id);toast('批次已创建，请登记主图与目标目录。')}
function showWelcome(){$('#welcome').hidden=false;$('#workbench').hidden=true;$('#bulkBar').hidden=true;$('#batchDraft').focus()}
async function openBatch(id){batch=id;localStorage.vm3Batch=id;const url=new URL(location.href);if(url.searchParams.get('batch')!==id){const existed=url.searchParams.has('batch');url.searchParams.set('batch',id);history[existed?'pushState':'replaceState']({batch:id},'',url)}
  const saved=load('vm3.ux.'+id,{});view={...view,...saved.view};selected=new Set(saved.selected||[]);active=saved.active||null;activeSource=null;runFolders=new Set(saved.runFolders||[]);editing=saved.editing||null;mappingDraft=saved.mappingDraft||saved.editing||null;plan=null;
  await refresh();await listBatches();
  view.screen=state.folders.length?(view.screen==='import'?'review':view.screen):'import';
  showScreen(view.screen);
  if(view.screen==='review')renderReview();else if(view.screen==='import')renderScanResults();
  if(editing&&editing.ids.every(id=>match(id)))openAssign(editing.ids,true);
  else if(editing){editing=null;mappingDraft=null}}

/* ---------- screen switching ---------- */
function showScreen(screen){view.screen=screen;
  $('#welcome').hidden=true;$('#workbench').hidden=false;
  $('#importView').hidden=screen!=='import';$('#namingView').hidden=screen!=='naming';
  $('#reviewBar').hidden=screen!=='review';$('#selectBar').hidden=screen!=='review';
  $('#reviewArea').hidden=screen!=='review';$('#runPanel').hidden=screen!=='review'||!view.runOpen;
  $('#sideLeft').hidden=screen==='import';$('#toggleLeft').hidden=screen==='import';
  syncPanels();updateBulk();save()}

/* ---------- refresh & render root ---------- */
async function refresh(poll=false){if(!batch)return;const previous=state;state=await api('/batches/'+batch);
  selected=new Set([...selected].filter(id=>match(id)));if(active&&!match(active))active=null;
  if(previous?.id===state.id&&previous.runs.some(r=>r.state==='RUNNING')&&!state.runs.some(r=>r.state==='RUNNING')){
    toast('本轮自动匹配已结束，可继续审核。');view.runOpen=false}
  const src=state.folders.find(f=>f.role==='SOURCE');
  $('#batchStats').textContent=`${src?.count||0} 张主图 · ${folders().length} 个目标目录 · ${Object.entries(state.counts).map(([k,v])=>(labels[k]||k)+' '+v).join(' / ')}`;
  const opt=$(`#batchPicker option[value="${batch}"]`);if(opt)opt.textContent=state.name+' · '+(labels[state.status]||state.status);
  $('#goNaming').disabled=!state.matches.length;$('#startMatchTop').disabled=!folders().length;
  renderProgress();renderSources();renderFolderNav();
  if(poll&&previous?.id===state.id&&previous.revision===state.revision){updateBulk();save();return}
  if(view.screen==='review')renderReview();
  if(plan&&plan.revision!==state.revision){plan=null;if(view.screen==='naming')$('#namingPreview').innerHTML='<p class="scope-note">审核结果已变化，旧预览已失效，请点击「更新预览」。</p>'}
  if(view.rightOpen&&active)renderDetail();
  updateBulk();save()}

/* ---------- top progress chip ---------- */
function renderProgress(){const runs=state.runs||[],activeRun=runs.find(r=>['RUNNING','BUDGET_PAUSED','PAUSED','PAUSED_UNKNOWN','QUOTA_OR_AUTH_PAUSED'].includes(r.state));
  const jobs=state.jobs||[],done=jobs.filter(j=>j.state==='SUCCESS').length,running=jobs.filter(j=>j.state==='RUNNING').length,failed=folders().filter(f=>['ERROR','PARTIAL'].includes(f.state)).length;
  const chip=$('#progressChip');
  if(activeRun){chip.hidden=false;const elapsed=activeRun.started?((new Date(activeRun.finished||Date.now())-new Date(activeRun.started))/1000):0;
    chip.textContent=`匹配中 ${done}/${jobs.length} · 已耗时 ${fmtTime(elapsed)}`}
  else chip.hidden=true;
  if(view.screen!=='review')return;
  $('#runPanel').hidden=!view.runOpen;
  const total=state.matches.length;
  $('#runTotals').textContent=`共 ${total} 张目标图 · ${folders().length} 目录 · 完成 ${done} · 运行 ${running} · 异常 ${failed}`;
  $('#folderProgress').innerHTML=folders().map(f=>`<div class="fp-row"><label class="check"><input type="checkbox" class="run-folder" value="${f.id}" ${runFolders.has(f.id)?'checked':''}></label>${dot(f.state)}<span>${esc(f.label)} · ${f.count} 张</span>${f.metrics.cache_hit?'<span class="muted">缓存回放</span>':''}${f.metrics.wall_seconds!==undefined?`<span class="muted">${f.metrics.wall_seconds}s</span>`:''}${f.error?`<span class="err">${esc(f.error)}</span>`:''}</div>`).join('');
  $('#runControls').innerHTML=runs.map(r=>`<div class="rc-row">${badge(r.state)}<span>${r.used_calls} / ${r.max_calls} 次预算 · 并发 ${r.concurrency}</span>${r.started?`<span class="muted">${fmtTime((new Date(r.finished||Date.now())-new Date(r.started))/1000)}</span>`:''}${['RUNNING','PAUSED','BUDGET_PAUSED','PAUSED_UNKNOWN','QUOTA_OR_AUTH_PAUSED'].includes(r.state)?`<button data-run="${r.id}" data-control="${r.state==='RUNNING'?'pause':'resume'}">${r.state==='RUNNING'?'暂停领取':'继续（原预算）'}</button><button data-run="${r.id}" data-control="cancel">停止领取</button>`:''}</div>`).join('');
  const n=runScopeIds().length;$('#startMatch').textContent=`开始匹配 ${n} 个目录`;$('#retryFailed').disabled=busy||!failed;
  $('#matchPlan').textContent=`本范围 ${n} 个目录，首轮最多 ${n} 次请求。预算包含失败；缓存回放不算新调用。不会自动扩大预算。${replayMode?' 当前是离线回放服务，不产生真实模型调用。':''}`}
function runScopeIds(override){const sel=folders().filter(f=>runFolders.has(f.id));const base=override==='failed'?folders().filter(f=>['ERROR','PARTIAL'].includes(f.state)):sel.length?sel:folders();return base.map(f=>f.id)}
async function startMatch(scope){const ids=runScopeIds(scope);if(!ids.length)throw Error('没有可执行的目录，请先导入目标目录。');if(!$('#approveCalls').checked||Number($('#budget').value)<1)throw Error('请填写并勾选确认本次模型调用预算。');
  await api(`/batches/${batch}/match`,{scope:scope||'all',folder_ids:ids,max_calls:Number($('#budget').value),concurrency:Number($('#concurrency').value),confirmed_budget:true});
  $('#approveCalls').checked=false;view.runOpen=true;showScreen('review');await refresh();toast(`已启动 ${ids.length} 个目录的自动匹配，审核可同步进行。`)}

/* ---------- left sidebar ---------- */
function renderSources(){const q=view.sourceSearch.toLowerCase();const total=Math.max(1,folders().length);
  $('#sourceList').innerHTML=sources().filter(s=>!q||(s.name+s.local_id).toLowerCase().includes(q)).map(s=>{
    const ms=state.matches.filter(m=>m.assigned_source_id===s.id);const done=new Set(ms.filter(m=>m.status==='CONFIRMED').map(m=>m.folder_id)).size;
    return `<div class="src-item ${activeSource===s.id?'active':''}" data-source="${s.id}" data-drop-source="${s.id}" title="${esc(s.name)}">${thumb(s.id)}<span class="src-name">${esc(s.local_id+' '+s.name)}</span><span class="src-progress ${done>=total?'done':''}">${done}/${total}</span></div>`}).join('')||'<p class="muted">尚未导入主图</p>'}
function renderFolderNav(){$('#folderNav').innerHTML=folders().map(f=>`<button data-focus-folder="${f.id}" class="${view.folder===f.id&&view.layout==='folder'?'active':''}">${dot(f.state)}<span style="flex:1;text-align:left;margin-left:4px">${esc(f.label)}</span><span class="cnt">${f.count}</span></button>`).join('')}

/* ---------- filtering / paging ---------- */
function filtered(){let rows=state.matches;const q=view.search.toLowerCase();
  if(view.layout==='folder'&&view.folder)rows=rows.filter(m=>m.folder_id===view.folder);else if(view.folder)rows=rows.filter(m=>m.folder_id===view.folder);
  if(view.status)rows=rows.filter(m=>m.status===view.status);
  if(q)rows=rows.filter(m=>[m.target.name,m.target.local_id,m.source?.name,m.source?.local_id].filter(Boolean).some(s=>s.toLowerCase().includes(q)));
  return rows}
function scopeRows(){if(view.scope==='all')return state.matches;if(view.scope==='folder'){if(!view.folder)throw Error('请先在左栏或筛选中选择一个目标目录');return state.matches.filter(m=>m.folder_id===view.folder)}if(view.scope==='page')return pageRows;return visibleRows}

/* ---------- review render ---------- */
function card(m,focus=false){const on=selected.has(m.target_id);
  return `<article class="tcard ${on?'selected':''} ${active===m.target_id?'current':''}" data-target="${m.target_id}" draggable="true" title="${esc(m.target.name)} · ${esc(labels[m.status]||m.status)}${m.locked?' · 人工锁定':''}">
  <input type="checkbox" class="sel" data-sel="${m.target_id}" ${on?'checked':''} tabindex="-1" aria-label="选择 ${esc(m.target.name)}">
  ${m.locked?'<span class="lock">🔒</span>':''}
  <div class="t-top">${dot(m.status)}<span class="t-name">${esc(m.target.local_id+' '+m.target.name)}</span></div>
  ${focus?'<div class="pair">'+(m.source?thumb(m.source.id):'<div class="empty-cell">未分配</div>')+thumb(m.target_id)+'</div>':thumb(m.target_id)}
  <div class="t-new">→ ${esc(m.new_name||(m.source?'':'未分配'))}</div>
</article>`}
function renderReview(){visibleRows=filtered();const cols=folders().filter(f=>!view.folder||f.id===view.folder);
  document.documentElement.style.setProperty('--thumb',['104px','150px','210px'][view.thumb]||'150px');
  const area=$('#reviewArea');const prevTop=area.querySelector('.matrix')?.scrollTop??view._top??0,prevLeft=area.querySelector('.matrix')?.scrollLeft??view._left??0;
  let total=0,content='';
  if(view.layout==='folder'){total=Math.max(1,Math.ceil(visibleRows.length/view.pageSize));view.page=Math.max(0,Math.min(view.page,total-1));
    pageRows=visibleRows.slice(view.page*view.pageSize,(view.page+1)*view.pageSize);
    content='<div class="focus-grid">'+pageRows.map(m=>card(m,true)).join('')+'</div>'}
  else{const groups=state.groups.map(g=>({source:g.source,matches:visibleRows.filter(m=>m.assigned_source_id===g.source.id)})).filter(g=>g.matches.length);
    const unassigned=visibleRows.filter(m=>!m.assigned_source_id);
    const units=[...unassigned.map(m=>({single:m})),...groups];
    total=Math.max(1,Math.ceil(units.length/view.pageSize));view.page=Math.max(0,Math.min(view.page,total-1));
    const slice=units.slice(view.page*view.pageSize,(view.page+1)*view.pageSize);
    pageRows=slice.flatMap(x=>x.single?[x.single]:x.matches);
    const loose=slice.filter(x=>x.single);
    if(loose.length)content+='<h3 class="unassigned-head">无归属 / 待处理</h3><div class="unassigned-grid">'+loose.map(x=>card(x.single)).join('')+'</div>';
    const grid=slice.filter(x=>!x.single);
    if(grid.length)content+=`<div class="matrix"><table><thead><tr><th>主图基准</th>${cols.map(f=>`<th>${esc(f.label)} <span class="muted">${f.count}</span></th>`).join('')}</tr></thead><tbody>${grid.map(g=>`<tr data-src-row="${g.source.id}" class="${activeSource===g.source.id?'hl':''}"><td class="src-cell" data-drop-source="${g.source.id}">${thumb(g.source.id,`data-zoom="${g.source.id}"`)}<div class="group-name">${esc(g.source.local_id+' '+g.source.name)}</div></td>${cols.map(f=>{const m=g.matches.find(m=>m.folder_id===f.id);return '<td data-cell-folder="'+f.id+'">'+(m?card(m):`<div class="empty-cell" data-drop-source="${g.source.id}" data-drop-folder="${f.id}">无归属，可拖入图片</div>`)+'</td>'}).join('')}</tr>`).join('')}</tbody></table></div>`}
  area.innerHTML=content;const mx=area.querySelector('.matrix');if(mx){mx.scrollTop=view._top||prevTop;mx.scrollLeft=view._left||prevLeft}else area.scrollTop=view._areaTop||0;
  $('#emptyReview').hidden=!!visibleRows.length||view.screen!=='review';
  $('#pageInfo').textContent=`${view.page+1} / ${total} 页`;
  $('#prevPage').disabled=view.page===0;$('#nextPage').disabled=view.page>=total-1;
  $('#reviewCount').textContent=`筛选 ${visibleRows.length} / 共 ${state.matches.length}`;
  renderFilters();
  $('#selectScope').textContent=`选择 ${{filtered:visibleRows.length,page:pageRows.length,folder:view.folder?state.matches.filter(m=>m.folder_id===view.folder).length:0,all:state.matches.length}[view.scope]} 张`;
  updateBulk();save()}
function renderFilters(){$('#statusFilters').innerHTML=statusOrder.map(st=>{const n=state.matches.filter(m=>(!view.folder||m.folder_id===view.folder)&&(!st||m.status===st)).length;return `<button data-status="${st}" class="${view.status===st?'active':''}">${st?esc(labels[st]):'全部'} ${n}</button>`}).join('');
  $('#folderFilter').innerHTML='<option value="">全部目标目录</option>'+folders().map(f=>`<option value="${f.id}" ${view.folder===f.id?'selected':''}>${esc(f.label)}</option>`).join('')}

/* ---------- selection ---------- */
function setActive(id){active=id==null?null:Number(id);$$('.tcard').forEach(c=>c.classList.toggle('current',Number(c.dataset.target)===active));syncPanels();if(view.rightOpen&&active)renderDetail();save()}
function clickSelect(id,{ctrl=false,shift=false}={}){id=Number(id);
  if(shift&&anchor!=null&&visibleRows.some(m=>m.target_id===anchor)){const a=visibleRows.findIndex(m=>m.target_id===anchor),b=visibleRows.findIndex(m=>m.target_id===id);
    if(b>=0)for(const m of visibleRows.slice(Math.min(a,b),Math.max(a,b)+1))selected.add(m.target_id)}
  else if(ctrl){selected.has(id)?selected.delete(id):selected.add(id)}
  else{if(!(selected.size===1&&selected.has(id))){selected=new Set([id])}}
  anchor=id;setActive(id);refreshCardSelection();updateBulk();save()}
function refreshCardSelection(){$$('.tcard').forEach(c=>{const id=Number(c.dataset.target),on=selected.has(id);c.classList.toggle('selected',on);const cb=c.querySelector('.sel');if(cb)cb.checked=on})}
function navigate(delta){const rows=visibleRows;if(!rows.length)return;let i=rows.findIndex(m=>m.target_id===active);i=Math.max(0,Math.min(rows.length-1,i+delta));const id=rows[i].target_id;setActive(id);
  let node=$(`.tcard[data-target="${id}"]`);
  if(!node){if(view.layout==='folder')view.page=Math.floor(i/view.pageSize);
    else{const units=[...visibleRows.filter(m=>!m.assigned_source_id).map(m=>'t'+m.target_id),...state.groups.filter(g=>visibleRows.some(m=>m.assigned_source_id===g.source.id)).map(g=>'s'+g.source.id)];const key=rows[i].assigned_source_id?'s'+rows[i].assigned_source_id:'t'+id;view.page=Math.max(0,Math.floor(units.indexOf(key)/view.pageSize))}
    renderReview();node=$(`.tcard[data-target="${id}"]`)}
  node?.scrollIntoView({block:'nearest',inline:'nearest'})}

/* ---------- bulk bar ---------- */
function updateBulk(){if(!state)return;const rows=selectedRows(),allowed=rows.filter(m=>m.can_bulk_confirm&&!m.locked);
  const counts={};for(const m of rows)counts[folderName(m.folder_id)]=(counts[folderName(m.folder_id)]||0)+1;
  const hiddenN=rows.filter(m=>!visibleRows.some(v=>v.target_id===m.target_id)).length;
  const bar=$('#bulkBar');bar.hidden=!rows.length||view.screen!=='review';
  $('#selectionSummary').textContent=`已选择 ${rows.length} 张图片 · ${Object.keys(counts).length} 个目录`;
  $('#selectionDetails').textContent=`可确认 ${allowed.length} · 其余 ${rows.length-allowed.length} 需单独处理${hiddenN?` · 筛选外 ${hiddenN} 张`:''}${rows.length?' · '+Object.entries(counts).map(([k,v])=>k+' '+v).join(' / '):''}`;
  $('#confirmSelected').textContent=`确认匹配 ${allowed.length} 张`;
  $('#confirmSelected').disabled=busy||!allowed.length;
  for(const id of ['markReview','markUnmatched','batchAssign','unconfirm','exportSelected']) $('#'+id).disabled=busy||!rows.length;
  $('#undoReview').disabled=busy||!state.undo;$('#previewNaming').disabled=busy||!state.matches.length}

/* ---------- review actions ---------- */
async function reviewAction(act,ids=[...selected],mappings,allowLocked=false){if(!ids.length)throw Error('请先选择图片');
  const result=await api(`/batches/${batch}/reviews`,{action:act,target_ids:ids,mappings,allow_locked:allowLocked});
  plan=null;await refresh();
  toast(`已处理 ${result.applied} 张${result.skipped.length?`，跳过 ${result.skipped.length} 张（${[...new Set(result.skipped.map(x=>x.reason))].join('；')}）`:''}`,{action:result.command_id?{label:'撤销',fn:()=>action(undo)}:null});
  return result}
async function undo(){if(!state?.undo)throw Error('没有可撤销的审核操作');
  await api(`/batches/${batch}/undo`,{command_id:state.undo.id});editing=null;mappingDraft=null;plan=null;await refresh();toast('已恢复上次操作前的归属、确认与锁定状态。')}

/* ---------- right detail panel ---------- */
function syncPanels(){$('#sideLeft').classList.toggle('collapsed',!view.leftOpen);$('#toggleLeft').classList.toggle('collapsed',!view.leftOpen);$('#toggleLeft').textContent=view.leftOpen?'‹':'›';
  const showRight=view.rightOpen&&!!active&&view.screen==='review';
  $('#sideRight').hidden=!showRight;$('#toggleRight').hidden=view.screen==='import';$('#toggleRight').classList.toggle('collapsed',!showRight);$('#toggleRight').textContent=showRight?'›':'‹'}
let cmp={s:1,x:0,y:0};
function renderDetail(){const m=match(active);if(!m)return;$('#detailTitle').textContent=`${folderName(m.folder_id)} · ${m.target.local_id}`;
  const idx=visibleRows.findIndex(x=>x.target_id===m.target_id);
  const cmpInner=cmpMode==='pair'
    ?`<div class="cmp" style="display:flex"><div class="cmp-img" style="position:relative;flex:1">${m.source?`<img src="/api/images/${m.source.id}?full=true" alt="主图原图">`:'<div class="empty-cell">暂无主图</div>'}</div><div class="cmp-img" style="position:relative;flex:1"><img src="/api/images/${m.target_id}?full=true" alt="目标原图"></div></div>`
    :`<div class="cmp"><div class="cmp-img" id="cmpZoom"><img src="/api/images/${cmpMode==='target'?m.target_id:m.source?.id}?full=true" alt="原图"></div></div>`;
  $('#detailBody').innerHTML=`
  <div class="d-nav"><button data-dstep="-1">‹ 上一张 K</button><button data-dstep="1">下一张 J ›</button><span class="muted">${idx>=0?idx+1+' / '+visibleRows.length:''}</span></div>
  <div class="cmp-tools">
    <div class="cmp-toggle"><button data-cmp="pair" class="${cmpMode==='pair'?'active':''}">对比</button><button data-cmp="target" class="${cmpMode==='target'?'active':''}">目标</button><button data-cmp="source" class="${cmpMode==='source'?'active':''}" ${m.source?'':'disabled'}>主图</button></div>
    <button data-cmpzoom="in">＋</button><button data-cmpzoom="out">−</button><button data-cmpzoom="fit">适应</button>
    <button id="openLightbox">灯箱 V</button>
  </div>
  ${cmpInner}
  <div class="d-meta">
    <div>${badge(m.status)} ${m.locked?'<span title="人工锁定">🔒 人工锁定</span>':''} ${m.can_bulk_confirm?'<span class="muted">可直接确认</span>':''}</div>
    <div><span class="k">目标：</span>${esc(m.target.name)}</div>
    <div><span class="k">主图：</span>${m.source?esc(m.source.name):'未分配'}</div>
    <div><span class="k">新名：</span><b>${esc(m.new_name||'—')}</b></div>
  </div>
  ${m.suggestion.evidence?.length?`<div class="evidence">${esc(m.suggestion.evidence.join('；'))}</div>`:''}
  ${m.suggestion.conflicts?.length?`<div class="evidence conflict">冲突：${esc(m.suggestion.conflicts.join('；'))}</div>`:''}
  ${(m.suggestion.review_reasons||m.notes||[]).length?`<div class="evidence">${esc((m.suggestion.review_reasons||m.notes||[]).join('；'))}</div>`:''}
  <div class="detail-sec">其他视觉候选</div>
  <div class="cand-strip">${m.candidates.map(c=>`<div class="cand ${pendingAssign===c.source_id?'picked':''}" data-cand="${c.source_id}" title="点击设为待保存归属">${thumb(c.source_id)}<div class="c-name">${esc(image(c.source_id)?.name||'')}</div>${badge(c.decision)}</div>`).join('')||'<span class="muted">模型未提供候选</span>'}</div>
  <div class="detail-sec">纠正归属</div>
  <div class="src-picker"><input id="srcPickSearch" placeholder="输入主图文件名搜索"><div class="src-pick-list">${sources().map(s=>`<div class="src-pick ${pendingAssign===s.id?'hl':''} ${m.assigned_source_id===s.id?'hl':''}" data-pick="${s.id}">${thumb(s.id)}<span>${esc(s.local_id+' '+s.name)}</span></div>`).join('')}</div></div>
  <div class="d-actions">
    <button id="saveAssign" class="primary" data-mutation ${pendingAssign?'':'disabled'}>保存归属并锁定</button>
    ${m.assigned_source_id?`<button id="confirmSingle" data-mutation>确认此归属</button>`:''}
    <button id="singleReview" data-mutation>待复核</button>
    <button id="singleUnmatched" data-mutation>未匹配</button>
    <button id="singleUnlock" data-mutation>${m.locked?'解锁':'恢复模型建议'}</button>
  </div>
  <details style="margin-top:10px"><summary class="muted">高级：原图复核与 PSD</summary>
    <p class="muted">选择 1～2 个候选后批准 1 次原图模型复核（单独计预算）。</p>
    <button id="requestOriginal" data-mutation>批准 1 次原图 / Top2 复核</button>
    <button id="psdLink">关联主图 PSD</button>
    ${m.source?.psd_path?`<a target="_blank" href="/api/images/${m.source.id}?psd=true">只读预览 PSD</a>`:''}
    ${m.review?`<details><summary>原图复核记录</summary><pre style="font-size:11px;overflow:auto">${esc(JSON.stringify(m.review.result,null,2))}</pre></details>`:''}
  </details>`;
  cmp={s:1,x:0,y:0};applyCmp()}
function applyCmp(){const img=$('#cmpZoom img');if(img)img.style.transform=`translate(${cmp.x}px,${cmp.y}px) scale(${cmp.s})`}
function pickSource(sid){pendingAssign=sid;const m=match(active);
  if(m&&cmpMode!=='target'){cmpMode='source'}
  renderDetail();
  if(sid&&m){const img=$('#cmpZoom img');if(img)img.src=`/api/images/${sid}?full=true`}
  const b=$('#saveAssign');if(b)b.disabled=!sid}

/* ---------- assign modal (batch correction) ---------- */
function openAssign(ids=[...selected],restore=false){if(!ids.length){toast('请先选择需要纠错的图片。',{error:true});return}
  editing=restore&&editing?editing:(mappingDraft&&JSON.stringify(mappingDraft.ids)===JSON.stringify(ids)?mappingDraft:{ids,mapping:Object.fromEntries(ids.map(id=>[id,match(id).assigned_source_id||''])),allowLocked:false});
  $('#assignTitle').textContent=`批量改归属 · ${ids.length} 张`;
  $('#assignBody').innerHTML=`
  <p class="muted">逐行核对后一次提交。同一目录内一对一；未选中的人工锁定结果不会被抢占；支持两行互换。</p>
  <div class="row" style="margin:0 0 8px">
    <input id="assignSearch" type="search" placeholder="搜索主图，过滤所有行的选项" style="flex:1">
    <select id="assignAllSource"><option value="">统一选择主图…</option>${sourceOptions(null)}</select>
    <button id="applyCommonSource">填入所有行</button>
    <label class="check"><input type="checkbox" id="allowLocked" ${editing.allowLocked?'checked':''}>也修改所选锁定项</label>
  </div>
  ${ids.map(id=>{const m=match(id);return `<div class="mapping-row">${thumb(id)}<div class="filename">${esc(folderName(m.folder_id))}<br>${esc(m.target.name)} ${m.locked?'🔒':''}</div><div class="mapping-preview">${editing.mapping[id]?thumb(Number(editing.mapping[id])):''}</div><select class="mapping-source" data-id="${id}"><option value="">选择主图…</option>${sourceOptions(Number(editing.mapping[id]))}</select></div>`}).join('')}
  <div class="row"><button id="commitMappings" class="primary" data-mutation>一次保存 ${ids.length} 条归属</button><span class="muted">草稿自动保留，刷新后可继续</span></div>`;
  $('#assignModal').hidden=false;save()}
function sourceOptions(sel){return sources().map(s=>`<option value="${s.id}" ${s.id===sel?'selected':''}>${esc(s.local_id+' '+s.name)}</option>`).join('')}
function closeAssign(){if(editing)mappingDraft=editing;editing=null;$('#assignModal').hidden=true;save()}

/* ---------- naming view ---------- */
async function openNaming(){if(!state?.matches.length)throw Error('请先导入目标图片');showScreen('naming');await generateNaming()}
async function generateNaming(){view.namingScope=$('#namingScope').value;view.namingMode=$('#namingMode').value;
  let ids=view.namingScope==='all'?undefined:view.namingScope==='filtered'?filtered().map(m=>m.target_id):[...selected];
  if(ids&&!ids.length)throw Error('命名范围为空，不会自动扩大为全部');
  $('#namingPreview').innerHTML='<p class="muted">正在校验命名范围、源图与文件冲突…</p>';
  plan=await api(`/batches/${batch}/naming`,{mode:view.namingMode,target_ids:ids});renderNaming();save()}
function renderNaming(){if(!plan)return;const rows=plan.result?.rows||plan.payload.rows;
  const ready=rows.filter(r=>r.status==='READY').length,done=rows.filter(r=>['COPIED_VERIFIED','SKIPPED_VALID','RENAMED_VERIFIED','NOOP'].includes(r.status)).length,blocked=rows.filter(r=>r.status==='BLOCKED').length;
  $('#namingSummary').innerHTML=`<span class="sum-chip"><b>${rows.length}</b>张 / ${new Set(rows.map(r=>r.folder_id)).size} 目录</span><span class="sum-chip"><b>${ready}</b>可执行</span><span class="sum-chip"><b>${done}</b>已完成</span>${blocked?`<span class="sum-chip bad"><b>${blocked}</b>冲突被阻止</span>`:''}<span class="sum-chip muted">${esc(plan.payload.output_root)}</span>`;
  const stMap={READY:'可执行',BLOCKED:'已阻止',COPIED_VERIFIED:'复制已校验',SKIPPED_VALID:'此前输出已校验',RENAMED_VERIFIED:'原地改名已校验',NOOP:'无需改名'};
  $('#namingPreview').innerHTML=`${blocked?'<p class="scope-note">被阻止的项目不会命名：存在冲突、未确认或文件变化。请返回审核处理后再更新预览。</p>':''}<div class="naming-table"><table><thead><tr><th>目录</th><th>原文件名</th><th>对应主图</th><th>新文件名</th><th>状态与原因</th></tr></thead><tbody>${rows.map(r=>`<tr class="${esc(r.status)}"><td>${esc(r.folder_id)}</td><td>${esc(r.original_name)}</td><td>${esc(r.source_name||'')}</td><td><b>${esc(r.new_name||'')}</b></td><td>${esc(stMap[r.status]||r.status)} ${esc(r.reason||'')}</td></tr>`).join('')}</tbody></table></div>`;
  const inPlace=plan.mode==='RENAME_IN_PLACE';
  $('#namingActions').innerHTML=`${inPlace?'<label class="check"><input type="checkbox" id="approveInPlace">明确允许按预览原地修改文件名（高风险）</label>':''}
  <button id="executeNaming" class="primary" data-mutation ${!ready&&!done?'disabled':''}>${plan.status==='EXECUTED'?'校验已有命名输出':plan.status==='EXECUTED_WITH_ERRORS'?'仅重试失败的复制项':'确认并'+(inPlace?'原地命名 ':'复制命名 ')+ready+' 张'}</button>
  ${inPlace&&plan.status==='EXECUTED'?'<button id="rollbackNaming">按清单回退原地改名</button>':''}
  <span class="muted">计划 ${esc(plan.status)} · 绑定审核版本 ${plan.revision}</span>`}

/* ---------- import view ---------- */
let targetRows=[];
function renderTargetRows(){$('#targetList').innerHTML=targetRows.map((r,i)=>`<div class="target-row"><input data-tlabel="${i}" placeholder="显示名（可选）" value="${esc(r.label||'')}"><input data-tpath="${i}" placeholder="完整路径，如 G:\\素材\\白底模特" value="${esc(r.path||'')}" spellcheck="false"><button class="del" data-tdel="${i}" title="删除该目录">✕</button></div>`).join('');
  $('#importCount').textContent=targetRows.filter(r=>r.path.trim()).length?`已登记 ${targetRows.filter(r=>r.path.trim()).length} 个目标目录`:'';saveImportDraft()}
function saveImportDraft(){const data={source:$('#sourcePath').value,sourceFolder:$('#sourceFolder').value,rows:targetRows};localStorage.setItem('vm3.importDraft',JSON.stringify(data))}
async function doImport(){const targets=targetRows.filter(r=>r.path.trim()).map(r=>({path:r.path.trim(),label:r.label.trim()||undefined}));
  if(!$('#sourcePath').value.trim())throw Error('请填写主图目录');if(!targets.length)throw Error('请至少添加一个目标目录');
  const result=await api(`/batches/${batch}/import`,{source:$('#sourcePath').value.trim(),targets});
  localStorage.setItem('vm3.lastImport',localStorage.getItem('vm3.importDraft'));
  await refresh();showScreen('review');view.runOpen=true;showScreen('review');
  $('#scanResults').innerHTML='';
  toast(`已扫描并导入 ${folders().length} 个目标目录，共 ${state.matches.length} 张。现在可一键开始匹配。${result.warnings?.length?' 注意：'+result.warnings.map(w=>w.folder+' '+w.message).join('；'):''}`)}
function renderScanResults(){const el=$('#scanResults');if(!state.folders.length){el.innerHTML='';return}
  el.innerHTML='<div class="import-card"><h1>已导入目录</h1>'+state.folders.map(f=>`<div class="scan-row">${dot(f.state)}<b>${esc(f.label)}</b><span>${f.count} 张</span><span class="muted">${esc(f.path)}</span></div>`).join('')+'<div class="row"><button id="goStartMatch" class="primary">开始自动匹配</button></div></div>'}
async function uploadFiles(files){if(!files.length)throw Error('请选择或拖入文件夹 / ZIP');$('#uploadStatus').textContent=`正在上传 ${files.length} 个文件…`;
  const data=new FormData();data.set('source_folder',$('#sourceFolder').value);
  for(const item of files)data.append('files',item.file||item,item.path||item.webkitRelativePath||item.name);
  const r=await fetch(`/api/batches/${batch}/upload`,{method:'POST',body:data});if(!r.ok)throw Error((await r.json()).detail);
  await refresh();showScreen('review');view.runOpen=true;showScreen('review');$('#uploadStatus').textContent='上传与扫描完成';toast('全部文件夹已上传并扫描，可统一启动匹配。')}
async function entries(entry,prefix=''){if(entry.isFile)return new Promise((resolve,reject)=>entry.file(f=>resolve([{file:f,path:prefix+f.name}]),reject));
  const reader=entry.createReader();let children=[];while(true){const chunk=await new Promise((r,j)=>reader.readEntries(r,j));if(!chunk.length)break;children.push(...chunk)}
  return (await Promise.all(children.map(e=>entries(e,prefix+entry.name+'/')))).flat()}

/* ---------- hover peek ---------- */
let peekTimer=null;
function showPeek(imgEl,id,name){clearTimeout(peekTimer);peekTimer=setTimeout(()=>{const peek=$('#hoverPeek');peek.querySelector('img').src=`/api/images/${id}`;peek.querySelector('.peek-name').textContent=name;
  const r=imgEl.getBoundingClientRect();peek.hidden=false;const w=350,h=350;let x=r.right+10,y=r.top-40;
  if(x+w>innerWidth)x=r.left-w-10;if(x<0)x=Math.min(r.right+10,innerWidth-w-8);y=Math.max(50,Math.min(y,innerHeight-h-10));
  peek.style.left=x+'px';peek.style.top=y+'px'},240)}
function hidePeek(){clearTimeout(peekTimer);$('#hoverPeek').hidden=true}

/* ---------- lightbox ---------- */
let lb={s:1,x:0,y:0};
function openLightbox(id){const m=match(id)||null;const img=image(id);if(!img)return;if(m)setActive(m.target_id);
  $('#lightbox').hidden=false;lb={s:1,x:0,y:0};loadLb(id)}
function loadLb(id){const img=image(id);const m=match(id);$('#lbImg').src=`/api/images/${id}?full=true`;
  $('#lbName').textContent=(m?folderName(m.folder_id)+' · ':'')+img.name+(m?' · '+(labels[m.status]||m.status):'');
  lb={s:1,x:0,y:0};applyLb()}
function applyLb(){const el=$('#lbImg'),st=$('#lbStage').getBoundingClientRect();
  if(lb.s==='fit'){const nw=el.naturalWidth||st.width,nh=el.naturalHeight||st.height;const s=Math.min(st.width/nw,st.height/nh);el.style.transform=`translate(${(st.width-nw*s)/2}px,${(st.height-nh*s)/2}px) scale(${s})`}
  else el.style.transform=`translate(${lb.x}px,${lb.y}px) scale(${lb.s})`}
function lbZoom(factor,cx,cy){if(lb.s==='fit')lb.s=1;const ns=Math.min(12,Math.max(.05,lb.s*factor));lb.x=cx-(cx-lb.x)*(ns/lb.s);lb.y=cy-(cy-lb.y)*(ns/lb.s);lb.s=ns;applyLb()}

/* ---------- drag & drop reassign ---------- */
function bindDragDrop(){
  document.addEventListener('dragstart',e=>{const c=e.target.closest('.tcard');if(c){e.dataTransfer.setData('text/plain',c.dataset.target);e.dataTransfer.effectAllowed='move'}});
  document.addEventListener('dragover',e=>{const d=e.target.closest('[data-drop-source]');if(d){e.preventDefault();d.classList.add('droptarget')}});
  document.addEventListener('dragleave',e=>{const d=e.target.closest('[data-drop-source]');if(d)d.classList.remove('droptarget')});
  document.addEventListener('drop',e=>{const d=e.target.closest('[data-drop-source]');if(!d)return;e.preventDefault();d.classList.remove('droptarget');
    const tid=Number(e.dataTransfer.getData('text/plain')),sid=Number(d.dataset.dropSource);if(!tid||!sid)return;
    action(async()=>{try{await reviewAction('assign',[tid],[{target_id:tid,source_id:sid}])}
      catch(err){toast(err.message,{error:true,action:{label:'包含锁定项强制执行',fn:()=>action(()=>reviewAction('assign',[tid],[{target_id:tid,source_id:sid}],true))}})}})})}

/* ---------- global events ---------- */
document.addEventListener('click',async e=>{
  const menuBtn=e.target.closest('#exportToggle');
  if(!e.target.closest('.menu-wrap'))$('#exportMenu').hidden=true;
  const b=e.target.closest('button,a.link,[data-open-batch]');
  const cardEl=e.target.closest('.tcard');
  if(cardEl&&!e.target.closest('button')&&e.target.type!=='checkbox'){clickSelect(cardEl.dataset.target,{ctrl:e.ctrlKey||e.metaKey,shift:e.shiftKey});return}
  if(!b)return;
  if(b.dataset.openBatch){action(()=>openBatch(b.dataset.openBatch));return}
  if(b.dataset.status!==undefined&&b.closest('#statusFilters')){view.status=b.dataset.status;view.page=0;renderReview();return}
  if(b.dataset.layout){view.layout=b.dataset.layout;view.page=0;$$('#layoutSeg button').forEach(x=>x.classList.toggle('active',x.dataset.layout===view.layout));
    if(view.layout==='folder'&&!view.folder)view.folder=folders()[0]?.id||'';renderReview();return}
  if(b.dataset.focusFolder){view.folder=b.dataset.focusFolder;view.layout='folder';view.page=0;showScreen('review');$$('#layoutSeg button').forEach(x=>x.classList.toggle('active',x.dataset.layout==='folder'));renderReview();return}
  if(b.dataset.dstep){navigate(Number(b.dataset.dstep));return}
  if(b.dataset.cmp){cmpMode=b.dataset.cmp;cmp={s:1,x:0,y:0};renderDetail();return}
  if(b.dataset.cmpzoom){const f=b.dataset.cmpzoom;const img=$('#cmpZoom img');if(f==='fit'){cmp={s:1,x:0,y:0};applyCmp()}else{cmp.s=Math.min(12,Math.max(.1,cmp.s*(f==='in'?1.3:1/1.3)));applyCmp()}return}
  if(b.dataset.run){action(async()=>{await api(`/runs/${b.dataset.run}/${b.dataset.control}`,{});await refresh();toast('任务领取状态已更新，已发送的请求会正常返回或超时。')});return}
  if(b.dataset.tdel!==undefined){targetRows.splice(Number(b.dataset.tdel),1);renderTargetRows();return}
  const id=b.id;
  if(id==='newBatchBtn'){showWelcome();return}
  if(id==='createFirst'){action(createBatch);return}
  if(id==='goImport'){showScreen('import');renderScanResults();return}
  if(id==='goNaming'||id==='previewNaming'){action(openNaming);return}
  if(id==='closeNaming'){showScreen('review');renderReview();return}
  if(id==='startMatchTop'||id==='goStartMatch'){view.runOpen=true;showScreen('review');renderProgress();$('#runPanel').scrollIntoView();$('#budget').focus();return}
  if(id==='progressChip'){view.runOpen=!view.runOpen;showScreen('review');renderProgress();return}
  if(id==='collapseRun'){view.runOpen=false;renderProgress();save();return}
  if(id==='startMatch'){action(()=>startMatch());return}
  if(id==='retryFailed'){action(()=>startMatch('failed'));return}
  if(id==='exportToggle'){$('#exportMenu').hidden=!$('#exportMenu').hidden;return}
  if(id==='addTarget'){targetRows.push({label:'',path:''});renderTargetRows();const inputs=$$('#targetList [data-tpath]');inputs[inputs.length-1]?.focus();return}
  if(id==='importLocal'){action(doImport);return}
  if(id==='upload'){action(()=>uploadFiles([...$('#uploadFolder').files,...$('#uploadZip').files]));return}
  if(id==='selectScope'){try{selected=new Set(scopeRows().map(m=>m.target_id));refreshCardSelection();updateBulk();save();toast(`已选择 ${selected.size} 张。`)}catch(err){toast(err.message,{error:true})}return}
  if(id==='invertSelection'){const cur=visibleRows.map(m=>m.target_id);for(const t of cur)selected.has(t)?selected.delete(t):selected.add(t);refreshCardSelection();updateBulk();save();return}
  if(id==='clearSelection'){selected.clear();refreshCardSelection();updateBulk();save();return}
  if(id==='prevPage'||id==='nextPage'){view.page+=id==='prevPage'?-1:1;renderReview();return}
  if(id==='confirmSelected'){action(()=>reviewAction('confirm'));return}
  if(id==='markReview'){action(()=>reviewAction('review'));return}
  if(id==='markUnmatched'){action(()=>reviewAction('unmatched'));return}
  if(id==='unconfirm'){action(()=>reviewAction('unconfirm',[...selected],null,true));return}
  if(id==='batchAssign'){openAssign();return}
  if(id==='undoReview'){action(undo);return}
  if(id==='exportSelected'){action(async()=>{const data=await api(`/batches/${batch}/selection-export`,{target_ids:[...selected]});const link=document.createElement('a');link.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));link.download='selected_matches.json';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000)});return}
  if(id==='toggleLeft'){view.leftOpen=!view.leftOpen;syncPanels();save();return}
  if(id==='toggleRight'){view.rightOpen=!view.rightOpen;syncPanels();if(view.rightOpen&&active)renderDetail();save();return}
  if(id==='closeDetail'){view.rightOpen=false;syncPanels();save();return}
  if(id==='keysHint'){$('#keysModal').hidden=false;return}
  if(id==='closeKeys'){$('#keysModal').hidden=true;return}
  if(id==='openLightbox'){openLightbox(active);return}
  if(id==='lbClose'){$('#lightbox').hidden=true;return}
  if(id==='lbPrev'||id==='lbNext'){navigate(id==='lbPrev'?-1:1);if(active)loadLb(active);return}
  if(id==='lbZoomIn'){lbZoom(1.3,innerWidth/2,innerHeight/2);return}
  if(id==='lbZoomOut'){lbZoom(1/1.3,innerWidth/2,innerHeight/2);return}
  if(id==='lbFit'){lb={s:'fit',x:0,y:0};applyLb();return}
  if(id==='lbOrig'){const el=$('#lbImg'),st=$('#lbStage').getBoundingClientRect();lb={s:1,x:(st.width-el.naturalWidth)/2,y:(st.height-el.naturalHeight)/2};applyLb();return}
  if(id==='saveAssign'){action(async()=>{if(!pendingAssign)throw Error('请先选择主图');await reviewAction('assign',[active],[{target_id:active,source_id:pendingAssign}],true);pendingAssign=null;renderDetail()});return}
  if(id==='confirmSingle'){action(async()=>{await api(`/batches/${batch}/review/${active}`,{action:'confirm'});plan=null;await refresh();renderDetail();toast('已人工确认并锁定此归属。',{action:{label:'撤销',fn:()=>action(undo)}})});return}
  if(id==='singleReview'){action(async()=>{await reviewAction('review',[active],null,true);renderDetail()});return}
  if(id==='singleUnmatched'){action(async()=>{await reviewAction('unmatched',[active],null,true);renderDetail()});return}
  if(id==='singleUnlock'){action(async()=>{await reviewAction(match(active)?.locked?'unlock':'restore',[active],null,true);renderDetail()});return}
  if(id==='requestOriginal'){action(async()=>{const m=match(active);const ids=pendingAssign?[pendingAssign]:m.candidates.slice(0,1).map(c=>c.source_id);
    if(!ids.length)throw Error('请先在候选或主图选择器中点选 1 个候选');
    await api(`/batches/${batch}/original-review`,{target_id:active,source_ids:ids,confirmed_budget:true});await refresh();toast('已提交 1 次原图复核，人工结果会保留。')});return}
  if(id==='psdLink'){const path=prompt('只读关联的 PSD 完整路径');if(path)action(async()=>{const m=match(active);if(!m?.source)throw Error('该图片尚未分配主图');await api(`/images/${m.source.id}/psd`,{path});await refresh();renderDetail()});return}
  if(id==='cancelMappings'){closeAssign();return}
  if(id==='applyCommonSource'){const sid=$('#assignAllSource').value;if(sid){for(const tid of editing.ids)editing.mapping[tid]=Number(sid);openAssign(editing.ids,true)}return}
  if(id==='commitMappings'){action(async()=>{await reviewAction('assign',editing.ids,editing.ids.map(t=>({target_id:t,source_id:Number(editing.mapping[t])})),editing.allowLocked);editing=null;mappingDraft=null;$('#assignModal').hidden=true;save()});return}
  if(id==='regenerateNaming'){action(generateNaming);return}
  if(id==='executeNaming'){action(async()=>{const inPlace=plan.mode==='RENAME_IN_PLACE';
    if(inPlace&&!$('#approveInPlace')?.checked)throw Error('原地重命名须先勾选明确授权');
    plan=await api(`/plans/${plan.id}/execute`,{confirmed_plan_id:plan.id,allow_in_place:inPlace});renderNaming();await refresh();
    toast(plan.status==='EXECUTED'?'命名完成，输出已通过 SHA256 校验，原图保留。':'命名部分失败，请查看原因后可仅重试失败项。',{error:plan.status!=='EXECUTED'})});return}
  if(id==='rollbackNaming'){action(async()=>{plan=await api(`/plans/${plan.id}/rollback`,{confirmed_plan_id:plan.id});renderNaming();await refresh();toast('已按清单回退原地改名。')});return}
});
document.addEventListener('click',e=>{const cb=e.target.closest('.sel');if(cb){e.stopPropagation();clickSelect(cb.dataset.sel,{ctrl:true,shift:e.shiftKey})}});
document.addEventListener('dblclick',e=>{const c=e.target.closest('.tcard');if(c){openLightbox(Number(c.dataset.target));return}
  const z=e.target.closest('[data-zoom]');if(z)openLightbox(Number(z.dataset.zoom));
  const cmp=e.target.closest('.cmp');if(cmp&&active)openLightbox(active)});
document.addEventListener('mouseover',e=>{const img=e.target.closest('.tcard img, .src-item img');if(!img)return;const c=img.closest('.tcard');
  if(c){const m=match(c.dataset.target);showPeek(img,c.dataset.target,m?m.target.name:'')}else{const s=img.closest('.src-item');if(s)showPeek(img,s.dataset.source,image(s.dataset.source)?.name||'')}});
document.addEventListener('mouseout',e=>{if(e.target.closest('.tcard img, .src-item img'))hidePeek()});
document.addEventListener('click',e=>{const c=e.target.closest('.cand');if(c){pickSource(Number(c.dataset.cand));return}
  const p=e.target.closest('.src-pick');if(p){pickSource(Number(p.dataset.pick));return}
  const si=e.target.closest('.src-item');if(si){const sid=Number(si.dataset.source);activeSource=activeSource===sid?null:sid;renderSources();
    const row=$(`tr[data-src-row="${sid}"]`);if(activeSource&&row){row.classList.add('hl');row.scrollIntoView({block:'nearest'})}else if(!activeSource)$$('.matrix tr.hl').forEach(r=>r.classList.remove('hl'));save()}});
document.addEventListener('change',e=>{const t=e.target;
  if(t.id==='batchPicker'){if(t.value)action(()=>openBatch(t.value));return}
  if(t.matches('.run-folder')){t.checked?runFolders.add(t.value):runFolders.delete(t.value);save();renderProgress();return}
  if(t.matches('.mapping-source')){editing.mapping[t.dataset.id]=Number(t.value)||'';const pv=t.closest('.mapping-row').querySelector('.mapping-preview');if(pv)pv.innerHTML=t.value?thumb(Number(t.value)):'';save();return}
  if(t.id==='allowLocked'){editing.allowLocked=t.checked;save();return}
  if(t.id==='folderFilter'){view.folder=t.value;view.page=0;renderReview();return}
  if(t.id==='selectionScope'){view.scope=t.value;renderReview();return}
  if(t.id==='concurrency'){view.concurrency=Number(t.value);save();return}
  if(t.id==='namingScope'||t.id==='namingMode')action(generateNaming)});
document.addEventListener('input',e=>{const t=e.target;
  if(t.id==='search'){view.search=t.value;view.page=0;renderReview();return}
  if(t.id==='sourceSearch'){view.sourceSearch=t.value;renderSources();return}
  if(t.id==='thumbSize'){view.thumb=Number(t.value);renderReview();return}
  if(t.id==='sourcePath'||t.id==='sourceFolder'){saveImportDraft();return}
  if(t.dataset.tlabel!==undefined){targetRows[Number(t.dataset.tlabel)].label=t.value;saveImportDraft();return}
  if(t.dataset.tpath!==undefined){targetRows[Number(t.dataset.tpath)].path=t.value;saveImportDraft();$('#importCount').textContent=`已登记 ${targetRows.filter(r=>r.path.trim()).length} 个目标目录`;return}
  if(t.id==='srcPickSearch'){const q=t.value.toLowerCase();$$('.src-pick').forEach(el=>{el.style.display=el.textContent.toLowerCase().includes(q)?'':'none'});return}
  if(t.id==='assignSearch'){const q=t.value.toLowerCase();$$('.mapping-source').forEach(sel=>{[...sel.options].forEach(o=>{o.hidden=!!q&&o.value&&!o.textContent.toLowerCase().includes(q)})});return}});
document.addEventListener('paste',e=>{const t=e.target;if(!t.matches||!t.matches('[data-tpath]'))return;const text=e.clipboardData?.getData('text')||'';
  if(!text.includes('\n'))return;e.preventDefault();
  const lines=text.split('\n').map(x=>x.trim()).filter(Boolean).map(x=>{const p=x.split('|').map(y=>y.trim());return p.length>1?{label:p[0],path:p[1]}:{label:'',path:x}});
  const i=Number(t.dataset.tpath);targetRows.splice(i,1,...lines);
  if(!targetRows.length)targetRows=[{label:'',path:''}];
  renderTargetRows();toast(`已按行拆分为 ${lines.length} 个目标目录。`)});
document.addEventListener('keydown',e=>{
  if(['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName)||e.target.isContentEditable)return;
  if(e.key==='Escape'){if(!$('#assignModal').hidden)closeAssign();$('#lightbox').hidden=true;$('#keysModal').hidden=true;hidePeek();return}
  if(e.key==='/'&&$('#lightbox').hidden){e.preventDefault();$('#search').focus();return}
  if(!state||view.screen!=='review')return;
  if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='z'){e.preventDefault();action(undo);return}
  if(e.key==='Enter'&&e.target.closest('button,a'))return;
  if(['j','ArrowDown','ArrowRight','k','ArrowUp','ArrowLeft'].includes(e.key)){e.preventDefault();navigate(['j','ArrowDown','ArrowRight'].includes(e.key)?1:-1);if(!$('#lightbox').hidden&&active)loadLb(active);return}
  const current=match(active)||visibleRows[0];if(!current)return;
  if(e.code==='Space'){e.preventDefault();clickSelect(current.target_id,{ctrl:true})}
  else if(e.key.toLowerCase()==='v'){e.preventDefault();openLightbox(current.target_id)}
  else if(e.key==='Enter'){e.preventDefault();if(current.can_bulk_confirm)action(()=>reviewAction('confirm',[current.target_id]));else toast('当前项不是可直接确认的建议，请在右侧详情中核对。',{error:true})}
  else if(e.key.toLowerCase()==='r')action(()=>reviewAction('review',[current.target_id]));
  else if(e.key.toLowerCase()==='u')action(()=>reviewAction('unmatched',[current.target_id]));
  else if(e.key.toLowerCase()==='a'&&(e.ctrlKey||e.metaKey)){e.preventDefault();selected=new Set(visibleRows.map(m=>m.target_id));refreshCardSelection();updateBulk();save()}});
/* zoom & pan interactions */
document.addEventListener('wheel',e=>{const z=e.target.closest('#cmpZoom');if(z){e.preventDefault();const r=z.getBoundingClientRect();const cx=e.clientX-r.left,cy=e.clientY-r.top;const f=e.deltaY<0?1.15:1/1.15;const ns=Math.min(12,Math.max(.1,cmp.s*f));cmp.x=cx-(cx-cmp.x)*(ns/cmp.s);cmp.y=cy-(cy-cmp.y)*(ns/cmp.s);cmp.s=ns;applyCmp()}
  if(e.target.closest('#lbStage')){e.preventDefault();lbZoom(e.deltaY<0?1.15:1/1.15,e.clientX,e.clientY-46)}},{passive:false});
document.addEventListener('mousedown',e=>{const z=e.target.closest('#cmpZoom,#lbStage');if(!z)return;e.preventDefault();
  const st=z.id==='lbStage'?lb:cmp;if(st.s==='fit')return;let lx=e.clientX,ly=e.clientY;
  const mv=ev=>{st.x+=ev.clientX-lx;st.y+=ev.clientY-ly;lx=ev.clientX;ly=ev.clientY;if(z.id==='lbStage')applyLb();else applyCmp()};
  const up=()=>{document.removeEventListener('mousemove',mv);document.removeEventListener('mouseup',up)};
  document.addEventListener('mousemove',mv);document.addEventListener('mouseup',up)});
window.addEventListener('popstate',()=>{const id=new URLSearchParams(location.search).get('batch');if(id&&id!==batch)action(()=>openBatch(id))});
$('#dropZone').addEventListener('dragover',e=>{e.preventDefault();e.currentTarget.classList.add('dragover')});
$('#dropZone').addEventListener('dragleave',e=>e.currentTarget.classList.remove('dragover'));
$('#dropZone').addEventListener('drop',e=>{e.preventDefault();e.currentTarget.classList.remove('dragover');const items=[...e.dataTransfer.items];
  action(async()=>{const all=(await Promise.all(items.map(i=>i.webkitGetAsEntry()).filter(Boolean).map(x=>entries(x)))).flat();await uploadFiles(all)})});
bindDragDrop();
/* ---------- boot ---------- */
action(async()=>{
  const health=await api('/health');replayMode=health.validation_replay===true;
  $('#health').textContent=replayMode?'离线回放验收 · 0 次模型调用':'本机 · GPT-6.1 Terra';
  $('#exportMenu').innerHTML=['zip','csv','json','html'].map(k=>`<a href="#" data-export="${k}">${{zip:'完整 ZIP',csv:'映射 CSV',json:'结果 JSON',html:'HTML 报告'}[k]}</a>`).join('');
  document.addEventListener('click',e=>{const a=e.target.closest('[data-export]');if(a){e.preventDefault();if(!batch)return;location.href=`/api/batches/${batch}/export/${a.dataset.export}`}});
  const d=load('vm3.importDraft',load('vm3.lastImport',{}));
  $('#sourcePath').value=d.source||'';$('#sourceFolder').value=d.sourceFolder||'主图';
  targetRows=Array.isArray(d.rows)&&d.rows.length?d.rows:[{label:'',path:''}];renderTargetRows();
  await listBatches();
  const id=new URLSearchParams(location.search).get('batch')||localStorage.vm3Batch;
  if(id){try{await openBatch(id)}catch(e){toast(e.message,{error:true});showWelcome()}}else showWelcome()});
setInterval(()=>{if(batch&&!busy&&state?.runs.some(r=>['RUNNING','BUDGET_PAUSED'].includes(r.state)))refresh(true).catch(e=>toast(e.message,{error:true}))},2000);
