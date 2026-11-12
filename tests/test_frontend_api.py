import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_api_handles_response_streams_and_prevents_overlapping_review_writes():
    script = r'''
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('app/static/app.js', 'utf8');
const start = source.indexOf('async function api(');
const end = source.indexOf('\nfunction toast(', start);
const context = {state: {reviewSaving: false}, Response};
vm.createContext(context);
vm.runInContext(source.slice(start, end), context);
async function check(status, body, expected) {
  const response = new Response(body, {status});
  let calls = 0;
  context.fetch = async () => {calls++; return response;};
  if (expected instanceof RegExp) await assert.rejects(context.api('/test'), expected);
  else assert.equal(JSON.stringify(await context.api('/test')), JSON.stringify(expected));
  assert.equal(calls, 1);
  assert.equal(response.bodyUsed, body !== null);
}
(async () => {
  await check(200, '{"ok":true}', {ok: true});
  await check(204, null, null);
  await check(400, '{"detail":"不能跨批次修改归属"}', /不能跨批次/);
  await check(422, '{"detail":[{"msg":"必填参数缺失"}]}', /必填参数缺失/);
  await check(503, '服务暂不可用', /服务暂不可用/);
  await check(500, 'Internal Server Error', /HTTP 500/);
  await check(502, '<html>Bad Gateway</html>', /HTTP 502/);
  await check(503, '', /HTTP 503/);
  await check(400, '{invalid JSON', /invalid JSON/);
  await check(200, 'not JSON', /数据格式异常/);

  let finish, calls = 0;
  context.fetch = async () => {
    calls++;
    return new Promise(resolve => {finish = () => resolve(new Response('{"confirmed":1}'));});
  };
  const first = context.api('/api/files/1/confirm', {method: 'POST'});
  await assert.rejects(context.api('/api/batches/1/confirm-groups', {method: 'POST'}), /正在保存审核/);
  assert.equal(calls, 1);
  finish();
  await first;
  assert.equal(context.state.reviewSaving, false);

  context.fetch = async () => {throw Error('network offline');};
  await assert.rejects(context.api('/api/files/1/confirm', {method: 'POST'}), /network offline/);
  assert.equal(context.state.reviewSaving, false);
  console.log('response and review submission regression checks passed');
})().catch(error => {console.error(error); process.exitCode = 1;});
'''
    subprocess.run(["node", "-e", script], cwd=ROOT, check=True, capture_output=True, text=True)


def test_already_named_files_can_finish_but_conflicts_and_completed_plans_cannot_execute():
    script = r'''
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('app/static/app.js', 'utf8');
const start = source.indexOf('const renameStatusNames =');
const end = source.indexOf('\n$("#generateRenamePlan").onclick', start);
const elements = new Map();
const context = {state: {renameSaving: false, conflictMarkSaving: false, reviewSaving: false}, esc: String, $$: () => [],
 $: selector => {
  if (!elements.has(selector)) elements.set(selector, {classList: {toggle() {}}, disabled: false});
  return elements.get(selector);
 }};
vm.createContext(context);
vm.runInContext(source.slice(start, end), context);
function check(statuses, disabled, label) {
 context.renderRenamePlan({plan_group: 'fixture', plans: statuses.map(rename_status => ({rename_status}))});
 assert.equal(elements.get('#executeRenamePlan').disabled, disabled);
 assert.equal(elements.get('#executeRenamePlan').textContent, label);
}
check(['NOOP', 'NOOP'], false, '确认完成');
assert.match(elements.get('#renameBusinessStatus').textContent, /2 个已确认文件名称已正确/);
check(['READY', 'NOOP'], false, '仅改名已确认项目');
check(['NOOP', 'RENAME_CONFLICT'], true, '请先处理异常图片');
check(['BLOCKED'], true, '请先处理异常图片');
check(['RENAMED', 'IGNORED'], true, '已完成');
check([], true, '请先预览命名结果');
context.state.renameSaving = true;
check(['NOOP'], true, '确认完成');
'''
    subprocess.run(["node", "-e", script], cwd=ROOT, check=True, capture_output=True, text=True)
