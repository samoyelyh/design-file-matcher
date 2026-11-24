// Only reads reach the service. Matching starts are captured before dispatch.
const {chromium} = require('playwright');
const fs = require('fs');
const path = require('path');
(async () => {
  const browser = await chromium.launch({channel: 'msedge', headless: true});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1050}, serviceWorkers: 'block'});
    const errors = [];
    const dialogs = [], starts = [], blocked = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/*', route => {
      const request = route.request();
      if (['GET', 'HEAD'].includes(request.method())) return route.continue();
      const pathname = new URL(request.url()).pathname;
      if (request.method() === 'POST' && /^\/api\/batches\/\d+\/match$/.test(pathname)) {
        starts.push(request.postDataJSON());
        return route.fulfill({status: 409, json: {detail: '浏览器验证已拦截匹配请求'}});
      }
      blocked.push(pathname);
      return route.abort();
    });
    page.on('dialog', async dialog => {
      dialogs.push(dialog.message());
      await dialog.dismiss();
    });
    const origin = 'http://127.0.0.1:8088';
    const batches = await (await page.request.get(`${origin}/api/batches`)).json();
    const batch = batches.find(row => row.status === 'SCANNED' && row.main_count) ||
      batches.find(row => row.status === 'COMPLETED' && row.main_count);
    if (!batch) throw Error('No scanned batch is available for a read-only UI check');
    await page.goto(`${origin}/#batch=${batch.id}&view=process`, {waitUntil: 'domcontentloaded', timeout: 60000});
    await page.locator('#visionmatchPanel').waitFor({state: 'visible'});
    if (await page.locator('#matchingModel, [data-folder-engine], [data-view="ai-lab"]').count()) {
      throw Error('Only VisionMatch should be available');
    }
    const fillLabels = await page.evaluate(() => {
      const file = {match_method: 'VISIONMATCH', status: 'AMBIGUOUS', review_status: 'PENDING', match_details: {auto_filled: true}};
      return {card: categoryStatus({files: [file]}).label, detail: confidenceLabel(file)};
    });
    if (!fillLabels.card.includes('自动补位') || !fillLabels.detail.includes('复核')) throw Error('Auto-fill must be marked for manual review');
    if (await page.locator('#matchBtn').isDisabled()) throw Error('VisionMatch incorrectly requires manual masks');
    await page.click('#matchBtn');
    await page.waitForFunction(() => document.querySelector('#matchBtn').disabled === false);
    const output = path.resolve('data/visionmatch-deployment');
    fs.mkdirSync(output, {recursive: true});
    await page.screenshot({path: path.join(output, '8088-visionmatch.png'), fullPage: true});
    const result = {fillLabels, dialogs, interceptedStarts: starts.length, unexpectedWrites: blocked, errors};
    fs.writeFileSync(path.join(output, 'browser.json'), JSON.stringify(result, null, 2));
    if (errors.length || dialogs.length || blocked.length || starts.length !== 1 ||
        starts[0].match_engine !== 'VISIONMATCH' || 'visionmatch_confirmed_budget' in starts[0] ||
        'visionmatch_max_calls' in starts[0]) throw Error(JSON.stringify(result));
    console.log(JSON.stringify(result));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
