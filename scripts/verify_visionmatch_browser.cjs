// Read-only browser verification. Budget plan is mocked, model-start is blocked.
const {chromium} = require('playwright');
const fs = require('fs');
const path = require('path');
(async () => {
  const browser = await chromium.launch({channel: 'msedge', headless: true});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1050}});
    const errors = [];
    let starts = 0, cancelledBudget = false;
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/api/batches/*/visionmatch-plan', route => route.fulfill({
      contentType: 'application/json', body: JSON.stringify({folders: [{category: 'UI test only'}], calls_upper_bound: 1}),
    }));
    await page.route(/\/api\/batches\/\d+\/match$/, route => { starts++; return route.abort(); });
    page.on('dialog', async dialog => {
      if (dialog.message().includes('VisionMatch AI')) { cancelledBudget = true; await dialog.dismiss(); }
      else await dialog.accept();
    });
    await page.goto('http://127.0.0.1:8088/#batch=125&view=process', {waitUntil: 'domcontentloaded', timeout: 60000});
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
    await page.waitForTimeout(300);
    const output = path.resolve('data/visionmatch-deployment');
    fs.mkdirSync(output, {recursive: true});
    await page.screenshot({path: path.join(output, '8088-visionmatch.png'), fullPage: true});
    const result = {fillLabels, cancelledBudget, modelStartRequests: starts, errors};
    fs.writeFileSync(path.join(output, 'browser.json'), JSON.stringify(result, null, 2));
    if (errors.length || starts !== 0 || !cancelledBudget) throw Error(JSON.stringify(result));
    console.log(JSON.stringify(result));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
