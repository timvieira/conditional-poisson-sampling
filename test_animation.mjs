/**
 * Smoke tests for the sampling animation widget.
 * Tests: page loads, no JS errors, tree renders, animation works,
 * math is correct, style matches blog.
 */
import puppeteer from 'puppeteer-core';
import { readFileSync } from 'fs';
import { resolve } from 'path';
import http from 'http';

const HTML_PATH = resolve('output/sampling-animation/index.html');
let server, browser, page;
let errors = [];
let testsPassed = 0;
let testsFailed = 0;

function assert(cond, msg) {
  if (cond) {
    testsPassed++;
    console.log(`  ✓ ${msg}`);
  } else {
    testsFailed++;
    console.log(`  ✗ FAIL: ${msg}`);
    errors.push(msg);
  }
}

async function setup() {
  // Serve the output directory with local D3/MathJax fallback
  server = http.createServer((req, res) => {
    let urlPath = req.url.split('?')[0];
    if (urlPath.endsWith('/')) urlPath += 'index.html';
    let filePath = resolve('output', urlPath.replace(/^\//, ''));
    try {
      let content = readFileSync(filePath, 'utf8');
      const ext = filePath.split('.').pop();
      const types = { html: 'text/html', js: 'application/javascript', css: 'text/css' };
      // Replace CDN URLs with local paths for offline testing
      if (ext === 'html') {
        content = content.replace(
          'https://d3js.org/d3.v7.min.js',
          '/d3.min.js'
        );
        // Remove MathJax (not needed for functional tests)
        content = content.replace(
          /<script>\s*MathJax[\s\S]*?<\/script>\s*<script src="https:\/\/cdn\.jsdelivr[^"]*"><\/script>/,
          ''
        );
      }
      res.writeHead(200, { 'Content-Type': types[ext] || 'text/plain' });
      res.end(content);
    } catch {
      // Try serving from node_modules
      if (urlPath === '/d3.min.js') {
        try {
          const d3Content = readFileSync(resolve('node_modules/d3/dist/d3.min.js'));
          res.writeHead(200, { 'Content-Type': 'application/javascript' });
          res.end(d3Content);
          return;
        } catch {}
      }
      res.writeHead(404);
      res.end('Not found');
    }
  });
  await new Promise(r => server.listen(0, r));
  const port = server.address().port;

  browser = await puppeteer.launch({
    executablePath: '/root/.cache/ms-playwright/chromium-1194/chrome-linux/chrome',
    headless: true,
    args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage']
  });
  page = await browser.newPage();

  // Collect console errors
  const jsErrors = [];
  page.on('console', msg => {
    if (msg.type() === 'error') {
      const text = msg.text();
      // Ignore 404s for external resources in offline test environment
      if (text.includes('Failed to load resource')) return;
      if (text.includes('net::ERR_')) return;
      jsErrors.push(text);
    }
  });
  page.on('pageerror', err => {
    if (err.message.includes('MathJax')) return;
    jsErrors.push(err.message);
  });
  page.on('requestfailed', req => {
    const url = req.url();
    // Ignore CDN failures in test env
    if (url.includes('cdn.') || url.includes('d3js.org') || url.includes('favicon')) return;
    jsErrors.push('Failed to load resource: ' + url);
  });

  await page.goto(`http://localhost:${port}/sampling-animation/`, { waitUntil: 'domcontentloaded', timeout: 30000 });

  // Debug: check what loaded
  const d3loaded = await page.evaluate(() => typeof d3 !== 'undefined');
  if (!d3loaded) {
    console.log('WARNING: D3 not loaded from CDN. Page HTML snippet:');
    const html = await page.content();
    console.log(html.substring(0, 500));
  }

  // Wait for D3 and MathJax to load
  await page.waitForSelector('svg', { timeout: 15000 });
  await new Promise(r => setTimeout(r, 2000));

  return jsErrors;
}

async function teardown() {
  if (browser) await browser.close();
  if (server) server.close();
}

async function testPageLoads(jsErrors) {
  console.log('\n--- Page Load Tests ---');
  assert(jsErrors.length === 0, `No JS errors on load (got ${jsErrors.length}: ${jsErrors.join('; ')})`);

  const title = await page.title();
  assert(title.includes('Sampling'), `Page title contains "Sampling": "${title}"`);

  const svgCount = await page.$$eval('svg', els => els.length);
  assert(svgCount >= 1, `At least one SVG element (got ${svgCount})`);
}

async function testTreeStructure() {
  console.log('\n--- Tree Structure Tests ---');

  const nodeCount = await page.$$eval('.tree-node', els => els.length);
  // N=8: 8 leaves + 4 + 2 + 1 = 15 nodes (minus pads)
  assert(nodeCount >= 8, `At least 8 tree nodes rendered (got ${nodeCount})`);
  assert(nodeCount <= 15, `At most 15 tree nodes for N=8 (got ${nodeCount})`);

  const edgeCount = await page.$$eval('.tree-edge', els => els.length);
  assert(edgeCount >= 7, `At least 7 tree edges rendered (got ${edgeCount})`);

  // Check initial quota balls at root
  const ballCount = await page.$$eval('.quota-ball', els => els.length);
  assert(ballCount === 3, `Initial quota balls = n = 3 (got ${ballCount})`);
}

async function testStepAnimation() {
  console.log('\n--- Step Animation Tests ---');

  // Click Step button
  await page.click('#btn-step');
  await new Promise(r => setTimeout(r, 1200));

  // After first step, root should have been split
  const statusText = await page.$eval('#status-bar', el => el.textContent);
  assert(statusText.includes('Level'), `Status shows level info after step: "${statusText.substring(0, 60)}"`);

  // Split panel should appear
  const panelExists = await page.$$eval('.split-info', els => els.length);
  assert(panelExists >= 1, `Split distribution panel is visible after step (found ${panelExists})`);

  // Balls should now be at child level
  const ballCount = await page.$$eval('.quota-ball', els => els.length);
  assert(ballCount >= 1, `Quota balls exist after step (got ${ballCount})`);
}

async function testFullAnimation() {
  console.log('\n--- Full Animation Tests ---');

  // Click Play
  await page.click('#btn-play');
  // Wait for animation to complete
  await new Promise(r => setTimeout(r, 8000));

  const statusText = await page.$eval('#status-bar', el => el.textContent);
  assert(statusText.includes('Done'), `Animation completes: "${statusText.substring(0, 40)}"`);

  // Result should show selected items
  const resultHtml = await page.$eval('#result-bar', el => el.textContent);
  assert(resultHtml.includes('S') || resultHtml.includes('Sample'), `Result shows selected set: "${resultHtml.substring(0, 50)}"`);

  // Check leaves have status indicators (only real leaves, not padding)
  const leafStatusCount = await page.$$eval('.leaf-status', els => els.length);
  assert(leafStatusCount >= 6 && leafStatusCount <= 8, `Leaves have status indicators (got ${leafStatusCount})`);

  // Count selected (checkmarks)
  const selectedCount = await page.$$eval('.leaf-status', els =>
    els.filter(el => el.textContent === '✓').length
  );
  assert(selectedCount === 3, `Exactly n=3 items selected (got ${selectedCount})`);
}

async function testReset() {
  console.log('\n--- Reset Tests ---');

  await page.click('#btn-reset');
  await new Promise(r => setTimeout(r, 1000));

  const statusText = await page.$eval('#status-bar', el => el.textContent);
  assert(statusText.includes('Step') || statusText.includes('Press') || statusText.includes('Ready'), `Status reset after New Sample: "${statusText.substring(0, 40)}"`);

  const ballCount = await page.$$eval('.quota-ball', els => els.length);
  assert(ballCount === 3, `Reset restores n=3 quota balls (got ${ballCount})`);
}

async function testNnControls() {
  console.log('\n--- N/n Controls Tests ---');

  // Change n to 2
  await page.$eval('#inp-n', el => { el.value = '2'; el.dispatchEvent(new Event('change')); });
  await new Promise(r => setTimeout(r, 1000));

  const ballCount = await page.$$eval('.quota-ball', els => els.length);
  assert(ballCount === 2, `After n=2, quota balls = 2 (got ${ballCount})`);

  // Change N to 4
  await page.$eval('#inp-N', el => { el.value = '4'; el.dispatchEvent(new Event('change')); });
  await new Promise(r => setTimeout(r, 1000));

  const nodeCount = await page.$$eval('.tree-node', els => els.length);
  assert(nodeCount >= 4, `After N=4, at least 4 nodes (got ${nodeCount})`);
  assert(nodeCount <= 7, `After N=4, at most 7 nodes (got ${nodeCount})`);

  // Run full animation with N=4, n=2 and verify
  await page.click('#btn-play');
  await new Promise(r => setTimeout(r, 6000));

  const selectedCount = await page.$$eval('.leaf-status', els =>
    els.filter(el => el.textContent === '✓').length
  );
  assert(selectedCount === 2, `N=4,n=2: exactly 2 items selected (got ${selectedCount})`);
}

async function testStyleConsistency() {
  console.log('\n--- Style Consistency Tests ---');

  // Reset to default
  await page.$eval('#inp-N', el => { el.value = '8'; el.dispatchEvent(new Event('change')); });
  await page.$eval('#inp-n', el => { el.value = '3'; el.dispatchEvent(new Event('change')); });
  await new Promise(r => setTimeout(r, 1000));

  // Check widget box has correct background gradient
  const bgStyle = await page.$eval('.widget-box', el => getComputedStyle(el).backgroundImage);
  assert(bgStyle.includes('gradient') || bgStyle.includes('linear'), `Widget box has gradient background`);

  // Check node boxes use correct fill
  const nodeFill = await page.$eval('.tree-node rect', el => el.getAttribute('fill'));
  assert(nodeFill === '#f9f9f9' || nodeFill === '#e3f2fd' || nodeFill === '#e8f0fa', `Node fill matches style: ${nodeFill}`);

  // Check font family on SVG text
  const svgFontRule = await page.evaluate(() => {
    const rules = [...document.styleSheets].flatMap(s => {
      try { return [...s.cssRules]; } catch { return []; }
    });
    return rules.find(r => r.selectorText === 'svg text')?.style?.fontFamily || '';
  });
  assert(svgFontRule.includes('Garamond') || svgFontRule.includes('Georgia'), `SVG text uses serif font: "${svgFontRule}"`);

  // Check ball color matches CW
  const ballFill = await page.$eval('.quota-ball', el => el.getAttribute('fill'));
  assert(ballFill === '#5b9bd5', `Ball color is CW (#5b9bd5): ${ballFill}`);

  // Edge style
  const edgeStroke = await page.$eval('.tree-edge', el => el.getAttribute('stroke'));
  assert(edgeStroke === '#ddd' || edgeStroke === '#ccc', `Edge stroke is gray: ${edgeStroke}`);
}

async function testAlgorithmCorrectness() {
  console.log('\n--- Algorithm Correctness Tests ---');

  // Run the animation 10 times and verify invariants
  let allValid = true;
  for (let trial = 0; trial < 10; trial++) {
    await page.click('#btn-reset');
    await new Promise(r => setTimeout(r, 300));
    await page.click('#btn-play');
    await new Promise(r => setTimeout(r, 6000));

    const selectedCount = await page.$$eval('.leaf-status', els =>
      els.filter(el => el.textContent === '✓').length
    );
    if (selectedCount !== 3) {
      allValid = false;
      break;
    }
  }
  assert(allValid, `10 trials all produce exactly n=3 selected items`);
}

// Run all tests
(async () => {
  try {
    const jsErrors = await setup();
    await testPageLoads(jsErrors);
    await testTreeStructure();
    await testStepAnimation();
    await testFullAnimation();
    await testReset();
    await testNnControls();
    await testStyleConsistency();
    await testAlgorithmCorrectness();
  } catch (e) {
    console.error('Test runner error:', e);
    testsFailed++;
  } finally {
    await teardown();
    console.log(`\n=== Results: ${testsPassed} passed, ${testsFailed} failed ===`);
    if (errors.length > 0) {
      console.log('Failures:');
      errors.forEach(e => console.log('  -', e));
    }
    process.exit(testsFailed > 0 ? 1 : 0);
  }
})();
