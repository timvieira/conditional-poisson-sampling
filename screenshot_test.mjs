import puppeteer from 'puppeteer-core';
import { readFileSync } from 'fs';
import { resolve } from 'path';
import http from 'http';

const server = http.createServer((req, res) => {
  let urlPath = req.url.split('?')[0];
  if (urlPath.endsWith('/')) urlPath += 'index.html';
  let filePath = resolve('output', urlPath.replace(/^\//, ''));
  try {
    let content = readFileSync(filePath, 'utf8');
    const ext = filePath.split('.').pop();
    if (ext === 'html') {
      content = content.replace('https://d3js.org/d3.v7.min.js', '/d3.min.js');
      content = content.replace(
        /<script>\s*MathJax[\s\S]*?<\/script>\s*<script src="https:\/\/cdn\.jsdelivr[^"]*"><\/script>/,
        ''
      );
    }
    res.writeHead(200, { 'Content-Type': ext === 'html' ? 'text/html' : ext === 'js' ? 'application/javascript' : 'text/plain' });
    res.end(content);
  } catch {
    if (urlPath === '/d3.min.js') {
      try {
        res.writeHead(200, { 'Content-Type': 'application/javascript' });
        res.end(readFileSync(resolve('node_modules/d3/dist/d3.min.js')));
        return;
      } catch {}
    }
    res.writeHead(404); res.end('Not found');
  }
});

await new Promise(r => server.listen(0, r));
const port = server.address().port;

const browser = await puppeteer.launch({
  executablePath: '/root/.cache/ms-playwright/chromium-1194/chrome-linux/chrome',
  headless: true,
  args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage']
});
const page = await browser.newPage();
await page.setViewport({ width: 1200, height: 900 });
await page.goto(`http://localhost:${port}/sampling-animation/`, { waitUntil: 'domcontentloaded', timeout: 30000 });
await page.waitForSelector('svg', { timeout: 10000 });
await new Promise(r => setTimeout(r, 1000));

// Screenshot 1: Initial state
await page.screenshot({ path: '/tmp/anim-initial.png', fullPage: true });
console.log('Screenshot 1: initial state -> /tmp/anim-initial.png');

// Screenshot 2: After one step
await page.click('#btn-step');
await new Promise(r => setTimeout(r, 1000));
await page.screenshot({ path: '/tmp/anim-step1.png', fullPage: true });
console.log('Screenshot 2: after step 1 -> /tmp/anim-step1.png');

// Screenshot 3: After second step
await page.click('#btn-step');
await new Promise(r => setTimeout(r, 1000));
await page.screenshot({ path: '/tmp/anim-step2.png', fullPage: true });
console.log('Screenshot 3: after step 2 -> /tmp/anim-step2.png');

// Screenshot 4: Completed
await page.click('#btn-play');
await new Promise(r => setTimeout(r, 8000));
await page.screenshot({ path: '/tmp/anim-complete.png', fullPage: true });
console.log('Screenshot 4: completed -> /tmp/anim-complete.png');

await browser.close();
server.close();
