// Captures the workbench flow as screenshots on the synthetic demo data, for docs/labeling-architecture.md:
// queue -> suggestion -> labels -> second opinion -> quality view -> adjudication -> export.
//   PYTHON=python node storyboard.mjs ../../docs/images
import { chromium } from '@playwright/test';
import { spawn } from 'node:child_process';
import { mkdirSync } from 'node:fs';
import { join } from 'node:path';

const out = process.argv[2] ?? 'storyboard';
mkdirSync(out, { recursive: true });
const port = 8799;
const server = spawn(process.env.PYTHON ?? 'python3',
  ['-m', 'labeling.server', '--demo', '--port', String(port), '--gold-every', '4', '--retrain-every', '3', '--overlap-rate', '1.0'],
  { cwd: '../..', stdio: 'ignore' });
const base = `http://127.0.0.1:${port}`;
for (let i = 0; i < 60; i++) {
  try { if ((await fetch(`${base}/api/health`)).ok) break; } catch { /* starting */ }
  await new Promise((r) => setTimeout(r, 500));
}
const browser = await chromium.launch();
const page = await (await browser.newContext({ viewport: { width: 1000, height: 640 }, colorScheme: 'light' })).newPage();
const shot = (name) => page.screenshot({ path: join(out, `labeling-${name}.png`) });
try {
  await page.goto(base);
  await page.getByLabel('Your name').fill('ana');
  await page.getByRole('button', { name: 'Start labeling' }).click();
  await page.getByTestId('review').waitFor();
  await shot('1-queue-cold-start');
  for (const key of ['1', '0', '1']) { await page.keyboard.press(key); await page.waitForTimeout(250); }
  await page.getByTestId('suggestion').filter({ hasText: 'Model suggests' }).waitFor();
  await shot('2-model-suggestion');
  await page.keyboard.press('0');
  await page.waitForTimeout(250);
  await page.getByRole('button', { name: 'switch' }).click();
  await page.getByLabel('Your name').fill('ben');
  await page.getByRole('button', { name: 'Start labeling' }).click();
  await page.getByTestId('review').waitFor();
  await page.getByRole('button', { name: /Not relevant/ }).click();
  await page.waitForTimeout(300);
  await page.getByRole('button', { name: 'Quality' }).click();
  await page.getByTestId('conflict').waitFor();
  await shot('3-quality-and-conflict');
  await page.getByRole('button', { name: 'Resolve as SUD-relevant' }).click();
  await page.getByTestId('no-conflicts').waitFor();
  await page.getByText('Export resolved labels').scrollIntoViewIfNeeded();
  await shot('4-adjudicated-and-export');
} finally {
  await browser.close();
  server.kill();
}
console.log('saved to', out);
