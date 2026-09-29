import { expect, test, type Page } from '@playwright/test';

// Serial flow per browser: two annotators, a gold check, a disagreement, adjudication and export.
test.describe.configure({ mode: 'serial' });

async function signIn(page: Page, name: string) {
  await page.goto('/');
  await page.getByLabel('Your name').fill(name);
  await page.getByRole('button', { name: 'Start labeling' }).click();
  await expect(page.getByTestId('annotator')).toHaveText(name);
}

async function currentItem(page: Page) {
  const review = page.getByTestId('review');
  await expect(review).toBeVisible();
  return review.getAttribute('data-item');
}

test('annotators label, disagree, and a reviewer resolves the conflict', async ({ browser }) => {
  const ana = await (await browser.newContext()).newPage();
  await signIn(ana, 'ana');
  await expect(ana.getByTestId('suggestion')).toContainText('No model yet');

  // Keyboard labeling: four tasks (the fourth is a gold item), model retrains after three.
  const first = await currentItem(ana);
  let sawHighlight = false;
  for (const [k, key] of ['1', '0', '1', '0'].entries()) {
    const before = await currentItem(ana);
    sawHighlight ||= (await ana.locator('blockquote mark').count()) > 0;
    await ana.waitForTimeout(350);                   // a person needs to see the item first
    await ana.keyboard.press(key);
    await expect(ana.getByTestId('session-count')).toHaveText(`Labeled this session: ${k + 1}`);
    await expect.poll(() => currentItem(ana)).not.toBe(before);
  }
  await expect(ana.getByTestId('suggestion')).toContainText('Model suggests:');
  expect(sawHighlight).toBe(true);

  // Typing in a field must not trigger shortcuts.
  await ana.getByRole('button', { name: 'switch' }).click();
  await ana.getByLabel('Your name').fill('ana 01');
  await expect(ana.getByLabel('Your name')).toHaveValue('ana 01');
  await ana.getByLabel('Your name').fill('ana');
  await ana.getByRole('button', { name: 'Start labeling' }).click();

  // Second annotator gets ana's first item for a second opinion and disagrees with the mouse.
  const ben = await (await browser.newContext()).newPage();
  await signIn(ben, 'ben');
  expect(await currentItem(ben)).toBe(first);
  await ben.waitForTimeout(350);
  await ben.getByRole('button', { name: /Not relevant/ }).click();
  await expect(ben.getByTestId('session-count')).toHaveText('Labeled this session: 1');

  // Quality view: the conflict, ana's gold score, and adjudication.
  await ben.getByRole('button', { name: 'Quality' }).click();
  await expect(ben.getByTestId('conflict-count')).toHaveText('1');
  await expect(ben.getByTestId('row-ana')).toContainText('(1 gold)');
  await ben.getByRole('button', { name: 'Resolve as SUD-relevant' }).click();
  await expect(ben.getByTestId('no-conflicts')).toBeVisible();
  await expect(ben.getByTestId('conflict-count')).toHaveText('0');

  // Export carries the adjudicated label.
  const exported = await ben.request.get('/api/export');
  const rows = (await exported.text()).trim().split('\n').map((l) => JSON.parse(l));
  const resolved = rows.find((r) => r.id === first);
  expect(resolved).toMatchObject({ label: 1, adjudicated: true, annotations: 2 });
});

test('layout works on a phone-sized screen', async ({ browser }) => {
  const page = await (await browser.newContext({ viewport: { width: 375, height: 740 } })).newPage();
  await signIn(page, 'mobile');
  await currentItem(page);
  const width = await page.evaluate(() => document.documentElement.scrollWidth);
  expect(width).toBeLessThanOrEqual(375);
  await page.waitForTimeout(350);
  await page.getByRole('button', { name: /SUD-relevant/ }).click();
  await expect(page.getByTestId('session-count')).toHaveText('Labeled this session: 1');
});
