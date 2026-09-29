import { expect, test, type Page } from '@playwright/test';

test.describe.configure({ mode: 'serial' });

async function signIn(page: Page, name: string) {
  await page.goto('/');
  await page.getByLabel('Your name').fill(name);
  await page.getByRole('button', { name: 'Start labeling' }).click();
  await expect(page.getByTestId('review')).toBeVisible();
}

async function labelsFor(page: Page, annotator: string) {
  const stats = await (await page.request.get('/api/stats')).json();
  return stats.annotators.find((a: { annotator: string }) => a.annotator === annotator)?.labels ?? 0;
}

test('a save whose response is lost is retried and stored exactly once', async ({ page }) => {
  await signIn(page, 'lost-response');
  let dropped = false;
  await page.route('**/api/labels', async (route) => {
    if (!dropped) {
      dropped = true;
      await route.fetch();                 // the server stores the label...
      await route.abort('connectionreset'); // ...but the browser never sees the answer
    } else {
      await route.continue();
    }
  });
  const first = await page.getByTestId('review').getAttribute('data-item');
  await page.waitForTimeout(350);                 // a person needs to see the item first
  await page.keyboard.press('1');
  await expect(page.getByTestId('session-count')).toHaveText('Labeled this session: 1');
  await expect(page.getByTestId('review')).not.toHaveAttribute('data-item', first!);
  expect(dropped).toBe(true);
  expect(await labelsFor(page, 'lost-response')).toBe(1);
});

test('a label saved while offline survives a reload and is sent once', async ({ page }) => {
  await signIn(page, 'offline');
  await page.route('**/api/labels', (route) => route.abort('internetdisconnected'));
  await page.waitForTimeout(350);                 // a person needs to see the item first
  await page.keyboard.press('0');
  await expect(page.getByTestId('save-failed')).toBeVisible({ timeout: 15_000 });
  await page.waitForTimeout(350);                 // a person needs to see the item first
  await page.keyboard.press('1');                              // blocked until the unsaved label is resolved
  expect(await labelsFor(page, 'offline')).toBe(0);
  await page.unroute('**/api/labels');
  await page.reload();
  await expect(page.getByTestId('notice')).toContainText('Saved 1 label left over from before');
  expect(await labelsFor(page, 'offline')).toBe(1);
});

test('a manual retry after a server error saves the label', async ({ page }) => {
  await signIn(page, 'server-error');
  let failures = 0;
  await page.route('**/api/labels', async (route) => {
    if (failures < 4) {                                         // outlast the automatic retries
      failures++;
      await route.fulfill({ status: 503, contentType: 'application/json', body: '{"detail":"maintenance"}' });
    } else {
      await route.continue();
    }
  });
  await page.waitForTimeout(350);                 // a person needs to see the item first
  await page.keyboard.press('1');
  await expect(page.getByTestId('save-failed')).toContainText('maintenance', { timeout: 15_000 });
  await page.getByRole('button', { name: 'Retry now' }).click();
  await expect(page.getByTestId('session-count')).toHaveText('Labeled this session: 1');
  expect(await labelsFor(page, 'server-error')).toBe(1);
});

test('a double press labels one review, not the next one unseen', async ({ page }) => {
  await signIn(page, 'double-press');
  await page.waitForTimeout(400);
  const first = await page.getByTestId('review').getAttribute('data-item');
  await page.waitForTimeout(350);                 // a person needs to see the item first
  await page.keyboard.press('1');
  await page.keyboard.press('1');                      // 2nd press arrives while or right after the next item loads
  await expect(page.getByTestId('session-count')).toHaveText('Labeled this session: 1');
  await page.waitForTimeout(600);
  await expect(page.getByTestId('session-count')).toHaveText('Labeled this session: 1');
  expect(await page.getByTestId('review').getAttribute('data-item')).not.toBe(first);
  expect(await labelsFor(page, 'double-press')).toBe(1);
});

test('a held key (auto-repeat) labels one review', async ({ page }) => {
  await signIn(page, 'held-key');
  await page.waitForTimeout(400);
  await page.keyboard.press('0');
  await page.evaluate(() => {                          // what a held key sends after the first keydown
    for (let i = 0; i < 5; i++) window.dispatchEvent(new KeyboardEvent('keydown', { key: '0', repeat: true }));
  });
  await page.waitForTimeout(800);
  await expect(page.getByTestId('session-count')).toHaveText('Labeled this session: 1');
  expect(await labelsFor(page, 'held-key')).toBe(1);
});
