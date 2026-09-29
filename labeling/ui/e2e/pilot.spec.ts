import { expect, test, type Page } from '@playwright/test';

test.describe.configure({ mode: 'serial' });

async function signIn(page: Page, name: string) {
  await page.goto('/');
  await page.getByLabel('Your name').fill(name);
  await page.getByRole('button', { name: 'Start labeling' }).click();
  await expect(page.getByTestId('review')).toBeVisible();
}

test('pilot alternates assisted and manual items and counterbalances annotators', async ({ browser }) => {
  const p1 = await (await browser.newContext()).newPage();
  await signIn(p1, 'pilot-one');
  await expect(p1.getByRole('heading', { name: 'ClinIQ labeling pilot' })).toBeVisible();
  await expect(p1.getByRole('button', { name: 'Quality' })).toHaveCount(0);
  await expect(p1.getByRole('button', { name: /Skip/ })).toHaveCount(0);
  await expect(p1.getByTestId('pilot-progress')).toHaveText(/Item 1 of \d+/);
  await expect(p1.getByTestId('suggestion')).toContainText('Model suggests');          // assisted first
  await p1.waitForTimeout(350);                 // a person needs to see the item first
  await p1.keyboard.press('1');
  await expect(p1.getByTestId('pilot-progress')).toHaveText(/Item 2 of/);
  await expect(p1.getByTestId('suggestion')).toContainText('Label this one on your own'); // then manual
  await expect(p1.locator('blockquote mark')).toHaveCount(0);
  await p1.waitForTimeout(350);                 // a person needs to see the item first
  await p1.keyboard.press('s');                                                          // no skipping
  await expect(p1.getByTestId('pilot-progress')).toHaveText(/Item 2 of/);
  await p1.waitForTimeout(350);                 // a person needs to see the item first
  await p1.keyboard.press('0');
  await expect(p1.getByTestId('pilot-progress')).toHaveText(/Item 3 of/);

  const p2 = await (await browser.newContext()).newPage();
  await signIn(p2, 'pilot-two');
  await expect(p2.getByTestId('suggestion')).toContainText('Label this one on your own'); // opposite order
  await p2.waitForTimeout(350);                 // a person needs to see the item first
  await p2.keyboard.press('1');
  await expect(p2.getByTestId('suggestion')).toContainText('Model suggests');

  const report = await (await p2.request.get('/api/pilot/report')).json();
  expect(report.labels).toBe(3);
  expect(report.by_condition.assisted.labels + report.by_condition.manual.labels).toBe(3);
});
