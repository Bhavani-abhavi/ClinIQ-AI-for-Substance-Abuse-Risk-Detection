import { defineConfig, devices } from '@playwright/test';

// One demo server per browser so parallel projects never share labeling state.
// PYTHON must point at an interpreter with the repo's labeling dependencies (fastapi, scikit-learn).
const python = process.env.PYTHON ?? 'python3';
const browsers = [
  { name: 'chromium', device: devices['Desktop Chrome'], port: 8771 },
  { name: 'firefox', device: devices['Desktop Firefox'], port: 8772 },
  { name: 'webkit', device: devices['Desktop Safari'], port: 8773 },
];

export default defineConfig({
  testDir: 'e2e',
  fullyParallel: false,
  retries: process.env.CI ? 1 : 0,
  reporter: [['list']],
  expect: { timeout: 10_000 },
  use: { trace: 'retain-on-failure' },
  projects: browsers.map((b) => ({ name: b.name, use: { ...b.device, baseURL: `http://127.0.0.1:${b.port}` } })),
  webServer: browsers.map((b) => ({
    command: `${python} -m labeling.server --demo --port ${b.port} --gold-every 4 --retrain-every 3 --overlap-rate 1.0`,
    cwd: '../..',
    url: `http://127.0.0.1:${b.port}/api/health`,
    reuseExistingServer: false,
    timeout: 60_000,
  })),
});
