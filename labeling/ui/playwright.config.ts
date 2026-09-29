import { defineConfig, devices } from '@playwright/test';

// One demo server per browser so parallel projects never share labeling state.
// PYTHON must point at an interpreter with the repo's labeling dependencies (fastapi, scikit-learn).
const python = process.env.PYTHON ?? 'python3';
const browsers = [
  { name: 'chromium', device: devices['Desktop Chrome'], port: 8771 },
  { name: 'firefox', device: devices['Desktop Firefox'], port: 8772 },
  { name: 'webkit', device: devices['Desktop Safari'], port: 8773 },
];
// Each spec file gets its own servers: the labeling flow depends on queue order, which other tests would change.
const suites = [
  { spec: 'workbench.spec.ts', suffix: '', offset: 0 },
  { spec: 'reliability.spec.ts', suffix: '-reliability', offset: 10 },
];
const runs = suites.flatMap((s) => browsers.map((b) => ({ ...b, spec: s.spec, project: b.name + s.suffix, port: b.port + s.offset })));

export default defineConfig({
  testDir: 'e2e',
  fullyParallel: false,
  retries: process.env.CI ? 1 : 0,
  reporter: [['list']],
  expect: { timeout: 10_000 },
  use: { trace: 'retain-on-failure' },
  projects: runs.map((r) => ({ name: r.project, testMatch: r.spec, use: { ...r.device, baseURL: `http://127.0.0.1:${r.port}` } })),
  webServer: runs.map((b) => ({
    command: `${python} -m labeling.server --demo --port ${b.port} --gold-every 4 --retrain-every 3 --overlap-rate 1.0`,
    cwd: '../..',
    url: `http://127.0.0.1:${b.port}/api/health`,
    reuseExistingServer: false,
    timeout: 60_000,
  })),
});
