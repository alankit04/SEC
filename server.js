/**
 * server.js — RAPHI preview shim
 * Spawns the FastAPI uvicorn backend and keeps it alive.
 * Used by .claude/launch.json so the Claude Preview tool can start the server.
 */
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const http = require('http');

const PROJECT_DIR = path.join(__dirname);
const UVICORN = path.join(PROJECT_DIR, '.venv', 'bin', 'uvicorn');
const HOST = '127.0.0.1';
const PORT = 9999;
const APP_URL = `http://${HOST}:${PORT}/static/index.html`;

function probeServer(timeoutMs = 1200) {
  return new Promise((resolve) => {
    const req = http.get(
      {
        hostname: HOST,
        port: PORT,
        path: '/api/health',
        timeout: timeoutMs,
      },
      (res) => {
        res.resume();
        resolve(res.statusCode && res.statusCode < 500);
      }
    );
    req.on('timeout', () => {
      req.destroy();
      resolve(false);
    });
    req.on('error', () => resolve(false));
  });
}

async function waitForServerReady(maxWaitMs = 20000) {
  const started = Date.now();
  while (Date.now() - started < maxWaitMs) {
    // eslint-disable-next-line no-await-in-loop
    const ok = await probeServer();
    if (ok) return true;
    // eslint-disable-next-line no-await-in-loop
    await new Promise((r) => setTimeout(r, 350));
  }
  return false;
}

// Load .env file so uvicorn inherits all secrets (ANTHROPIC_API_KEY etc.)
const envFile = path.join(PROJECT_DIR, '.env');
const dotenv = {};
if (fs.existsSync(envFile)) {
  fs.readFileSync(envFile, 'utf8').split('\n').forEach(line => {
    const m = line.match(/^([A-Z_][A-Z0-9_]*)=(.*)$/);
    if (m) dotenv[m[1]] = m[2];
  });
  console.log('[RAPHI] Loaded .env:', Object.keys(dotenv).join(', '));
}

console.log('[RAPHI] Starting unified server (A2A + API) on port 9999...');
console.log(`[RAPHI] uvicorn: ${UVICORN}`);
console.log(`[RAPHI] cwd: ${PROJECT_DIR}`);

(async () => {
  if (await probeServer()) {
    console.log('[RAPHI] Backend already running.');
    console.log(`[RAPHI] Open: ${APP_URL}`);
    process.exit(0);
  }

  const server = spawn(UVICORN, [
    'backend.raphi_server:app',
    '--host', HOST,
    '--port', String(PORT),
    '--reload',
  ], {
    cwd: PROJECT_DIR,
    env: { ...process.env, ...dotenv },
    stdio: 'inherit',
  });

  server.on('error', (err) => {
    console.error('[RAPHI] Failed to start server:', err.message);
    process.exit(1);
  });

  server.on('exit', (code) => {
    console.log(`[RAPHI] Server exited with code ${code}`);
    process.exit(code || 0);
  });

  const ready = await waitForServerReady();
  if (ready) {
    console.log(`[RAPHI] Ready: ${APP_URL}`);
  } else {
    console.warn('[RAPHI] Server is still starting; check logs above.');
  }

  // Forward signals to child
  ['SIGINT', 'SIGTERM'].forEach((sig) => {
    process.on(sig, () => {
      server.kill(sig);
    });
  });
})();
