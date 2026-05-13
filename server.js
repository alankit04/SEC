/**
 * server.js — RAPHI preview shim
 * Spawns the FastAPI uvicorn backend and keeps it alive.
 * Used by .claude/launch.json so the Claude Preview tool can start the server.
 */
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const PROJECT_DIR = path.join(__dirname);
const UVICORN = path.join(PROJECT_DIR, '.venv', 'bin', 'uvicorn');

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

const server = spawn(UVICORN, [
  'backend.raphi_server:app',
  '--host', '0.0.0.0',
  '--port', '9999',
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

// Forward signals to child
['SIGINT', 'SIGTERM'].forEach((sig) => {
  process.on(sig, () => {
    server.kill(sig);
  });
});
