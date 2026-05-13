import { spawn } from "node:child_process";
import fs from "node:fs";
import fsp from "node:fs/promises";
import os from "node:os";
import path from "node:path";

const ROOT = process.cwd();
const APP_URL = process.env.RAPHI_E2E_URL || "http://127.0.0.1:9999";
const DEBUG_PORT = Number(process.env.RAPHI_E2E_DEBUG_PORT || 9333);
const CHROME = process.env.CHROME_BIN || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const PROFILE = await fsp.mkdtemp(path.join(os.tmpdir(), "raphi-e2e-"));
const PORTFOLIO_FILE = path.join(ROOT, "portfolio.json");
const BACKUP = fs.existsSync(PORTFOLIO_FILE) ? await fsp.readFile(PORTFOLIO_FILE, "utf8") : null;

function loadEnv() {
  const envPath = path.join(ROOT, ".env");
  if (!fs.existsSync(envPath)) return {};
  return Object.fromEntries(
    fs.readFileSync(envPath, "utf8")
      .split(/\r?\n/)
      .map(line => line.trim())
      .filter(line => line && !line.startsWith("#") && line.includes("="))
      .map(line => {
        const idx = line.indexOf("=");
        const key = line.slice(0, idx);
        const val = line.slice(idx + 1).replace(/^['"]|['"]$/g, "");
        return [key, val];
      })
  );
}

const envFile = loadEnv();
const apiKey = process.env.RAPHI_API_KEY || envFile.RAPHI_API_KEY || "";
if (!apiKey) {
  throw new Error("RAPHI_API_KEY is required in environment or .env for E2E login.");
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function rmRetry(target) {
  for (let i = 0; i < 6; i += 1) {
    try {
      await fsp.rm(target, { recursive: true, force: true });
      return;
    } catch (error) {
      if (!["ENOTEMPTY", "EBUSY", "EPERM"].includes(error.code)) throw error;
      await sleep(250 * (i + 1));
    }
  }
  await fsp.rm(target, { recursive: true, force: true });
}

async function fetchJson(url, options = {}, timeoutMs = 8000) {
  const started = Date.now();
  let lastError;
  while (Date.now() - started < timeoutMs) {
    try {
      const response = await fetch(url, options);
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      return await response.json();
    } catch (error) {
      lastError = error;
      await sleep(250);
    }
  }
  throw lastError;
}

class CDP {
  constructor(wsUrl) {
    this.wsUrl = wsUrl;
    this.id = 0;
    this.pending = new Map();
  }

  async connect() {
    this.ws = new WebSocket(this.wsUrl);
    await new Promise((resolve, reject) => {
      this.ws.addEventListener("open", resolve, { once: true });
      this.ws.addEventListener("error", reject, { once: true });
    });
    this.ws.addEventListener("message", event => {
      const msg = JSON.parse(event.data);
      if (!msg.id || !this.pending.has(msg.id)) return;
      const { resolve, reject } = this.pending.get(msg.id);
      this.pending.delete(msg.id);
      if (msg.error) reject(new Error(msg.error.message || JSON.stringify(msg.error)));
      else resolve(msg.result);
    });
  }

  send(method, params = {}) {
    const id = ++this.id;
    this.ws.send(JSON.stringify({ id, method, params }));
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      setTimeout(() => {
        if (this.pending.has(id)) {
          this.pending.delete(id);
          reject(new Error(`CDP timeout: ${method}`));
        }
      }, 30000);
    });
  }

  async eval(expression, awaitPromise = false) {
    const result = await this.send("Runtime.evaluate", {
      expression,
      awaitPromise,
      returnByValue: true,
      userGesture: true,
    });
    if (result.exceptionDetails) {
      throw new Error(result.exceptionDetails.text || "Runtime exception");
    }
    return result.result?.value;
  }

  close() {
    this.ws?.close();
  }
}

async function waitFor(cdp, expression, label, timeoutMs = 25000) {
  const started = Date.now();
  let last;
  while (Date.now() - started < timeoutMs) {
    last = await cdp.eval(expression).catch(error => error.message);
    if (last === true) return;
    await sleep(300);
  }
  throw new Error(`Timed out waiting for ${label}. Last value: ${last}`);
}

async function click(cdp, selector, label = selector) {
  const ok = await cdp.eval(`(() => {
    const el = document.querySelector(${JSON.stringify(selector)});
    if (!el) return false;
    el.click();
    return true;
  })()`);
  if (!ok) throw new Error(`Could not click ${label}`);
}

async function clickByText(cdp, scope, text) {
  const ok = await cdp.eval(`(() => {
    const root = document.querySelector(${JSON.stringify(scope)});
    if (!root) return false;
    const el = [...root.querySelectorAll('button,.tab,.nav-item')].find(node => node.textContent.trim().includes(${JSON.stringify(text)}));
    if (!el) return false;
    el.click();
    return true;
  })()`);
  if (!ok) throw new Error(`Could not click ${text} in ${scope}`);
}

async function assertActivePage(cdp, page) {
  await waitFor(
    cdp,
    `document.querySelector('#page-${page}.active') !== null`,
    `active page ${page}`
  );
}

async function main() {
  const chrome = spawn(CHROME, [
    "--headless=new",
    `--remote-debugging-port=${DEBUG_PORT}`,
    `--user-data-dir=${PROFILE}`,
    "--no-first-run",
    "--disable-gpu",
    "--disable-dev-shm-usage",
    "about:blank",
  ], { stdio: "ignore" });

  let cdp;
  try {
    const targets = await fetchJson(`http://127.0.0.1:${DEBUG_PORT}/json/list`);
    const page = targets.find(t => t.type === "page") || targets[0];
    cdp = new CDP(page.webSocketDebuggerUrl);
    await cdp.connect();
    await cdp.send("Page.enable");
    await cdp.send("Runtime.enable");
    await cdp.send("Page.addScriptToEvaluateOnNewDocument", {
      source: `
        localStorage.setItem('raphi_api_key', ${JSON.stringify(apiKey)});
        localStorage.setItem('raphi_onboarded', '1');
        localStorage.setItem('raphi_initials', 'AL');
      `,
    });
    await cdp.send("Page.navigate", { url: APP_URL });
    await waitFor(cdp, "document.querySelector('#main-app.active') !== null", "authenticated app shell");

    for (const pageId of ["dashboard", "ask", "stock", "signals", "news", "portfolio", "shap", "memo", "convictions", "research", "alerts", "models", "settings"]) {
      await click(cdp, `.nav-item[onclick*="'${pageId}'"]`, `nav ${pageId}`);
      await assertActivePage(cdp, pageId);
    }

    await click(cdp, ".topbar-logo", "RAPHI logo");
    await assertActivePage(cdp, "dashboard");

    await click(cdp, ".topbar-btn[data-tip='Alerts']", "topbar alerts");
    await assertActivePage(cdp, "alerts");
    await click(cdp, ".topbar-btn[data-tip='Settings']", "topbar settings");
    await assertActivePage(cdp, "settings");

    await cdp.eval(`(() => {
      const input = document.querySelector('.topbar-search input');
      input.value = 'AAPL';
      input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    })()`);
    await assertActivePage(cdp, "stock");
    await waitFor(cdp, "document.querySelector('#page-stock .page-title h1')?.textContent.trim() === 'AAPL'", "ticker search navigates to AAPL");

    for (const tab of ["Overview", "Technicals", "Fundamentals", "Options Flow", "SEC Filings"]) {
      await clickByText(cdp, "#page-stock .tabs", tab);
      await waitFor(cdp, "document.querySelector('#stock-tab-content')?.innerText.trim().length > 20 && !document.querySelector('#stock-tab-content')?.innerText.includes('Loading')", `stock tab ${tab}`, 45000);
    }

    await clickByText(cdp, "#page-stock", "Add to Portfolio");
    await waitFor(cdp, "document.querySelector('#page-stock .btn-teal')?.textContent.includes('Added')", "Add to Portfolio button result", 45000);

    await click(cdp, `.nav-item[onclick*="'signals'"]`, "nav signals");
    for (const tab of ["All", "Equities", "Macro", "Crypto", "Fixed Income"]) {
      await clickByText(cdp, "#page-signals .tabs", tab);
      await waitFor(cdp, "document.querySelector('#signals-tab-content')?.innerText.trim().length > 20 && !document.querySelector('#signals-tab-content')?.innerText.includes('Loading')", `signals tab ${tab}`, 45000);
    }

    await click(cdp, `.nav-item[onclick*="'shap'"]`, "nav explainability");
    for (const tab of ["NVDA Forecast", "Portfolio Drivers", "Model Comparison"]) {
      await clickByText(cdp, "#page-shap .tabs", tab);
      await waitFor(cdp, "document.querySelector('#shap-tab-content')?.innerText.trim().length > 20 && !document.querySelector('#shap-tab-content')?.innerText.includes('Loading')", `explainability tab ${tab}`, 45000);
    }

    await click(cdp, `.nav-item[onclick*="'ask'"]`, "nav ask");
    await clickByText(cdp, "#page-ask", "New Thread");
    await waitFor(cdp, "document.querySelector('#consoleMessages')?.innerText.includes('New thread ready')", "new thread action");

    await click(cdp, `.nav-item[onclick*="'portfolio'"]`, "nav portfolio");
    await clickByText(cdp, "#page-portfolio", "Stress Test");
    await waitFor(cdp, "document.querySelector('#portfolio-action-panel')?.innerText.includes('Stress Test')", "stress test action");
    await clickByText(cdp, "#page-portfolio", "Run Scenario");
    await waitFor(cdp, "document.querySelector('#portfolio-action-panel')?.innerText.includes('Scenario')", "scenario action");

    await click(cdp, `.nav-item[onclick*="'alerts'"]`, "nav alerts");
    await clickByText(cdp, "#page-alerts", "Create Alert");
    await waitFor(cdp, "document.querySelector('#page-alerts')?.innerText.includes('price and signal watch')", "create alert action");

    await click(cdp, `.nav-item[onclick*="'models'"]`, "nav models");
    await clickByText(cdp, "#page-models", "Run Backtest");
    await waitFor(cdp, "document.querySelector('#model-backtest-panel')?.innerText.includes('Backtest Snapshot')", "backtest action");

    console.log("E2E smoke passed: navigation, logo home, tabs, and key action buttons are working.");
  } finally {
    cdp?.close();
    chrome.kill("SIGTERM");
    await new Promise(resolve => {
      const timer = setTimeout(resolve, 2000);
      chrome.once("exit", () => {
        clearTimeout(timer);
        resolve();
      });
    });
    if (BACKUP !== null) await fsp.writeFile(PORTFOLIO_FILE, BACKUP);
    else if (fs.existsSync(PORTFOLIO_FILE)) await fsp.rm(PORTFOLIO_FILE);
    await rmRetry(PROFILE);
  }
}

main().catch(async error => {
  if (BACKUP !== null) await fsp.writeFile(PORTFOLIO_FILE, BACKUP);
  else if (fs.existsSync(PORTFOLIO_FILE)) await fsp.rm(PORTFOLIO_FILE);
  await rmRetry(PROFILE).catch(() => {});
  console.error(error.stack || error.message);
  process.exit(1);
});
