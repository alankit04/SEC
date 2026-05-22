# RAPHI

RAPHI is a local research assistant for SEC and market analysis. It gathers evidence (SEC filings + market context), runs guarded AI synthesis, and produces citation-backed research responses in a browser UI.

## What This Project Does

- Accepts a ticker or research question in chat.
- Pulls SEC filing data and market context.
- Uses guardrailed AI flows to generate a structured answer.
- Enforces evidence and quality checks (citations, unsupported-claim controls, trace/run records).
- Provides a local web UI and API endpoints for research workflows.

## Why It Is Useful

- Faster first-pass equity research with linked evidence.
- Better transparency than a plain chatbot (citations + run status).
- Local-first workflow for development, testing, and iteration.
- Useful for analysts, AI engineers, and teams building auditable research assistants.

## 5-Minute Quick Start

1. Create and activate the environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

2. Create a minimal .env file.

```bash
RAPHI_API_KEY=local-development-key
RAPHI_INTERNAL_TOKEN=local-internal-token
ANTHROPIC_API_KEY=your_anthropic_key
```

3. Start the app.

```bash
node server.js
```

4. Open the UI.

```text
http://localhost:9999
```

## Quick Check

```bash
source .venv/bin/activate
pytest -q
```

## Notes

- RAPHI is a research tool, not an auto-trading system.
- Outputs should be reviewed before making investment decisions.
