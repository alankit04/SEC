#!/usr/bin/env python3
"""Run RAPHI's formal eval harness from the repository root."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from eval_harness import run_cli  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(run_cli())
