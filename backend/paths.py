"""Shared filesystem paths for RAPHI."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SETTINGS_FILE = PROJECT_ROOT / "settings.json"
PORTFOLIO_FILE = PROJECT_ROOT / "portfolio.json"
MODEL_CACHE_DIR = PROJECT_ROOT / ".model_cache"
AUDIT_DIR = PROJECT_ROOT / ".raphi_audit"
COMPANY_TICKERS_FILE = PROJECT_ROOT / "company_tickers.json"

