from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
USER_DATA_ROOT = BASE_DIR / "data" / "users"

_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:@+-]{1,127}$")


def _clean(part: str, fallback: str) -> str:
    value = str(part or "").strip().lower()
    if not value:
        return fallback
    if not _ID_RE.match(value):
        value = re.sub(r"[^a-z0-9_.:@+-]", "_", value)
    return value[:128] if value else fallback


def split_scope(user_scope: str) -> tuple[str, str]:
    raw = str(user_scope or "").strip()
    if ":" in raw:
        tenant, user = raw.split(":", 1)
    else:
        tenant, user = "local", raw or "anonymous"
    return _clean(tenant, "local"), _clean(user, "anonymous")


def user_dir(user_scope: str) -> Path:
    tenant, user = split_scope(user_scope)
    path = USER_DATA_ROOT / tenant / user
    path.mkdir(parents=True, exist_ok=True)
    return path


def settings_path(user_scope: str) -> Path:
    return user_dir(user_scope) / "settings.json"


def portfolio_path(user_scope: str) -> Path:
    return user_dir(user_scope) / "portfolio.json"


def compliance_path(user_scope: str) -> Path:
    return user_dir(user_scope) / "compliance.json"


def load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(default)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return dict(default)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    path.chmod(0o600)
