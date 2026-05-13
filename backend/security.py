"""
security.py — Shared security utilities for RAPHI

Fixes addressed:
  C2  TokenAuth middleware — Bearer / X-API-Key auth on A2A server
  H2  sanitize_user_input — prompt injection guard + 4000-char cap
  H3  SessionCipher — Fernet encryption for sessions.json
  Sentry  init_sentry — runtime monitoring wired to raphi.sentry.io

Usage:
    from backend.security import TokenAuth, sanitize_user_input, init_sentry, SessionCipher
"""

import hashlib
import logging
import os
import re

from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("raphi.security")

# ── Public paths that skip auth ──────────────────────────────────────
PUBLIC_PATHS = {
    "/.well-known/agent.json",        # A2A agent card (legacy path)
    "/.well-known/agent-card.json",   # A2A agent card (current A2A spec)
    "/health",
    "/api/health",                    # unified server health probe
    "/",                              # dashboard HTML (GET only — POST / is A2A, checked below)
    "/static",                        # static assets prefix
    "/docs",                          # FastAPI Swagger UI
    "/openapi.json",
}

# Paths that require auth regardless of method (A2A task endpoint)
AUTH_REQUIRED_PATHS = {"/"}          # POST / = A2A task submission needs a key

# ── Prompt injection patterns ────────────────────────────────────────
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?|context)",
    r"you\s+are\s+now\s+a?\s*(different|new|another|evil|uncensored)",
    r"(print|reveal|show|output|display|leak)\s+(the\s+)?(api[\s_]?key|secret|password|token|credential)",
    r"disregard\s+(all\s+)?(previous|prior|above)",
    r"new\s+(system\s+)?instructions?\s*:",
    r"(system|assistant)\s*:\s*(you|ignore|forget|override)",
    r"<\s*system\s*>",
    r"\[\s*system\s*\]",
    r"<\s*/?instructions?\s*>",
    r"forget\s+(everything|all|your|previous)",
    r"(override|bypass)\s+(your\s+)?(safety|rules?|guidelines?|instructions?)",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _INJECTION_PATTERNS]
MAX_INPUT_LENGTH = 4000


# ── C2: Bearer / X-API-Key / X-Internal-Token authentication middleware ──
class TokenAuth:
    """
    Pure ASGI middleware: validates two token types:
      - X-API-Key or Authorization: Bearer  →  external client auth (RAPHI_API_KEY)
      - X-Internal-Token                    →  MCP bridge auth (RAPHI_INTERNAL_TOKEN, H1/M3)
    Written as a raw ASGI callable (not BaseHTTPMiddleware) to preserve
    FastAPI 0.115+ dependency-injection scope (fastapi_middleware_astack).
    """

    def __init__(self, app, api_key: str, internal_token: str = "") -> None:
        self.app = app
        self._api_key = api_key
        self._internal_token = internal_token  # H1/M3: MCP bridge shared secret

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        path = request.url.path

        if path.startswith("/static/"):
            await self.app(scope, receive, send)
            return

        if path in PUBLIC_PATHS and not (path in AUTH_REQUIRED_PATHS and request.method == "POST"):
            await self.app(scope, receive, send)
            return

        # H1/M3: MCP bridge uses X-Internal-Token — validate before external key check
        internal = request.headers.get("X-Internal-Token", "").strip()
        if internal:
            if self._internal_token and internal == self._internal_token:
                await self.app(scope, receive, send)
                return
            # Token present but wrong — reject immediately (don't fall through to API key)
            _capture_security_event(
                "[RAPHI] Invalid X-Internal-Token on MCP bridge call",
                level="warning",
                path=path,
                ip=request.client.host if request.client else "unknown",
            )
            response = Response(
                content='{"error": "Unauthorized. Invalid internal token."}',
                status_code=401,
                media_type="application/json",
            )
            await response(scope, receive, send)
            return

        # External client auth: X-API-Key or Authorization: Bearer
        token = (
            request.headers.get("X-API-Key")
            or request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        )

        if not self._api_key:
            logger.warning("RAPHI_API_KEY not set — A2A server is UNPROTECTED")
            await self.app(scope, receive, send)
            return

        if not token or token != self._api_key:
            _capture_security_event(
                "[RAPHI] Unauthorized A2A access attempt",
                level="warning",
                path=str(request.url.path),
                ip=request.client.host if request.client else "unknown",
            )
            logger.warning(
                "Unauthorized request from %s to %s",
                request.client.host if request.client else "?",
                request.url.path,
            )
            response = Response(
                content='{"error": "Unauthorized. Provide X-API-Key header."}',
                status_code=401,
                media_type="application/json",
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


# ── H2: Prompt injection guard ────────────────────────────────────────
def sanitize_user_input(text: str) -> str:
    """
    Guard against prompt injection and enforce length cap.
    Raises ValueError (caught by executor) if injection detected.
    Captured to Sentry as a warning-level security event.
    """
    text = text.strip()

    if len(text) > MAX_INPUT_LENGTH:
        logger.warning("Input truncated: %d → %d chars", len(text), MAX_INPUT_LENGTH)
        text = text[:MAX_INPUT_LENGTH]

    for pattern in _INJECTION_RE:
        if pattern.search(text):
            _capture_security_event(
                "[RAPHI] Prompt injection attempt detected",
                level="warning",
                pattern=pattern.pattern,
                preview=text[:200],
            )
            logger.warning("Prompt injection rejected — pattern: %s", pattern.pattern)
            raise ValueError(
                "Input rejected: contains instructions that attempt to override AI behaviour. "
                "Please ask your investment question directly."
            )

    return text


# ── Sentry integration ────────────────────────────────────────────────
def init_sentry() -> None:
    """
    Initialize Sentry SDK.
    Set SENTRY_DSN env var from https://raphi.sentry.io → Settings → Projects → DSN
    """
    dsn = os.environ.get("SENTRY_DSN", "")
    if not dsn:
        logger.info("SENTRY_DSN not configured — Sentry monitoring disabled")
        return

    try:
        import sentry_sdk
        from sentry_sdk.integrations.asyncio import AsyncioIntegration
        from sentry_sdk.integrations.logging import LoggingIntegration

        sentry_sdk.init(
            dsn=dsn,
            environment=os.environ.get("RAPHI_ENV", "development"),
            release=os.environ.get("RAPHI_VERSION", "1.0.0"),
            traces_sample_rate=0.0,
            profiles_sample_rate=0.0,
            # Disable auto-detection of FastAPI/Starlette — incompatible with FastAPI 0.115+
            auto_enabling_integrations=False,
            integrations=[
                LoggingIntegration(),
                AsyncioIntegration(),
            ],
            before_send=_scrub_sensitive_data,
            send_default_pii=False,
        )
        logger.info("Sentry initialized → raphi.sentry.io")
    except ImportError:
        logger.warning("sentry-sdk not installed — run: pip install sentry-sdk[starlette]")


def _capture_security_event(message: str, level: str = "warning", **extras) -> None:
    """Send a security event to Sentry if SDK is initialised, else just log."""
    try:
        import sentry_sdk
        with sentry_sdk.new_scope() as scope:
            for k, v in extras.items():
                scope.set_extra(k, str(v)[:500])  # cap extra value size
            sentry_sdk.capture_message(message, level=level, scope=scope)
    except Exception:
        pass  # Never let Sentry errors crash the app


def _scrub_sensitive_data(event, hint):
    """
    Sentry before_send hook — redact API keys and portfolio dollar amounts
    before any event leaves this machine.
    """
    for section in ("extra", "contexts", "tags"):
        block = event.get(section, {})
        if isinstance(block, dict):
            for key in list(block.keys()):
                val = str(block.get(key, ""))
                if re.search(r"sk-ant", val, re.IGNORECASE):
                    block[key] = "[API_KEY_REDACTED]"
                elif re.search(r"\$[\d,]{4,}", val):
                    block[key] = "[FINANCIAL_VALUE_REDACTED]"
    return event


# ── H3: Fernet encryption for sessions.json ──────────────────────────
class SessionCipher:
    """
    Symmetric encryption for the sessions.json store.

    Key priority:
      1. RAPHI_SESSION_KEY env var (base64url 32-byte key from `Fernet.generate_key()`)
      2. Machine-derived key (SHA-256 of hostname — stable but not secret)
      3. Plaintext fallback if `cryptography` not installed
    """

    def __init__(self) -> None:
        self._available = False
        try:
            import base64
            import socket

            from cryptography.fernet import Fernet

            raw = os.environ.get("RAPHI_SESSION_KEY", "")
            if raw:
                key = raw.encode()
            else:
                seed = hashlib.sha256(socket.gethostname().encode()).digest()
                key = base64.urlsafe_b64encode(seed)
                logger.info(
                    "SessionCipher: using machine-derived key. "
                    "Set RAPHI_SESSION_KEY for a fixed secret key."
                )

            self._fernet = Fernet(key)
            self._available = True
        except ImportError:
            logger.warning("cryptography not installed — sessions.json stored as plaintext")

    def encrypt(self, data: str) -> str:
        if not self._available:
            return data
        return self._fernet.encrypt(data.encode()).decode()

    def decrypt(self, data: str) -> str:
        if not self._available:
            return data
        try:
            return self._fernet.decrypt(data.encode()).decode()
        except Exception:
            return data  # Migration: treat as plaintext on first run
