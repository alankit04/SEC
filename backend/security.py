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

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import time
from typing import Any

from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("raphi.security")

try:
    import jwt
except Exception:  # pragma: no cover
    jwt = None


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}

# ── Public paths that skip auth ──────────────────────────────────────
PUBLIC_PATHS = {
    "/.well-known/agent.json",        # A2A agent card (legacy path)
    "/.well-known/agent-card.json",   # A2A agent card (current A2A spec)
    "/health",
    "/api/health",                    # unified server health probe
    "/api/auth/login",               # email-first browser session bootstrap
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


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64url_decode(raw: str) -> bytes:
    padded = raw + "=" * ((4 - len(raw) % 4) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def issue_browser_session_token(
    *,
    user_id: str,
    tenant: str = "local",
    role: str = "analyst",
    secret: str,
    ttl_seconds: int = 60 * 60 * 12,
) -> str:
    now = int(time.time())
    payload = {
        "sub": str(user_id).strip(),
        "tenant": str(tenant).strip().lower() or "local",
        "role": str(role).strip().lower() or "analyst",
        "iat": now,
        "exp": now + max(300, int(ttl_seconds)),
        "typ": "raphi-session",
    }
    body = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    sig = _b64url_encode(hmac.new(secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest())
    return f"raphi.{body}.{sig}"


def decode_browser_session_token(token: str, secret: str) -> dict[str, Any]:
    parts = str(token or "").split(".")
    if len(parts) != 3 or parts[0] != "raphi":
        raise ValueError("Invalid session token format")

    body, sig = parts[1], parts[2]
    expected = _b64url_encode(hmac.new(secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest())
    if not hmac.compare_digest(sig, expected):
        raise ValueError("Invalid session signature")

    payload = json.loads(_b64url_decode(body).decode("utf-8"))
    exp = int(payload.get("exp") or 0)
    if exp and int(time.time()) > exp:
        raise ValueError("Session expired")
    return payload


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
        self._allow_local_api_bypass = _env_bool("RAPHI_ALLOW_LOCAL_API_BYPASS", False)
        self._allow_no_api_key = _env_bool("RAPHI_ALLOW_NO_API_KEY", False)
        self._jwt_secret = os.environ.get("RAPHI_JWT_SECRET", "").strip()
        self._jwt_alg = os.environ.get("RAPHI_JWT_ALGORITHM", "HS256").strip() or "HS256"
        self._jwt_aud = os.environ.get("RAPHI_JWT_AUDIENCE", "").strip()
        self._jwt_iss = os.environ.get("RAPHI_JWT_ISSUER", "").strip()
        self._require_jwt = _env_bool("RAPHI_REQUIRE_JWT", False)
        self._session_secret = os.environ.get("RAPHI_SESSION_SECRET", "").strip() or api_key

    def _decode_jwt(self, token: str) -> dict[str, Any]:
        if jwt is None:
            raise ValueError("pyjwt is not installed")
        if not self._jwt_secret:
            raise ValueError("RAPHI_JWT_SECRET is not configured")
        options: dict[str, bool] = {"verify_aud": bool(self._jwt_aud), "verify_iss": bool(self._jwt_iss)}
        kwargs: dict[str, Any] = {"algorithms": [self._jwt_alg], "options": options}
        if self._jwt_aud:
            kwargs["audience"] = self._jwt_aud
        if self._jwt_iss:
            kwargs["issuer"] = self._jwt_iss
        return jwt.decode(token, self._jwt_secret, **kwargs)

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

        # Optional local UI mode: allow same-machine API calls without requiring X-API-Key.
        # Disabled by default for secure-by-default deployments.
        client_host = request.client.host if request.client else ""
        if (
            self._allow_local_api_bypass
            and path.startswith("/api/")
            and client_host in {"127.0.0.1", "::1", "localhost"}
        ):
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

        bearer = request.headers.get("Authorization", "")
        bearer_token = bearer.removeprefix("Bearer ").strip() if bearer.startswith("Bearer ") else ""
        if bearer_token and self._session_secret:
            try:
                claims = decode_browser_session_token(bearer_token, self._session_secret)
                scope["raphi_auth"] = {
                    "auth_type": "session",
                    "sub": str(claims.get("sub") or "").strip(),
                    "tenant": str(claims.get("tenant") or "local").strip().lower(),
                    "role": str(claims.get("role") or "analyst").strip().lower(),
                    "claims": claims,
                }
                await self.app(scope, receive, send)
                return
            except Exception:
                pass

        if bearer_token and "." in bearer_token and bearer_token.count(".") == 2:
            try:
                claims = self._decode_jwt(bearer_token)
                scope["raphi_auth"] = {
                    "auth_type": "jwt",
                    "sub": str(claims.get("sub") or "").strip(),
                    "tenant": str(claims.get("tenant") or claims.get("tid") or "local").strip().lower(),
                    "role": str(claims.get("role") or claims.get("raphi_role") or "analyst").strip().lower(),
                    "claims": claims,
                }
                await self.app(scope, receive, send)
                return
            except Exception as exc:
                if self._require_jwt:
                    response = Response(
                        content=f'{{"error": "Unauthorized JWT: {str(exc)[:180]}"}}',
                        status_code=401,
                        media_type="application/json",
                    )
                    await response(scope, receive, send)
                    return

        if not self._api_key:
            if self._allow_no_api_key:
                logger.warning("RAPHI_API_KEY not set — allowing request because RAPHI_ALLOW_NO_API_KEY=1")
                await self.app(scope, receive, send)
                return
            response = Response(
                content='{"error": "Unauthorized. RAPHI_API_KEY is required."}',
                status_code=401,
                media_type="application/json",
            )
            await response(scope, receive, send)
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

        scope["raphi_auth"] = {"auth_type": "api_key", "role": "analyst", "tenant": "local", "sub": "api-key-user"}

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


# ── Sentry integration (disabled — no SENTRY_DSN configured) ─────────
def init_sentry() -> None:
    """No-op. Sentry is not used; kept so call sites in raphi_server/a2a_server don't break."""


def _capture_security_event(message: str, level: str = "warning", **extras) -> None:
    """Log security events locally only."""
    extra_str = " ".join(f"{k}={str(v)[:200]}" for k, v in extras.items())
    getattr(logger, level if level in ("debug", "info", "warning", "error") else "warning")(
        "SECURITY: %s %s", message, extra_str
    )


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
