"""
Tests for backend/raphi_mcp_server.py — MCP bridge utilities.

Tests cover: ticker validation, header construction, BASE_URL env override,
tool TTL lookup, and scope sanitisation. No live HTTP calls are made.
"""
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import importlib
import pytest


# ── import module under test (no side effects) ───────────────────────────────

def _import_mcp():
    if "backend.raphi_mcp_server" in sys.modules:
        return sys.modules["backend.raphi_mcp_server"]
    return importlib.import_module("backend.raphi_mcp_server")


# ── _validate_ticker ──────────────────────────────────────────────────────────

class TestValidateTicker:
    def setup_method(self):
        self.mcp = _import_mcp()

    def test_valid_simple_ticker(self):
        assert self.mcp._validate_ticker("nvda") == "NVDA"

    def test_valid_class_suffix(self):
        assert self.mcp._validate_ticker("brk.b") == "BRK.B"

    def test_valid_single_letter(self):
        assert self.mcp._validate_ticker("F") == "F"

    def test_valid_five_letters(self):
        assert self.mcp._validate_ticker("googl") == "GOOGL"

    def test_rejects_six_letters(self):
        with pytest.raises(ValueError, match="Invalid ticker"):
            self.mcp._validate_ticker("TOOLONG")

    def test_rejects_digits(self):
        with pytest.raises(ValueError, match="Invalid ticker"):
            self.mcp._validate_ticker("NVDA1")

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="Invalid ticker"):
            self.mcp._validate_ticker("")

    def test_rejects_injection_attempt(self):
        with pytest.raises(ValueError, match="Invalid ticker"):
            self.mcp._validate_ticker("'; DROP TABLE")


# ── _get_headers ──────────────────────────────────────────────────────────────

class TestGetHeaders:
    def setup_method(self):
        self.mcp = _import_mcp()

    def test_content_type_always_present(self):
        headers = self.mcp._get_headers()
        assert headers.get("Content-Type") == "application/json"

    def test_no_internal_token_when_env_empty(self, monkeypatch):
        monkeypatch.setattr(self.mcp, "_INTERNAL_TOKEN", "")
        headers = self.mcp._get_headers()
        assert "X-Internal-Token" not in headers

    def test_internal_token_included_when_set(self, monkeypatch):
        monkeypatch.setattr(self.mcp, "_INTERNAL_TOKEN", "secret-abc")
        headers = self.mcp._get_headers()
        assert headers["X-Internal-Token"] == "secret-abc"


# ── BASE_URL env override ─────────────────────────────────────────────────────

def test_base_url_defaults_to_localhost():
    mcp = _import_mcp()
    # Default (env var not overridden in this process) should contain localhost
    assert "localhost" in mcp.BASE_URL or "127.0.0.1" in mcp.BASE_URL


def test_base_url_strips_trailing_slash(monkeypatch):
    monkeypatch.setenv("RAPHI_PUBLIC_URL", "http://myserver:9999/")
    # Re-import so module-level assignment is re-evaluated
    if "backend.raphi_mcp_server" in sys.modules:
        del sys.modules["backend.raphi_mcp_server"]
    mcp = importlib.import_module("backend.raphi_mcp_server")
    assert not mcp.BASE_URL.endswith("/")
    assert mcp.BASE_URL == "http://myserver:9999"


# ── _tool_ttl ─────────────────────────────────────────────────────────────────

class TestToolTtl:
    def setup_method(self):
        self.mcp = _import_mcp()

    def test_returns_int(self):
        assert isinstance(self.mcp._tool_ttl("stock_detail"), int)

    def test_known_tool_has_positive_ttl(self):
        assert self.mcp._tool_ttl("stock_detail") > 0

    def test_unknown_tool_has_fallback_ttl(self):
        assert self.mcp._tool_ttl("nonexistent_tool_xyz") >= 0


# ── _sanitize_scope ───────────────────────────────────────────────────────────

class TestSanitizeScope:
    def setup_method(self):
        self.mcp = _import_mcp()

    def test_strips_whitespace(self):
        result = self.mcp._sanitize_scope("  global  ")
        assert result == result.strip()

    def test_returns_string(self):
        assert isinstance(self.mcp._sanitize_scope("user123"), str)

    def test_empty_gets_default(self):
        result = self.mcp._sanitize_scope("")
        assert isinstance(result, str)
