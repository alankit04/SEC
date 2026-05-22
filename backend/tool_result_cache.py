"""tool_result_cache.py — versioned async tool-result cache for RAPHI.

Features:
- Stable cache keys: tool_name + normalized_args_hash + data_version + model_version + scope
- TTL freshness with optional stale-grace window
- Per-key single-flight to prevent cache stampedes under concurrency
- Rich metadata on each cached object for trust/audit visibility
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable


def _now_ts() -> float:
    return time.time()


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _stable_args_hash(arguments: dict[str, Any]) -> str:
    try:
        payload = json.dumps(arguments or {}, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        payload = str(arguments)
    return hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()[:16]


def build_cache_key(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    data_version: str,
    model_version: str,
    user_scope: str,
) -> str:
    args_hash = _stable_args_hash(arguments)
    return (
        f"{tool_name}:{args_hash}:"
        f"{data_version or 'na'}:{model_version or 'na'}:{user_scope or 'global'}"
    )


@dataclass
class CacheRecord:
    value: Any
    created_ts: float
    expires_ts: float
    stale_until_ts: float
    source: str
    data_version: str
    model_version: str
    user_scope: str
    latency_ms: int

    def freshness_state(self, now_ts: float | None = None) -> str:
        now = now_ts if now_ts is not None else _now_ts()
        if now <= self.expires_ts:
            return "fresh"
        if now <= self.stale_until_ts:
            return "stale"
        return "expired"

    def metadata(self, *, cache_key: str, cache_hit: bool) -> dict[str, Any]:
        now = _now_ts()
        return {
            "cache_key": cache_key,
            "cache_hit": bool(cache_hit),
            "created_at": _iso(self.created_ts),
            "expires_at": _iso(self.expires_ts),
            "source": self.source,
            "freshness_state": self.freshness_state(now),
            "data_version": self.data_version,
            "model_version": self.model_version,
            "user_scope": self.user_scope,
            "latency_ms": self.latency_ms,
            "age_ms": int((now - self.created_ts) * 1000),
        }


class ToolResultCache:
    """Async in-memory tool-result cache with single-flight deduping."""

    def __init__(self, *, default_ttl_s: int = 120, default_stale_grace_s: int = 0) -> None:
        self.default_ttl_s = max(1, int(default_ttl_s))
        self.default_stale_grace_s = max(0, int(default_stale_grace_s))
        self._records: dict[str, CacheRecord] = {}
        self._inflight: dict[str, asyncio.Future] = {}
        self._state_lock = asyncio.Lock()

    async def invalidate_tool(self, tool_name: str) -> int:
        prefix = f"{tool_name}:"
        async with self._state_lock:
            keys = [k for k in self._records if k.startswith(prefix)]
            for key in keys:
                self._records.pop(key, None)
            return len(keys)

    async def invalidate_all(self) -> int:
        async with self._state_lock:
            count = len(self._records)
            self._records.clear()
            return count

    async def get_or_compute(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        source: str,
        producer: Callable[[], Awaitable[Any]],
        ttl_s: int | None = None,
        stale_grace_s: int | None = None,
        data_version: str = "",
        model_version: str = "",
        user_scope: str = "global",
        serve_stale_on_error: bool = True,
    ) -> tuple[Any, dict[str, Any]]:
        ttl = max(1, int(ttl_s if ttl_s is not None else self.default_ttl_s))
        stale_grace = max(0, int(stale_grace_s if stale_grace_s is not None else self.default_stale_grace_s))
        key = build_cache_key(
            tool_name=tool_name,
            arguments=arguments,
            data_version=data_version,
            model_version=model_version,
            user_scope=user_scope,
        )

        stale_candidate: CacheRecord | None = None

        while True:
            now = _now_ts()
            async with self._state_lock:
                record = self._records.get(key)
                if record is not None:
                    state = record.freshness_state(now)
                    if state == "fresh":
                        return record.value, record.metadata(cache_key=key, cache_hit=True)
                    if state == "stale":
                        stale_candidate = record
                    else:
                        self._records.pop(key, None)

                inflight = self._inflight.get(key)
                if inflight is None:
                    inflight = asyncio.get_running_loop().create_future()
                    self._inflight[key] = inflight
                    leader = True
                else:
                    leader = False

            if leader:
                break

            try:
                value, meta = await inflight
                return value, dict(meta)
            except Exception:
                # If the leader failed, retry and attempt to become the next leader.
                continue

        started = time.perf_counter()
        try:
            value = await producer()
            created_ts = _now_ts()
            latency_ms = int((time.perf_counter() - started) * 1000)
            record = CacheRecord(
                value=value,
                created_ts=created_ts,
                expires_ts=created_ts + ttl,
                stale_until_ts=created_ts + ttl + stale_grace,
                source=source,
                data_version=data_version,
                model_version=model_version,
                user_scope=user_scope,
                latency_ms=latency_ms,
            )
            meta = record.metadata(cache_key=key, cache_hit=False)
            async with self._state_lock:
                self._records[key] = record
                waiter = self._inflight.pop(key, None)
                if waiter is not None and not waiter.done():
                    waiter.set_result((value, meta))
            return value, meta
        except Exception as exc:
            if stale_candidate is not None and serve_stale_on_error:
                stale_meta = stale_candidate.metadata(cache_key=key, cache_hit=True)
                stale_meta["served_stale_after_error"] = True
                stale_meta["refresh_error"] = str(exc)
                async with self._state_lock:
                    waiter = self._inflight.pop(key, None)
                    if waiter is not None and not waiter.done():
                        waiter.set_result((stale_candidate.value, stale_meta))
                return stale_candidate.value, stale_meta

            async with self._state_lock:
                waiter = self._inflight.pop(key, None)
                if waiter is not None and not waiter.done():
                    waiter.set_exception(exc)
            raise
