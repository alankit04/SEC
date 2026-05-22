import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from tool_result_cache import ToolResultCache, build_cache_key


def test_build_cache_key_is_stable_for_argument_order():
    key_a = build_cache_key(
        tool_name="sec_filings",
        arguments={"ticker": "NVDA", "limit": 5},
        data_version="sec-v1",
        model_version="",
        user_scope="global",
    )
    key_b = build_cache_key(
        tool_name="sec_filings",
        arguments={"limit": 5, "ticker": "NVDA"},
        data_version="sec-v1",
        model_version="",
        user_scope="global",
    )
    key_c = build_cache_key(
        tool_name="sec_filings",
        arguments={"limit": 5, "ticker": "NVDA"},
        data_version="sec-v1",
        model_version="",
        user_scope="local-user",
    )

    assert key_a == key_b
    assert key_a != key_c


def test_cache_hit_after_first_compute():
    cache = ToolResultCache(default_ttl_s=60)
    calls = {"count": 0}

    async def producer():
        calls["count"] += 1
        return {"ok": True}

    async def run_case():
        first_value, first_meta = await cache.get_or_compute(
            tool_name="market_overview",
            arguments={},
            source="test",
            producer=producer,
            data_version="v1",
            user_scope="global",
        )
        second_value, second_meta = await cache.get_or_compute(
            tool_name="market_overview",
            arguments={},
            source="test",
            producer=producer,
            data_version="v1",
            user_scope="global",
        )
        return first_value, first_meta, second_value, second_meta

    first_value, first_meta, second_value, second_meta = asyncio.run(run_case())

    assert calls["count"] == 1
    assert first_value == second_value == {"ok": True}
    assert first_meta["cache_hit"] is False
    assert second_meta["cache_hit"] is True


def test_single_flight_dedupes_concurrent_calls():
    cache = ToolResultCache(default_ttl_s=60)
    calls = {"count": 0}

    async def producer():
        calls["count"] += 1
        await asyncio.sleep(0.05)
        return {"ticker": "NVDA"}

    async def run_case():
        async def one_call():
            return await cache.get_or_compute(
                tool_name="stock_detail",
                arguments={"ticker": "NVDA"},
                source="test",
                producer=producer,
                data_version="market-v1",
                user_scope="global",
            )

        results = await asyncio.gather(*[one_call() for _ in range(5)])
        return results

    results = asyncio.run(run_case())

    assert calls["count"] == 1
    assert len(results) == 5
    assert all(item[0] == {"ticker": "NVDA"} for item in results)


def test_stale_value_served_when_refresh_fails():
    cache = ToolResultCache(default_ttl_s=1, default_stale_grace_s=5)

    async def ok_producer():
        return {"value": 1}

    async def failing_producer():
        raise RuntimeError("upstream failed")

    async def seed():
        await cache.get_or_compute(
            tool_name="web_citations",
            arguments={"query": "NVDA"},
            source="test",
            producer=ok_producer,
            data_version="citation-v1",
            user_scope="global",
        )

    asyncio.run(seed())
    time.sleep(1.2)

    async def refresh_with_error():
        return await cache.get_or_compute(
            tool_name="web_citations",
            arguments={"query": "NVDA"},
            source="test",
            producer=failing_producer,
            data_version="citation-v1",
            user_scope="global",
        )

    value, meta = asyncio.run(refresh_with_error())

    assert value == {"value": 1}
    assert meta["freshness_state"] == "stale"
    assert meta.get("served_stale_after_error") is True
