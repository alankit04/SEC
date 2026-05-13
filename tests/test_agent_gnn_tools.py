import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import a2a_executor_v2
import raphi_mcp_server


def test_a2a_allows_gnn_and_memory_mcp_tools():
    allowed = set(a2a_executor_v2.ALLOWED_TOOLS)

    assert "mcp__raphi__sec_universe" in allowed
    assert "mcp__raphi__sec_industries" in allowed
    assert "mcp__raphi__gnn_signal" in allowed
    assert "mcp__raphi__gnn_status" in allowed
    assert "mcp__raphi__gnn_train" in allowed
    assert "mcp__raphi__memory_status" in allowed
    assert "mcp__raphi__memory_retrieve" in allowed
    assert "GraphSAGE" in a2a_executor_v2.SYSTEM_PROMPT


def test_mcp_lists_real_gnn_tools():
    tools = asyncio.run(raphi_mcp_server.list_tools())
    names = {tool.name for tool in tools}

    assert {"gnn_signal", "gnn_status", "gnn_train"}.issubset(names)
    assert {"sec_universe", "sec_industries"}.issubset(names)
    gnn_signal = next(tool for tool in tools if tool.name == "gnn_signal")
    assert gnn_signal.inputSchema["properties"]["ticker"]["pattern"] == "^[A-Z]{1,5}$"
