"""Tests for AgentMemory and memory tool integration."""

from __future__ import annotations

import json

from sec_agent.memory.agent_memory import AgentMemory, MemoryEntry
from sec_agent.agent import _execute_tool


# ---------------------------------------------------------------------------
# MemoryEntry tests
# ---------------------------------------------------------------------------


class TestMemoryEntry:
    def test_to_dict_roundtrip(self):
        entry = MemoryEntry(
            ticker="AAPL",
            filing_type="10-K",
            summary="Risk: supply chain",
            sections_analyzed=["1A", "7"],
            timestamp="2026-01-27T00:00:00+00:00",
        )
        d = entry.to_dict()
        restored = MemoryEntry.from_dict(d)
        assert restored.ticker == "AAPL"
        assert restored.filing_type == "10-K"
        assert restored.summary == "Risk: supply chain"
        assert restored.sections_analyzed == ["1A", "7"]
        assert restored.timestamp == "2026-01-27T00:00:00+00:00"

    def test_from_dict_default_timestamp(self):
        entry = MemoryEntry.from_dict(
            {
                "ticker": "MSFT",
                "filing_type": "10-Q",
                "summary": "Growth",
                "sections_analyzed": ["2"],
            }
        )
        assert entry.ticker == "MSFT"
        assert entry.timestamp  # should be auto-populated


# ---------------------------------------------------------------------------
# AgentMemory tests
# ---------------------------------------------------------------------------


class TestAgentMemory:
    def test_session_id_auto_generated(self):
        mem = AgentMemory()
        assert mem.session_id
        assert isinstance(mem.session_id, str)

    def test_session_id_custom(self):
        mem = AgentMemory(session_id="test-session")
        assert mem.session_id == "test-session"

    def test_store_and_retrieve(self):
        mem = AgentMemory()
        mem.store_analysis("AAPL", "10-K", "Risks noted", ["1A"])
        analyses = mem.get_recent_analyses()
        assert len(analyses) == 1
        assert analyses[0]["ticker"] == "AAPL"

    def test_filter_by_ticker(self):
        mem = AgentMemory()
        mem.store_analysis("AAPL", "10-K", "Apple risks", ["1A"])
        mem.store_analysis("MSFT", "10-Q", "MSFT growth", ["2"])
        assert len(mem.get_recent_analyses(ticker="AAPL")) == 1
        assert len(mem.get_recent_analyses(ticker="msft")) == 1  # case-insensitive
        assert len(mem.get_recent_analyses()) == 2

    def test_get_session_context(self):
        mem = AgentMemory(session_id="s1")
        mem.store_analysis("TSLA", "8-K", "Event", ["1.01"])
        ctx = mem.get_session_context()
        assert ctx["session_id"] == "s1"
        assert len(ctx["entries"]) == 1

    def test_clear(self):
        mem = AgentMemory()
        mem.store_analysis("AAPL", "10-K", "Risks", ["1A"])
        mem.clear()
        assert mem.get_recent_analyses() == []

    def test_empty_memory(self):
        mem = AgentMemory()
        assert mem.get_recent_analyses() == []
        ctx = mem.get_session_context()
        assert ctx["entries"] == []


# ---------------------------------------------------------------------------
# Memory tool dispatch tests (via _execute_tool)
# ---------------------------------------------------------------------------


class TestMemoryToolDispatch:
    def test_save_memory(self):
        mem = AgentMemory()
        result = json.loads(
            _execute_tool(
                "save_memory",
                {
                    "ticker": "AAPL",
                    "filing_type": "10-K",
                    "summary": "Key risks identified",
                    "sections_analyzed": ["1A"],
                },
                memory=mem,
            )
        )
        assert result["status"] == "ok"
        assert len(mem.get_recent_analyses()) == 1

    def test_get_memory_all(self):
        mem = AgentMemory(session_id="s1")
        mem.store_analysis("AAPL", "10-K", "Risks", ["1A"])
        result = json.loads(_execute_tool("get_memory", {}, memory=mem))
        assert result["status"] == "ok"
        assert result["session_id"] == "s1"
        assert len(result["entries"]) == 1

    def test_get_memory_by_ticker(self):
        mem = AgentMemory()
        mem.store_analysis("AAPL", "10-K", "Risks", ["1A"])
        mem.store_analysis("MSFT", "10-Q", "Growth", ["2"])
        result = json.loads(
            _execute_tool("get_memory", {"ticker": "AAPL"}, memory=mem)
        )
        assert result["status"] == "ok"
        assert len(result["analyses"]) == 1

    def test_memory_tool_without_memory(self):
        result = json.loads(_execute_tool("get_memory", {}, memory=None))
        assert result["status"] == "error"

    def test_non_memory_tool_still_works(self):
        # list_available_filings with a ticker that has no cache should still work
        # (it will try S3 and fail, but the dispatch path should work)
        # We just verify it doesn't crash on the dispatch logic
        result = json.loads(
            _execute_tool("unknown_tool_xyz", {}, memory=AgentMemory())
        )
        assert result["status"] == "error"
        assert "Unknown tool" in result["message"]
