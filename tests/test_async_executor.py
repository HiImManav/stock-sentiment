"""Tests for the AsyncAgentExecutor."""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import Mock

import pytest

from orchestrator.execution.async_executor import (
    AsyncAgentExecutor,
    ExecutionResult,
)
from orchestrator.execution.result import AgentResult, AgentStatus


class MockAgent:
    """Mock agent for testing."""

    def __init__(
        self,
        response: str = "Mock response",
        delay: float = 0.0,
        raise_exception: Exception | None = None,
    ) -> None:
        self.response = response
        self.delay = delay
        self.raise_exception = raise_exception
        self.query_count = 0
        self.last_query: str | None = None

    def query(self, user_message: str) -> str:
        self.query_count += 1
        self.last_query = user_message

        if self.delay > 0:
            time.sleep(self.delay)

        if self.raise_exception is not None:
            raise self.raise_exception

        return self.response


class TestAsyncAgentExecutor:
    """Tests for AsyncAgentExecutor."""

    # -------------------------------------------------------------------------
    # Initialization tests
    # -------------------------------------------------------------------------

    def test_init_with_no_agents(self) -> None:
        """Executor can be initialized without agents."""
        executor = AsyncAgentExecutor()
        assert executor.news_agent is None
        assert executor.sec_agent is None
        assert executor.timeout_seconds == 60.0

    def test_init_with_agents(self) -> None:
        """Executor can be initialized with agent instances."""
        news_agent = MockAgent()
        sec_agent = MockAgent()

        executor = AsyncAgentExecutor(
            news_agent=news_agent,
            sec_agent=sec_agent,
            timeout_seconds=30.0,
        )

        assert executor.news_agent is news_agent
        assert executor.sec_agent is sec_agent
        assert executor.timeout_seconds == 30.0

    # -------------------------------------------------------------------------
    # Single agent execution tests
    # -------------------------------------------------------------------------

    def test_execute_news_success(self) -> None:
        """Execute news agent successfully."""

        async def run_test() -> None:
            news_agent = MockAgent(response="News analysis result")
            executor = AsyncAgentExecutor(news_agent=news_agent)

            result = await executor.execute_news("What's happening with AAPL?")

            assert result.agent_name == "news_agent"
            assert result.status == AgentStatus.SUCCESS
            assert result.response == "News analysis result"
            assert result.error_message is None
            assert result.execution_time_ms > 0
            assert result.is_success is True
            assert news_agent.last_query == "What's happening with AAPL?"

        asyncio.run(run_test())

    def test_execute_sec_success(self) -> None:
        """Execute SEC agent successfully."""

        async def run_test() -> None:
            sec_agent = MockAgent(response="SEC filing analysis")
            executor = AsyncAgentExecutor(sec_agent=sec_agent)

            result = await executor.execute_sec("What are the risk factors?")

            assert result.agent_name == "sec_agent"
            assert result.status == AgentStatus.SUCCESS
            assert result.response == "SEC filing analysis"
            assert result.is_success is True
            assert sec_agent.last_query == "What are the risk factors?"

        asyncio.run(run_test())

    def test_execute_news_not_configured(self) -> None:
        """Executing news agent when not configured raises ValueError."""

        async def run_test() -> None:
            executor = AsyncAgentExecutor()

            with pytest.raises(ValueError, match="News agent is not configured"):
                await executor.execute_news("test query")

        asyncio.run(run_test())

    def test_execute_sec_not_configured(self) -> None:
        """Executing SEC agent when not configured raises ValueError."""

        async def run_test() -> None:
            executor = AsyncAgentExecutor()

            with pytest.raises(ValueError, match="SEC agent is not configured"):
                await executor.execute_sec("test query")

        asyncio.run(run_test())

    # -------------------------------------------------------------------------
    # Timeout handling tests
    # -------------------------------------------------------------------------

    def test_timeout_handling(self) -> None:
        """Agent that exceeds timeout should return timeout status."""

        async def run_test() -> None:
            slow_agent = MockAgent(delay=0.5)
            executor = AsyncAgentExecutor(news_agent=slow_agent, timeout_seconds=0.1)

            result = await executor.execute_news("test query")

            assert result.status == AgentStatus.TIMEOUT
            assert result.is_timeout is True
            assert result.response is None
            assert result.error_message is not None
            assert "timed out" in result.error_message.lower()

        asyncio.run(run_test())

    def test_execution_within_timeout(self) -> None:
        """Agent that completes within timeout should succeed."""

        async def run_test() -> None:
            fast_agent = MockAgent(delay=0.01)
            executor = AsyncAgentExecutor(news_agent=fast_agent, timeout_seconds=1.0)

            result = await executor.execute_news("test query")

            assert result.status == AgentStatus.SUCCESS
            assert result.is_success is True

        asyncio.run(run_test())

    # -------------------------------------------------------------------------
    # Error handling tests
    # -------------------------------------------------------------------------

    def test_error_handling(self) -> None:
        """Agent that raises exception should return error status."""

        async def run_test() -> None:
            error_agent = MockAgent(raise_exception=RuntimeError("Connection failed"))
            executor = AsyncAgentExecutor(sec_agent=error_agent)

            result = await executor.execute_sec("test query")

            assert result.status == AgentStatus.ERROR
            assert result.is_error is True
            assert result.response is None
            assert result.error_message == "Connection failed"

        asyncio.run(run_test())

    def test_error_with_execution_time(self) -> None:
        """Error result should still include execution time."""

        async def run_test() -> None:
            error_agent = MockAgent(
                delay=0.05,
                raise_exception=ValueError("Bad input"),
            )
            executor = AsyncAgentExecutor(news_agent=error_agent)

            result = await executor.execute_news("test")

            assert result.status == AgentStatus.ERROR
            assert result.execution_time_ms >= 50  # At least 50ms delay

        asyncio.run(run_test())

    # -------------------------------------------------------------------------
    # Parallel execution tests
    # -------------------------------------------------------------------------

    def test_execute_both_parallel(self) -> None:
        """Execute both agents in parallel."""

        async def run_test() -> None:
            news_agent = MockAgent(response="News result", delay=0.05)
            sec_agent = MockAgent(response="SEC result", delay=0.05)
            executor = AsyncAgentExecutor(news_agent=news_agent, sec_agent=sec_agent)

            start = time.perf_counter()
            result = await executor.execute_both("test query")
            elapsed = time.perf_counter() - start

            # Should complete in roughly 0.05s (parallel), not 0.1s (sequential)
            assert elapsed < 0.15  # Allow some overhead

            assert result.news_result is not None
            assert result.sec_result is not None
            assert result.news_result.status == AgentStatus.SUCCESS
            assert result.sec_result.status == AgentStatus.SUCCESS
            assert result.news_result.response == "News result"
            assert result.sec_result.response == "SEC result"

        asyncio.run(run_test())

    def test_execute_both_with_only_news(self) -> None:
        """Execute both with only news agent configured."""

        async def run_test() -> None:
            news_agent = MockAgent(response="News only")
            executor = AsyncAgentExecutor(news_agent=news_agent)

            result = await executor.execute_both("test query")

            assert result.news_result is not None
            assert result.sec_result is None
            assert result.has_any_success is True

        asyncio.run(run_test())

    def test_execute_both_with_only_sec(self) -> None:
        """Execute both with only SEC agent configured."""

        async def run_test() -> None:
            sec_agent = MockAgent(response="SEC only")
            executor = AsyncAgentExecutor(sec_agent=sec_agent)

            result = await executor.execute_both("test query")

            assert result.news_result is None
            assert result.sec_result is not None
            assert result.has_any_success is True

        asyncio.run(run_test())

    def test_execute_both_no_agents_raises(self) -> None:
        """Execute both with no agents should raise ValueError."""

        async def run_test() -> None:
            executor = AsyncAgentExecutor()

            with pytest.raises(ValueError, match="At least one agent must be configured"):
                await executor.execute_both("test query")

        asyncio.run(run_test())

    def test_execute_both_one_fails(self) -> None:
        """When one agent fails, the other should still succeed."""

        async def run_test() -> None:
            news_agent = MockAgent(response="News works")
            sec_agent = MockAgent(raise_exception=RuntimeError("SEC failed"))
            executor = AsyncAgentExecutor(news_agent=news_agent, sec_agent=sec_agent)

            result = await executor.execute_both("test query")

            assert result.news_result is not None
            assert result.news_result.status == AgentStatus.SUCCESS
            assert result.sec_result is not None
            assert result.sec_result.status == AgentStatus.ERROR
            assert result.has_any_success is True
            assert result.all_succeeded is False

        asyncio.run(run_test())

    def test_execute_both_one_times_out(self) -> None:
        """When one agent times out, the other should still succeed."""

        async def run_test() -> None:
            news_agent = MockAgent(response="Fast news", delay=0.01)
            sec_agent = MockAgent(response="Slow SEC", delay=0.5)
            executor = AsyncAgentExecutor(
                news_agent=news_agent,
                sec_agent=sec_agent,
                timeout_seconds=0.1,
            )

            result = await executor.execute_both("test query")

            assert result.news_result is not None
            assert result.news_result.status == AgentStatus.SUCCESS
            assert result.sec_result is not None
            assert result.sec_result.status == AgentStatus.TIMEOUT
            assert result.has_any_success is True
            assert result.all_succeeded is False

        asyncio.run(run_test())

    # -------------------------------------------------------------------------
    # execute() method tests
    # -------------------------------------------------------------------------

    def test_execute_with_news_only(self) -> None:
        """Execute with only news agent flag."""

        async def run_test() -> None:
            news_agent = MockAgent(response="News result")
            executor = AsyncAgentExecutor(news_agent=news_agent)

            result = await executor.execute("test", run_news=True, run_sec=False)

            assert result.news_result is not None
            assert result.sec_result is None
            assert result.news_result.response == "News result"

        asyncio.run(run_test())

    def test_execute_with_sec_only(self) -> None:
        """Execute with only SEC agent flag."""

        async def run_test() -> None:
            sec_agent = MockAgent(response="SEC result")
            executor = AsyncAgentExecutor(sec_agent=sec_agent)

            result = await executor.execute("test", run_news=False, run_sec=True)

            assert result.news_result is None
            assert result.sec_result is not None
            assert result.sec_result.response == "SEC result"

        asyncio.run(run_test())

    def test_execute_with_both_flags(self) -> None:
        """Execute with both agent flags."""

        async def run_test() -> None:
            news_agent = MockAgent(response="News")
            sec_agent = MockAgent(response="SEC")
            executor = AsyncAgentExecutor(news_agent=news_agent, sec_agent=sec_agent)

            result = await executor.execute("test", run_news=True, run_sec=True)

            assert result.news_result is not None
            assert result.sec_result is not None

        asyncio.run(run_test())

    def test_execute_with_no_flags_raises(self) -> None:
        """Execute with no flags should raise ValueError."""

        async def run_test() -> None:
            executor = AsyncAgentExecutor()

            with pytest.raises(ValueError, match="At least one agent must be specified"):
                await executor.execute("test", run_news=False, run_sec=False)

        asyncio.run(run_test())

    def test_execute_news_requested_but_not_configured(self) -> None:
        """Execute with news requested but not configured raises ValueError."""

        async def run_test() -> None:
            sec_agent = MockAgent()
            executor = AsyncAgentExecutor(sec_agent=sec_agent)

            with pytest.raises(ValueError, match="News agent is not configured"):
                await executor.execute("test", run_news=True, run_sec=False)

        asyncio.run(run_test())

    def test_execute_sec_requested_but_not_configured(self) -> None:
        """Execute with SEC requested but not configured raises ValueError."""

        async def run_test() -> None:
            news_agent = MockAgent()
            executor = AsyncAgentExecutor(news_agent=news_agent)

            with pytest.raises(ValueError, match="SEC agent is not configured"):
                await executor.execute("test", run_news=False, run_sec=True)

        asyncio.run(run_test())


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_all_results_empty(self) -> None:
        """all_results with no results."""
        result = ExecutionResult()
        assert result.all_results == []

    def test_all_results_with_news_only(self) -> None:
        """all_results with only news result."""
        news = AgentResult(agent_name="news_agent", status=AgentStatus.SUCCESS)
        result = ExecutionResult(news_result=news)

        assert len(result.all_results) == 1
        assert result.all_results[0] == news

    def test_all_results_with_both(self) -> None:
        """all_results with both results."""
        news = AgentResult(agent_name="news_agent", status=AgentStatus.SUCCESS)
        sec = AgentResult(agent_name="sec_agent", status=AgentStatus.SUCCESS)
        result = ExecutionResult(news_result=news, sec_result=sec)

        assert len(result.all_results) == 2

    def test_successful_results_filters_failures(self) -> None:
        """successful_results should only include SUCCESS status."""
        news = AgentResult(agent_name="news_agent", status=AgentStatus.SUCCESS)
        sec = AgentResult(agent_name="sec_agent", status=AgentStatus.ERROR)
        result = ExecutionResult(news_result=news, sec_result=sec)

        assert len(result.successful_results) == 1
        assert result.successful_results[0] == news

    def test_has_any_success_true(self) -> None:
        """has_any_success when at least one succeeded."""
        news = AgentResult(agent_name="news_agent", status=AgentStatus.SUCCESS)
        sec = AgentResult(agent_name="sec_agent", status=AgentStatus.TIMEOUT)
        result = ExecutionResult(news_result=news, sec_result=sec)

        assert result.has_any_success is True

    def test_has_any_success_false(self) -> None:
        """has_any_success when all failed."""
        news = AgentResult(agent_name="news_agent", status=AgentStatus.ERROR)
        sec = AgentResult(agent_name="sec_agent", status=AgentStatus.TIMEOUT)
        result = ExecutionResult(news_result=news, sec_result=sec)

        assert result.has_any_success is False

    def test_has_any_success_empty(self) -> None:
        """has_any_success with no results."""
        result = ExecutionResult()
        assert result.has_any_success is False

    def test_all_succeeded_true(self) -> None:
        """all_succeeded when all agents succeeded."""
        news = AgentResult(agent_name="news_agent", status=AgentStatus.SUCCESS)
        sec = AgentResult(agent_name="sec_agent", status=AgentStatus.SUCCESS)
        result = ExecutionResult(news_result=news, sec_result=sec)

        assert result.all_succeeded is True

    def test_all_succeeded_false_with_failure(self) -> None:
        """all_succeeded when one failed."""
        news = AgentResult(agent_name="news_agent", status=AgentStatus.SUCCESS)
        sec = AgentResult(agent_name="sec_agent", status=AgentStatus.ERROR)
        result = ExecutionResult(news_result=news, sec_result=sec)

        assert result.all_succeeded is False

    def test_all_succeeded_false_when_empty(self) -> None:
        """all_succeeded with no results should be False."""
        result = ExecutionResult()
        assert result.all_succeeded is False

    def test_total_execution_time(self) -> None:
        """total_execution_time_ms is preserved."""
        result = ExecutionResult(total_execution_time_ms=1234.5)
        assert result.total_execution_time_ms == 1234.5
