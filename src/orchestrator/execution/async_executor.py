"""Async executor for running agents in parallel with timeout handling."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from src.orchestrator.execution.result import AgentResult, AgentStatus


class AgentProtocol(Protocol):
    """Protocol defining the interface for sub-agents."""

    def query(self, user_message: str) -> str:
        """Execute a query and return the response."""
        ...


@dataclass
class ExecutionResult:
    """Result of executing one or more agents.

    Attributes:
        news_result: Result from the news agent (None if not executed).
        sec_result: Result from the SEC agent (None if not executed).
        total_execution_time_ms: Total wall-clock time for all executions.
    """

    news_result: AgentResult | None = None
    sec_result: AgentResult | None = None
    total_execution_time_ms: float = 0.0

    @property
    def all_results(self) -> list[AgentResult]:
        """Return all non-None results."""
        results = []
        if self.news_result is not None:
            results.append(self.news_result)
        if self.sec_result is not None:
            results.append(self.sec_result)
        return results

    @property
    def successful_results(self) -> list[AgentResult]:
        """Return only successful results."""
        return [r for r in self.all_results if r.is_success]

    @property
    def has_any_success(self) -> bool:
        """Check if at least one agent succeeded."""
        return len(self.successful_results) > 0

    @property
    def all_succeeded(self) -> bool:
        """Check if all executed agents succeeded."""
        results = self.all_results
        return len(results) > 0 and all(r.is_success for r in results)


class AsyncAgentExecutor:
    """Executes agents asynchronously with timeout handling.

    Uses asyncio.to_thread() to run synchronous agent.query() calls
    in parallel. Supports configurable timeout per agent.

    Attributes:
        news_agent: Optional news sentiment agent instance.
        sec_agent: Optional SEC filings agent instance.
        timeout_seconds: Maximum time to wait for each agent (default: 60).
    """

    def __init__(
        self,
        news_agent: AgentProtocol | None = None,
        sec_agent: AgentProtocol | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        """Initialize the executor with agent instances.

        Args:
            news_agent: Instance of the news sentiment agent.
            sec_agent: Instance of the SEC filings agent.
            timeout_seconds: Timeout for each agent execution in seconds.
        """
        self.news_agent = news_agent
        self.sec_agent = sec_agent
        self.timeout_seconds = timeout_seconds

    async def _execute_agent(
        self,
        agent: AgentProtocol,
        agent_name: Literal["news_agent", "sec_agent"],
        query: str,
    ) -> AgentResult:
        """Execute a single agent with timeout handling.

        Args:
            agent: The agent instance to execute.
            agent_name: Name identifier for the agent.
            query: The user query to send to the agent.

        Returns:
            AgentResult with status, response, and timing information.
        """
        start_time = time.perf_counter()

        try:
            # Run synchronous agent.query() in a thread pool
            response = await asyncio.wait_for(
                asyncio.to_thread(agent.query, query),
                timeout=self.timeout_seconds,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            return AgentResult(
                agent_name=agent_name,
                status=AgentStatus.SUCCESS,
                response=response,
                execution_time_ms=elapsed_ms,
            )

        except asyncio.TimeoutError:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return AgentResult(
                agent_name=agent_name,
                status=AgentStatus.TIMEOUT,
                error_message=f"Agent timed out after {self.timeout_seconds} seconds",
                execution_time_ms=elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return AgentResult(
                agent_name=agent_name,
                status=AgentStatus.ERROR,
                error_message=str(e),
                execution_time_ms=elapsed_ms,
            )

    async def execute_news(self, query: str) -> AgentResult:
        """Execute only the news agent.

        Args:
            query: The user query.

        Returns:
            AgentResult from the news agent.

        Raises:
            ValueError: If news agent is not configured.
        """
        if self.news_agent is None:
            raise ValueError("News agent is not configured")

        return await self._execute_agent(self.news_agent, "news_agent", query)

    async def execute_sec(self, query: str) -> AgentResult:
        """Execute only the SEC agent.

        Args:
            query: The user query.

        Returns:
            AgentResult from the SEC agent.

        Raises:
            ValueError: If SEC agent is not configured.
        """
        if self.sec_agent is None:
            raise ValueError("SEC agent is not configured")

        return await self._execute_agent(self.sec_agent, "sec_agent", query)

    async def execute_both(self, query: str) -> ExecutionResult:
        """Execute both agents in parallel.

        Args:
            query: The user query.

        Returns:
            ExecutionResult containing results from both agents.

        Raises:
            ValueError: If neither agent is configured.
        """
        if self.news_agent is None and self.sec_agent is None:
            raise ValueError("At least one agent must be configured")

        start_time = time.perf_counter()
        tasks: list[asyncio.Task[AgentResult]] = []

        # Create tasks for available agents
        if self.news_agent is not None:
            tasks.append(
                asyncio.create_task(
                    self._execute_agent(self.news_agent, "news_agent", query)
                )
            )
        if self.sec_agent is not None:
            tasks.append(
                asyncio.create_task(
                    self._execute_agent(self.sec_agent, "sec_agent", query)
                )
            )

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=False)

        total_time_ms = (time.perf_counter() - start_time) * 1000

        # Map results to the appropriate fields
        news_result: AgentResult | None = None
        sec_result: AgentResult | None = None

        for result in results:
            if result.agent_name == "news_agent":
                news_result = result
            elif result.agent_name == "sec_agent":
                sec_result = result

        return ExecutionResult(
            news_result=news_result,
            sec_result=sec_result,
            total_execution_time_ms=total_time_ms,
        )

    async def execute(
        self,
        query: str,
        run_news: bool = False,
        run_sec: bool = False,
    ) -> ExecutionResult:
        """Execute specified agents based on flags.

        This is a convenience method that dispatches to the appropriate
        execution method based on which agents should run.

        Args:
            query: The user query.
            run_news: Whether to run the news agent.
            run_sec: Whether to run the SEC agent.

        Returns:
            ExecutionResult containing results from executed agents.

        Raises:
            ValueError: If no agents are specified or required agents not configured.
        """
        if not run_news and not run_sec:
            raise ValueError("At least one agent must be specified to run")

        start_time = time.perf_counter()
        tasks: list[asyncio.Task[AgentResult]] = []

        if run_news:
            if self.news_agent is None:
                raise ValueError("News agent is not configured but was requested")
            tasks.append(
                asyncio.create_task(
                    self._execute_agent(self.news_agent, "news_agent", query)
                )
            )

        if run_sec:
            if self.sec_agent is None:
                raise ValueError("SEC agent is not configured but was requested")
            tasks.append(
                asyncio.create_task(
                    self._execute_agent(self.sec_agent, "sec_agent", query)
                )
            )

        results = await asyncio.gather(*tasks, return_exceptions=False)
        total_time_ms = (time.perf_counter() - start_time) * 1000

        news_result: AgentResult | None = None
        sec_result: AgentResult | None = None

        for result in results:
            if result.agent_name == "news_agent":
                news_result = result
            elif result.agent_name == "sec_agent":
                sec_result = result

        return ExecutionResult(
            news_result=news_result,
            sec_result=sec_result,
            total_execution_time_ms=total_time_ms,
        )
