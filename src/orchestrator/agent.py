"""Main OrchestrationAgent class - coordinates news and SEC agents."""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from typing import Any, Literal

from news_agent.agent import NewsSentimentAgent
from sec_agent.agent import SECFilingsAgent
from orchestrator.comparison import (
    ComparisonResult,
    DiscrepancyDetector,
    SignalExtractionResult,
    SignalExtractor,
)
from orchestrator.execution import (
    AgentResult,
    AsyncAgentExecutor,
    ExecutionResult,
    RouteType,
)
from orchestrator.memory import AgentResultEntry, OrchestratorMemory
from orchestrator.routing import QueryClassifier
from orchestrator.synthesis import (
    ResponseSynthesizer,
    SynthesisInput,
    SynthesisResult,
)


@dataclass
class OrchestrationResult:
    """Result of an orchestrated query.

    Attributes:
        response: The synthesized response text.
        route_type: How the query was routed (news_only, sec_only, both).
        agents_used: List of agent names that were executed.
        had_discrepancies: Whether discrepancies were detected between sources.
        confidence: Confidence score (0.0-1.0) based on source alignment.
        execution_time_ms: Total execution time in milliseconds.
        query_id: Unique identifier for this query in memory.
        news_result: Raw result from news agent (if executed).
        sec_result: Raw result from SEC agent (if executed).
        comparison: Comparison result (if both agents executed).
    """

    response: str
    route_type: Literal["news_only", "sec_only", "both"]
    agents_used: list[str]
    had_discrepancies: bool
    confidence: float
    execution_time_ms: float
    query_id: str
    news_result: AgentResult | None = None
    sec_result: AgentResult | None = None
    comparison: ComparisonResult | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "response": self.response,
            "route_type": self.route_type,
            "agents_used": self.agents_used,
            "had_discrepancies": self.had_discrepancies,
            "confidence": self.confidence,
            "execution_time_ms": self.execution_time_ms,
            "query_id": self.query_id,
            "news_result": self.news_result.to_dict() if self.news_result else None,
            "sec_result": self.sec_result.to_dict() if self.sec_result else None,
            "comparison": self.comparison.to_dict() if self.comparison else None,
        }


class OrchestrationAgent:
    """Coordinates news and SEC agents for unified company intelligence.

    The orchestration agent:
    1. Classifies queries to determine which agent(s) to call
    2. Executes agents in parallel for performance
    3. Extracts signals and detects discrepancies between sources
    4. Synthesizes a unified response using an LLM
    5. Tracks query history in memory

    Example:
        >>> agent = OrchestrationAgent()
        >>> result = agent.query("What's the outlook for Apple?", ticker="AAPL")
        >>> print(result.response)
    """

    def __init__(
        self,
        model_id: str | None = None,
        bedrock_client: Any | None = None,
        news_agent: NewsSentimentAgent | None = None,
        sec_agent: SECFilingsAgent | None = None,
        timeout_seconds: float | None = None,
        enable_comparison: bool | None = None,
        memory: OrchestratorMemory | None = None,
    ) -> None:
        """Initialize the orchestration agent.

        Args:
            model_id: Bedrock model ID for synthesis (default from env or Claude Opus 4.5).
            bedrock_client: Optional pre-configured boto3 Bedrock client.
            news_agent: Optional pre-configured news agent instance.
            sec_agent: Optional pre-configured SEC agent instance.
            timeout_seconds: Timeout for agent execution (default from env or 60s).
            enable_comparison: Whether to enable discrepancy detection (default from env or True).
            memory: Optional pre-configured memory instance.
        """
        # Model configuration
        self.model_id = model_id or os.environ.get(
            "BEDROCK_MODEL_ID", "anthropic.claude-opus-4-5-20251101-v1:0"
        )
        self.bedrock_client = bedrock_client

        # Timeout configuration
        default_timeout = float(
            os.environ.get("ORCHESTRATOR_TIMEOUT_SECONDS", "60")
        )
        self.timeout_seconds = (
            timeout_seconds if timeout_seconds is not None else default_timeout
        )

        # Comparison configuration
        default_comparison = (
            os.environ.get("ORCHESTRATOR_ENABLE_COMPARISON", "true").lower() == "true"
        )
        self.enable_comparison = (
            enable_comparison if enable_comparison is not None else default_comparison
        )

        # Initialize sub-agents (lazy initialization if not provided)
        self._news_agent = news_agent
        self._sec_agent = sec_agent
        self._agents_initialized = news_agent is not None or sec_agent is not None

        # Initialize components
        self._classifier = QueryClassifier()
        self._executor: AsyncAgentExecutor | None = None
        self._signal_extractor = SignalExtractor()
        self._discrepancy_detector = DiscrepancyDetector()
        self._synthesizer = ResponseSynthesizer(
            model_id=self.model_id,
            bedrock_client=self.bedrock_client,
        )
        self._memory = memory or OrchestratorMemory()

    def _ensure_agents_initialized(self) -> None:
        """Lazily initialize sub-agents if not already done."""
        if not self._agents_initialized:
            self._news_agent = NewsSentimentAgent(
                bedrock_client=self.bedrock_client
            )
            self._sec_agent = SECFilingsAgent(
                bedrock_client=self.bedrock_client
            )
            self._agents_initialized = True

        # Initialize executor with current agents
        if self._executor is None:
            self._executor = AsyncAgentExecutor(
                news_agent=self._news_agent,
                sec_agent=self._sec_agent,
                timeout_seconds=self.timeout_seconds,
            )

    @property
    def memory(self) -> OrchestratorMemory:
        """Get the orchestrator memory instance."""
        return self._memory

    @property
    def session_id(self) -> str:
        """Get the current session ID."""
        return self._memory.session_id

    def reset(self) -> None:
        """Reset the orchestrator state.

        Clears memory and resets sub-agents if initialized.
        """
        self._memory.clear()
        if self._news_agent is not None:
            self._news_agent.reset()
        if self._sec_agent is not None:
            self._sec_agent.reset()

    def _extract_ticker_from_query(self, query: str) -> str | None:
        """Extract ticker symbol from query if present.

        Args:
            query: The user query.

        Returns:
            Extracted ticker symbol or None.
        """
        # Common patterns for tickers
        patterns = [
            r"\b([A-Z]{1,5})\b(?:\s+stock|\s+shares|\s+company)?",  # AAPL, TSLA
            r"ticker[:\s]+([A-Z]{1,5})\b",  # ticker: AAPL
            r"\$([A-Z]{1,5})\b",  # $AAPL
        ]

        # Known company name to ticker mappings
        company_tickers = {
            "apple": "AAPL",
            "microsoft": "MSFT",
            "google": "GOOGL",
            "alphabet": "GOOGL",
            "amazon": "AMZN",
            "tesla": "TSLA",
            "meta": "META",
            "facebook": "META",
            "nvidia": "NVDA",
            "netflix": "NFLX",
        }

        # Check company names first
        query_lower = query.lower()
        for company, ticker in company_tickers.items():
            if company in query_lower:
                return ticker

        # Try regex patterns
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                ticker = match.group(1)
                # Filter out common words that look like tickers
                if ticker not in {"THE", "AND", "FOR", "NOT", "ARE", "BUT", "SEC", "CEO"}:
                    return ticker

        return None

    def query(
        self,
        user_message: str,
        ticker: str | None = None,
        force_route: Literal["news_only", "sec_only", "both"] | None = None,
    ) -> OrchestrationResult:
        """Execute an orchestrated query.

        This is the main entry point for querying the orchestration agent.
        It classifies the query, executes appropriate agents, compares results,
        and synthesizes a unified response.

        Args:
            user_message: The user's query.
            ticker: Optional ticker symbol (extracted from query if not provided).
            force_route: Optional override for query routing.

        Returns:
            OrchestrationResult containing the synthesized response and metadata.
        """
        # Run the async query in the event loop
        return asyncio.run(
            self._query_async(user_message, ticker, force_route)
        )

    async def query_async(
        self,
        user_message: str,
        ticker: str | None = None,
        force_route: Literal["news_only", "sec_only", "both"] | None = None,
    ) -> OrchestrationResult:
        """Execute an orchestrated query asynchronously.

        Same as query() but for use in async contexts.

        Args:
            user_message: The user's query.
            ticker: Optional ticker symbol (extracted from query if not provided).
            force_route: Optional override for query routing.

        Returns:
            OrchestrationResult containing the synthesized response and metadata.
        """
        return await self._query_async(user_message, ticker, force_route)

    async def _query_async(
        self,
        user_message: str,
        ticker: str | None = None,
        force_route: Literal["news_only", "sec_only", "both"] | None = None,
    ) -> OrchestrationResult:
        """Internal async implementation of query.

        Args:
            user_message: The user's query.
            ticker: Optional ticker symbol.
            force_route: Optional routing override.

        Returns:
            OrchestrationResult with synthesized response.
        """
        # Ensure agents are initialized
        self._ensure_agents_initialized()
        assert self._executor is not None  # For type checker

        # Extract ticker if not provided
        if ticker is None:
            ticker = self._extract_ticker_from_query(user_message)

        # Classify the query or use forced route
        if force_route is not None:
            route_type = RouteType(force_route)
        else:
            classification = self._classifier.classify(user_message)
            route_type = classification.route_type

        # Determine which agents to run
        run_news = route_type in (RouteType.NEWS_ONLY, RouteType.BOTH)
        run_sec = route_type in (RouteType.SEC_ONLY, RouteType.BOTH)

        # Execute agents
        execution_result = await self._executor.execute(
            query=user_message,
            run_news=run_news,
            run_sec=run_sec,
        )

        # Extract signals for comparison (if enabled and both agents ran)
        news_signals: SignalExtractionResult | None = None
        sec_signals: SignalExtractionResult | None = None
        comparison: ComparisonResult | None = None

        if self.enable_comparison and run_news and run_sec:
            # Extract signals from successful results
            if (
                execution_result.news_result is not None
                and execution_result.news_result.is_success
                and execution_result.news_result.response
            ):
                news_signals = self._signal_extractor.extract(
                    execution_result.news_result.response, "news_agent"
                )

            if (
                execution_result.sec_result is not None
                and execution_result.sec_result.is_success
                and execution_result.sec_result.response
            ):
                sec_signals = self._signal_extractor.extract(
                    execution_result.sec_result.response, "sec_agent"
                )

            # Compare signals
            comparison = self._discrepancy_detector.compare(
                news_signals, sec_signals
            )

        # Synthesize response
        synthesis_input = SynthesisInput(
            user_query=user_message,
            news_result=execution_result.news_result,
            sec_result=execution_result.sec_result,
            comparison=comparison,
            ticker=ticker,
        )
        synthesis_result = self._synthesizer.synthesize(synthesis_input)

        # Build list of agents used
        agents_used: list[str] = []
        if execution_result.news_result is not None:
            agents_used.append("news_agent")
        if execution_result.sec_result is not None:
            agents_used.append("sec_agent")

        # Convert route_type to string
        route_type_str: Literal["news_only", "sec_only", "both"] = route_type.value

        # Build agent result entries for memory
        agent_results: list[AgentResultEntry] = []
        if execution_result.news_result is not None:
            agent_results.append(
                AgentResultEntry(
                    agent_name=execution_result.news_result.agent_name,
                    status=execution_result.news_result.status.value,
                    response=execution_result.news_result.response,
                    error_message=execution_result.news_result.error_message,
                    execution_time_ms=execution_result.news_result.execution_time_ms,
                )
            )
        if execution_result.sec_result is not None:
            agent_results.append(
                AgentResultEntry(
                    agent_name=execution_result.sec_result.agent_name,
                    status=execution_result.sec_result.status.value,
                    response=execution_result.sec_result.response,
                    error_message=execution_result.sec_result.error_message,
                    execution_time_ms=execution_result.sec_result.execution_time_ms,
                )
            )

        # Store in memory
        query_id = self._memory.store_query(
            user_query=user_message,
            route_type=route_type_str,
            agents_called=agents_used,
            agent_results=agent_results,
            synthesized_response=synthesis_result.response,
            confidence=synthesis_result.confidence,
            ticker=ticker,
            had_discrepancies=synthesis_result.had_discrepancies,
            total_execution_time_ms=execution_result.total_execution_time_ms,
        )

        return OrchestrationResult(
            response=synthesis_result.response,
            route_type=route_type_str,
            agents_used=agents_used,
            had_discrepancies=synthesis_result.had_discrepancies,
            confidence=synthesis_result.confidence,
            execution_time_ms=execution_result.total_execution_time_ms,
            query_id=query_id,
            news_result=execution_result.news_result,
            sec_result=execution_result.sec_result,
            comparison=comparison,
        )

    def compare(self, ticker: str) -> OrchestrationResult:
        """Run an explicit comparison between news and SEC data for a ticker.

        This forces both agents to run and enables comparison.

        Args:
            ticker: The ticker symbol to analyze.

        Returns:
            OrchestrationResult with comparison data.
        """
        query = f"What is the current outlook and any potential risks for {ticker}?"
        return self.query(query, ticker=ticker, force_route="both")

    def get_session_summary(self) -> dict[str, Any]:
        """Get a summary of the current session.

        Returns:
            Dictionary with session statistics and analytics.
        """
        return self._memory.get_session_summary()

    def get_recent_queries(
        self,
        ticker: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get recent queries from memory.

        Args:
            ticker: Optional filter by ticker symbol.
            limit: Optional limit on number of results.

        Returns:
            List of query records.
        """
        return self._memory.get_recent_queries(ticker=ticker, limit=limit)
