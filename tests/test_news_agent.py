"""Tests for the News Sentiment Agent."""

import json
from unittest.mock import MagicMock, patch

import pytest

from news_agent.agent import NewsSentimentAgent, _execute_tool
from news_agent.memory.agent_memory import AgentMemory


@pytest.fixture
def mock_bedrock_client() -> MagicMock:
    """Create a mock Bedrock client."""
    mock = MagicMock()
    return mock


@pytest.fixture
def agent(mock_bedrock_client: MagicMock) -> NewsSentimentAgent:
    """Create an agent with mock Bedrock client."""
    return NewsSentimentAgent(bedrock_client=mock_bedrock_client)


class TestToolExecution:
    """Tests for tool execution."""

    def test_execute_unknown_tool(self) -> None:
        """Test executing an unknown tool returns error."""
        result = json.loads(_execute_tool("unknown_tool", {}))
        assert result["status"] == "error"
        assert "Unknown tool" in result["message"]

    def test_execute_memory_tool_without_memory(self) -> None:
        """Test memory tool without memory instance returns error."""
        result = json.loads(_execute_tool("get_memory", {}, memory=None))
        assert result["status"] == "error"
        assert "Memory not available" in result["message"]

    def test_execute_get_memory(self) -> None:
        """Test get_memory tool execution."""
        memory = AgentMemory()
        memory.store_analysis(
            ticker="AAPL",
            company_name="Apple Inc",
            summary="Test summary",
            sentiment="positive",
            sentiment_score=0.5,
            material_events=["Event 1"],
            articles_analyzed=10,
        )

        result = json.loads(_execute_tool("get_memory", {"ticker": "AAPL"}, memory=memory))
        assert result["status"] == "ok"
        assert len(result["analyses"]) == 1

    def test_execute_save_memory(self) -> None:
        """Test save_memory tool execution."""
        memory = AgentMemory()

        result = json.loads(
            _execute_tool(
                "save_memory",
                {
                    "ticker": "AAPL",
                    "company_name": "Apple Inc",
                    "summary": "Test summary",
                    "sentiment": "positive",
                    "sentiment_score": 0.5,
                },
                memory=memory,
            )
        )

        assert result["status"] == "ok"
        assert len(memory.get_recent_analyses()) == 1

    @patch("news_agent.tools.fetch_news.NewsCache")
    @patch("news_agent.tools.fetch_news.EntityResolver")
    def test_execute_fetch_news(
        self, mock_resolver_cls: MagicMock, mock_cache_cls: MagicMock
    ) -> None:
        """Test fetch_news tool execution."""
        # Mock EntityResolver
        mock_resolver = MagicMock()
        mock_resolver.resolve.return_value = {
            "ticker": "AAPL",
            "cik": "0000320193",
            "company_name": "Apple Inc",
        }
        mock_resolver_cls.return_value = mock_resolver

        # Mock NewsCache with cached articles
        mock_cache = MagicMock()
        mock_cache.get_cached_articles.return_value = ([], True, None)
        mock_cache.is_rate_limited.return_value = True
        mock_cache_cls.return_value = mock_cache

        result = json.loads(_execute_tool("fetch_news", {"ticker": "AAPL"}))
        # Should return rate_limited since cache is empty and rate limited
        assert result["status"] == "rate_limited"


class TestNewsSentimentAgent:
    """Tests for NewsSentimentAgent class."""

    def test_init_default_values(self, agent: NewsSentimentAgent) -> None:
        """Test agent initializes with correct defaults."""
        assert agent._max_turns == 10
        assert agent._messages == []
        assert isinstance(agent._memory, AgentMemory)

    def test_reset(self, agent: NewsSentimentAgent) -> None:
        """Test agent reset clears messages and memory."""
        # Add some state
        agent._messages.append({"role": "user", "content": [{"text": "test"}]})
        agent._memory.store_analysis(
            ticker="AAPL",
            company_name="Apple Inc",
            summary="Test",
            sentiment="positive",
            sentiment_score=0.5,
            material_events=[],
            articles_analyzed=5,
        )

        agent.reset()

        assert agent._messages == []
        assert len(agent._memory.get_recent_analyses()) == 0

    def test_memory_property(self, agent: NewsSentimentAgent) -> None:
        """Test memory property returns memory instance."""
        assert agent.memory is agent._memory

    def test_query_end_turn(
        self, agent: NewsSentimentAgent, mock_bedrock_client: MagicMock
    ) -> None:
        """Test query returns text on end_turn."""
        mock_bedrock_client.converse.return_value = {
            "stopReason": "end_turn",
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "Test response"}],
                }
            },
        }

        result = agent.query("Hello")

        assert result == "Test response"
        assert len(agent._messages) == 2  # User + assistant

    def test_query_with_tool_use(
        self, agent: NewsSentimentAgent, mock_bedrock_client: MagicMock
    ) -> None:
        """Test query handles tool use and continues."""
        # First call returns tool_use, second returns end_turn
        mock_bedrock_client.converse.side_effect = [
            {
                "stopReason": "tool_use",
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "test-id",
                                    "name": "get_memory",
                                    "input": {},
                                }
                            }
                        ],
                    }
                },
            },
            {
                "stopReason": "end_turn",
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": "Final response"}],
                    }
                },
            },
        ]

        result = agent.query("What have I analyzed?")

        assert result == "Final response"
        assert mock_bedrock_client.converse.call_count == 2

    def test_query_max_turns_exceeded(
        self, agent: NewsSentimentAgent, mock_bedrock_client: MagicMock
    ) -> None:
        """Test query stops after max turns."""
        agent._max_turns = 2

        # Always return tool_use to force max turns
        mock_bedrock_client.converse.return_value = {
            "stopReason": "tool_use",
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "test-id",
                                "name": "get_memory",
                                "input": {},
                            }
                        }
                    ],
                }
            },
        }

        result = agent.query("Endless loop")

        assert "unable to complete" in result.lower()
        assert mock_bedrock_client.converse.call_count == 2


class TestAgentMemory:
    """Tests for AgentMemory integration."""

    def test_memory_session_id(self) -> None:
        """Test memory has a session ID."""
        memory = AgentMemory()
        assert memory.session_id is not None
        assert len(memory.session_id) > 0

    def test_memory_store_and_retrieve(self) -> None:
        """Test storing and retrieving analyses."""
        memory = AgentMemory()

        memory.store_analysis(
            ticker="AAPL",
            company_name="Apple Inc",
            summary="Positive earnings",
            sentiment="positive",
            sentiment_score=0.6,
            material_events=["Earnings beat"],
            articles_analyzed=25,
        )

        memory.store_analysis(
            ticker="TSLA",
            company_name="Tesla Inc",
            summary="Mixed coverage",
            sentiment="mixed",
            sentiment_score=-0.1,
            material_events=["Price cuts"],
            articles_analyzed=30,
        )

        # Get all
        all_analyses = memory.get_recent_analyses()
        assert len(all_analyses) == 2

        # Filter by ticker
        aapl_analyses = memory.get_recent_analyses(ticker="AAPL")
        assert len(aapl_analyses) == 1
        assert aapl_analyses[0]["ticker"] == "AAPL"

    def test_memory_session_context(self) -> None:
        """Test getting full session context."""
        memory = AgentMemory()
        memory.store_analysis(
            ticker="AAPL",
            company_name="Apple Inc",
            summary="Test",
            sentiment="positive",
            sentiment_score=0.5,
            material_events=[],
            articles_analyzed=10,
        )

        context = memory.get_session_context()

        assert "session_id" in context
        assert "entries" in context
        assert len(context["entries"]) == 1

    def test_memory_clear(self) -> None:
        """Test clearing memory."""
        memory = AgentMemory()
        memory.store_analysis(
            ticker="AAPL",
            company_name="Apple Inc",
            summary="Test",
            sentiment="positive",
            sentiment_score=0.5,
            material_events=[],
            articles_analyzed=10,
        )

        memory.clear()

        assert len(memory.get_recent_analyses()) == 0
