"""Tests for the orchestrator CLI interface."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from src.orchestrator.cli.main import cli


# Helper to create a mock OrchestrationResult
def _mock_result(
    response: str = "Test response",
    route_type: str = "both",
    agents_used: list[str] | None = None,
    had_discrepancies: bool = False,
    confidence: float = 0.8,
    execution_time_ms: float = 1500.0,
    query_id: str = "test-query-id",
    comparison: MagicMock | None = None,
) -> MagicMock:
    """Create a mock OrchestrationResult."""
    result = MagicMock()
    result.response = response
    result.route_type = route_type
    result.agents_used = agents_used or ["news_agent", "sec_agent"]
    result.had_discrepancies = had_discrepancies
    result.confidence = confidence
    result.execution_time_ms = execution_time_ms
    result.query_id = query_id
    result.comparison = comparison
    result.news_result = None
    result.sec_result = None
    result.to_dict.return_value = {
        "response": response,
        "route_type": route_type,
        "agents_used": agents_used or ["news_agent", "sec_agent"],
        "had_discrepancies": had_discrepancies,
        "confidence": confidence,
        "execution_time_ms": execution_time_ms,
        "query_id": query_id,
    }
    return result


class TestChatCommand:
    """Tests for the chat command."""

    @patch("src.orchestrator.cli.main.OrchestrationAgent")
    def test_chat_exit(self, mock_agent_cls: MagicMock) -> None:
        """Test that chat exits cleanly on 'exit' command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["chat"], input="exit\n")
        assert result.exit_code == 0
        assert "Interactive Mode" in result.output
        assert "Goodbye" in result.output

    @patch("src.orchestrator.cli.main.OrchestrationAgent")
    def test_chat_quit(self, mock_agent_cls: MagicMock) -> None:
        """Test that chat exits cleanly on 'quit' command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["chat"], input="quit\n")
        assert result.exit_code == 0
        assert "Goodbye" in result.output

    @patch("src.orchestrator.cli.main.OrchestrationAgent")
    def test_chat_sends_query(self, mock_agent_cls: MagicMock) -> None:
        """Test that chat sends queries to the agent."""
        mock_agent = MagicMock()
        mock_agent.query.return_value = _mock_result(response="Apple outlook is positive.")
        mock_agent_cls.return_value = mock_agent

        runner = CliRunner()
        result = runner.invoke(cli, ["chat"], input="What's the outlook for Apple?\nexit\n")
        assert result.exit_code == 0
        mock_agent.query.assert_called_once_with("What's the outlook for Apple?")
        assert "Apple outlook is positive." in result.output

    @patch("src.orchestrator.cli.main.OrchestrationAgent")
    def test_chat_handles_error(self, mock_agent_cls: MagicMock) -> None:
        """Test that chat handles errors gracefully."""
        mock_agent = MagicMock()
        mock_agent.query.side_effect = RuntimeError("Connection failed")
        mock_agent_cls.return_value = mock_agent

        runner = CliRunner()
        result = runner.invoke(cli, ["chat"], input="test\nexit\n")
        assert "Error: Connection failed" in result.output

    @patch("src.orchestrator.cli.main.OrchestrationAgent")
    def test_chat_skips_empty_input(self, mock_agent_cls: MagicMock) -> None:
        """Test that chat skips empty input."""
        mock_agent = MagicMock()
        mock_agent_cls.return_value = mock_agent

        runner = CliRunner()
        result = runner.invoke(cli, ["chat"], input="\nexit\n")
        assert result.exit_code == 0
        assert "Goodbye" in result.output
        mock_agent.query.assert_not_called()

    @patch("src.orchestrator.cli.main.OrchestrationAgent")
    def test_chat_summary_command(self, mock_agent_cls: MagicMock) -> None:
        """Test that 'summary' command shows session summary."""
        mock_agent = MagicMock()
        mock_agent.get_session_summary.return_value = {
            "session_id": "test-session",
            "query_count": 5,
            "tickers_analyzed": ["AAPL", "TSLA"],
            "discrepancy_count": 2,
            "average_confidence": 0.75,
        }
        mock_agent_cls.return_value = mock_agent

        runner = CliRunner()
        result = runner.invoke(cli, ["chat"], input="summary\nexit\n")
        assert result.exit_code == 0
        assert "SESSION SUMMARY" in result.output
        assert "test-session" in result.output
        assert "AAPL" in result.output

    @patch("src.orchestrator.cli.main.OrchestrationAgent")
    def test_chat_history_command(self, mock_agent_cls: MagicMock) -> None:
        """Test that 'history' command shows query history."""
        mock_agent = MagicMock()
        mock_agent.get_recent_queries.return_value = [
            {"ticker": "AAPL", "user_query": "What's Apple's outlook?", "confidence": 0.8, "had_discrepancies": False},
            {"ticker": "TSLA", "user_query": "Tesla risk factors?", "confidence": 0.6, "had_discrepancies": True},
        ]
        mock_agent_cls.return_value = mock_agent

        runner = CliRunner()
        result = runner.invoke(cli, ["chat"], input="history\nexit\n")
        assert result.exit_code == 0
        assert "RECENT QUERIES" in result.output
        assert "AAPL" in result.output
        assert "TSLA" in result.output

    @patch("src.orchestrator.cli.main.OrchestrationAgent")
    def test_chat_reset_command(self, mock_agent_cls: MagicMock) -> None:
        """Test that 'reset' command resets the session."""
        mock_agent = MagicMock()
        mock_agent_cls.return_value = mock_agent

        runner = CliRunner()
        result = runner.invoke(cli, ["chat"], input="reset\nexit\n")
        assert result.exit_code == 0
        assert "Session reset" in result.output
        mock_agent.reset.assert_called_once()

    @patch("src.orchestrator.cli.main.OrchestrationAgent")
    def test_chat_shows_discrepancy_indicator(self, mock_agent_cls: MagicMock) -> None:
        """Test that chat shows discrepancy indicator when present."""
        mock_agent = MagicMock()
        mock_agent.query.return_value = _mock_result(
            response="Sources disagree.",
            had_discrepancies=True,
        )
        mock_agent_cls.return_value = mock_agent

        runner = CliRunner()
        result = runner.invoke(cli, ["chat"], input="Compare Apple\nexit\n")
        assert result.exit_code == 0
        assert "Discrepancies detected" in result.output


class TestQueryCommand:
    """Tests for the query command."""

    @patch("src.orchestrator.cli.main.OrchestrationAgent")
    def test_query_basic(self, mock_agent_cls: MagicMock) -> None:
        """Test basic query execution."""
        mock_agent = MagicMock()
        mock_agent.query.return_value = _mock_result(response="Apple is doing well.")
        mock_agent_cls.return_value = mock_agent

        runner = CliRunner()
        result = runner.invoke(cli, ["query", "What's the outlook for Apple?"])
        assert result.exit_code == 0
        assert "Apple is doing well." in result.output
        mock_agent.query.assert_called_once_with(
            "What's the outlook for Apple?",
            ticker=None,
            force_route=None,
        )

    @patch("src.orchestrator.cli.main.OrchestrationAgent")
    def test_query_with_ticker(self, mock_agent_cls: MagicMock) -> None:
        """Test query with explicit ticker."""
        mock_agent = MagicMock()
        mock_agent.query.return_value = _mock_result()
        mock_agent_cls.return_value = mock_agent

        runner = CliRunner()
        result = runner.invoke(cli, ["query", "What's the outlook?", "-t", "AAPL"])
        assert result.exit_code == 0
        mock_agent.query.assert_called_once_with(
            "What's the outlook?",
            ticker="AAPL",
            force_route=None,
        )

    @patch("src.orchestrator.cli.main.OrchestrationAgent")
    def test_query_with_source_news(self, mock_agent_cls: MagicMock) -> None:
        """Test query with news source."""
        mock_agent = MagicMock()
        mock_agent.query.return_value = _mock_result(route_type="news_only", agents_used=["news_agent"])
        mock_agent_cls.return_value = mock_agent

        runner = CliRunner()
        result = runner.invoke(cli, ["query", "What's the sentiment?", "-s", "news"])
        assert result.exit_code == 0
        mock_agent.query.assert_called_once_with(
            "What's the sentiment?",
            ticker=None,
            force_route="news_only",
        )

    @patch("src.orchestrator.cli.main.OrchestrationAgent")
    def test_query_with_source_sec(self, mock_agent_cls: MagicMock) -> None:
        """Test query with SEC source."""
        mock_agent = MagicMock()
        mock_agent.query.return_value = _mock_result(route_type="sec_only", agents_used=["sec_agent"])
        mock_agent_cls.return_value = mock_agent

        runner = CliRunner()
        result = runner.invoke(cli, ["query", "What are the risk factors?", "-s", "sec"])
        assert result.exit_code == 0
        mock_agent.query.assert_called_once_with(
            "What are the risk factors?",
            ticker=None,
            force_route="sec_only",
        )

    @patch("src.orchestrator.cli.main.OrchestrationAgent")
    def test_query_with_source_both(self, mock_agent_cls: MagicMock) -> None:
        """Test query with both sources."""
        mock_agent = MagicMock()
        mock_agent.query.return_value = _mock_result()
        mock_agent_cls.return_value = mock_agent

        runner = CliRunner()
        result = runner.invoke(cli, ["query", "Compare sources", "-s", "both"])
        assert result.exit_code == 0
        mock_agent.query.assert_called_once_with(
            "Compare sources",
            ticker=None,
            force_route="both",
        )

    @patch("src.orchestrator.cli.main.OrchestrationAgent")
    def test_query_json_output(self, mock_agent_cls: MagicMock) -> None:
        """Test query with JSON output."""
        mock_agent = MagicMock()
        mock_agent.query.return_value = _mock_result(response="JSON response")
        mock_agent_cls.return_value = mock_agent

        runner = CliRunner()
        result = runner.invoke(cli, ["query", "test", "-j"])
        assert result.exit_code == 0
        assert '"response": "JSON response"' in result.output
        assert '"confidence": 0.8' in result.output

    @patch("src.orchestrator.cli.main.OrchestrationAgent")
    def test_query_error(self, mock_agent_cls: MagicMock) -> None:
        """Test query handles errors."""
        mock_agent = MagicMock()
        mock_agent.query.side_effect = RuntimeError("API error")
        mock_agent_cls.return_value = mock_agent

        runner = CliRunner()
        result = runner.invoke(cli, ["query", "test"])
        assert result.exit_code == 1
        assert "Error: API error" in result.output

    @patch("src.orchestrator.cli.main.OrchestrationAgent")
    def test_query_shows_metadata(self, mock_agent_cls: MagicMock) -> None:
        """Test query shows metadata (agents, route, confidence)."""
        mock_agent = MagicMock()
        mock_agent.query.return_value = _mock_result(
            response="Test response",
            route_type="both",
            agents_used=["news_agent", "sec_agent"],
            confidence=0.85,
            execution_time_ms=2500.0,
        )
        mock_agent_cls.return_value = mock_agent

        runner = CliRunner()
        result = runner.invoke(cli, ["query", "test"])
        assert result.exit_code == 0
        assert "news_agent" in result.output
        assert "sec_agent" in result.output
        assert "both" in result.output
        assert "85%" in result.output
        assert "2500ms" in result.output

    def test_query_invalid_source(self) -> None:
        """Test query rejects invalid source."""
        runner = CliRunner()
        result = runner.invoke(cli, ["query", "test", "-s", "invalid"])
        assert result.exit_code != 0


class TestCompareCommand:
    """Tests for the compare command."""

    @patch("src.orchestrator.cli.main.OrchestrationAgent")
    def test_compare_basic(self, mock_agent_cls: MagicMock) -> None:
        """Test basic compare execution."""
        mock_agent = MagicMock()
        mock_agent.compare.return_value = _mock_result(
            response="Comparison analysis for AAPL",
            had_discrepancies=True,
        )
        mock_agent_cls.return_value = mock_agent

        runner = CliRunner()
        result = runner.invoke(cli, ["compare", "AAPL"])
        assert result.exit_code == 0
        assert "COMPARISON ANALYSIS: AAPL" in result.output
        mock_agent.compare.assert_called_once_with("AAPL")

    @patch("src.orchestrator.cli.main.OrchestrationAgent")
    def test_compare_lowercase_ticker(self, mock_agent_cls: MagicMock) -> None:
        """Test compare converts ticker to uppercase."""
        mock_agent = MagicMock()
        mock_agent.compare.return_value = _mock_result()
        mock_agent_cls.return_value = mock_agent

        runner = CliRunner()
        result = runner.invoke(cli, ["compare", "aapl"])
        assert result.exit_code == 0
        mock_agent.compare.assert_called_once_with("AAPL")

    @patch("src.orchestrator.cli.main.OrchestrationAgent")
    def test_compare_json_output(self, mock_agent_cls: MagicMock) -> None:
        """Test compare with JSON output."""
        mock_agent = MagicMock()
        mock_agent.compare.return_value = _mock_result(
            response="Comparison",
            had_discrepancies=True,
        )
        mock_agent_cls.return_value = mock_agent

        runner = CliRunner()
        result = runner.invoke(cli, ["compare", "TSLA", "-j"])
        assert result.exit_code == 0
        assert '"response": "Comparison"' in result.output
        assert '"had_discrepancies": true' in result.output

    @patch("src.orchestrator.cli.main.OrchestrationAgent")
    def test_compare_with_comparison_details(self, mock_agent_cls: MagicMock) -> None:
        """Test compare shows comparison details when available."""
        mock_comparison = MagicMock()
        mock_comparison.overall_alignment = 0.5
        mock_comparison.summary = "Sources mostly agree"
        mock_comparison.discrepancies = []
        mock_comparison.agreements = []

        mock_agent = MagicMock()
        mock_agent.compare.return_value = _mock_result(
            response="Analysis",
            had_discrepancies=False,
            comparison=mock_comparison,
        )
        mock_agent_cls.return_value = mock_agent

        runner = CliRunner()
        result = runner.invoke(cli, ["compare", "AAPL"])
        assert result.exit_code == 0
        assert "COMPARISON DETAILS" in result.output
        assert "Alignment Score" in result.output
        assert "Sources mostly agree" in result.output

    @patch("src.orchestrator.cli.main.OrchestrationAgent")
    def test_compare_shows_discrepancies_list(self, mock_agent_cls: MagicMock) -> None:
        """Test compare shows list of discrepancies."""
        mock_discrepancy = MagicMock()
        mock_discrepancy.severity.name = "HIGH"
        mock_discrepancy.description = "News positive but SEC shows risk"

        mock_comparison = MagicMock()
        mock_comparison.overall_alignment = -0.3
        mock_comparison.summary = "Sources disagree"
        mock_comparison.discrepancies = [mock_discrepancy]
        mock_comparison.agreements = []

        mock_agent = MagicMock()
        mock_agent.compare.return_value = _mock_result(
            response="Analysis",
            had_discrepancies=True,
            comparison=mock_comparison,
        )
        mock_agent_cls.return_value = mock_agent

        runner = CliRunner()
        result = runner.invoke(cli, ["compare", "AAPL"])
        assert result.exit_code == 0
        assert "DISCREPANCIES" in result.output
        assert "News positive but SEC shows risk" in result.output

    @patch("src.orchestrator.cli.main.OrchestrationAgent")
    def test_compare_shows_agreements_list(self, mock_agent_cls: MagicMock) -> None:
        """Test compare shows list of agreements."""
        mock_agreement = MagicMock()
        mock_agreement.description = "Both sources agree on growth outlook"

        mock_comparison = MagicMock()
        mock_comparison.overall_alignment = 0.8
        mock_comparison.summary = "Sources aligned"
        mock_comparison.discrepancies = []
        mock_comparison.agreements = [mock_agreement]

        mock_agent = MagicMock()
        mock_agent.compare.return_value = _mock_result(
            response="Analysis",
            had_discrepancies=False,
            comparison=mock_comparison,
        )
        mock_agent_cls.return_value = mock_agent

        runner = CliRunner()
        result = runner.invoke(cli, ["compare", "AAPL"])
        assert result.exit_code == 0
        assert "AGREEMENTS" in result.output
        assert "Both sources agree on growth outlook" in result.output

    @patch("src.orchestrator.cli.main.OrchestrationAgent")
    def test_compare_error(self, mock_agent_cls: MagicMock) -> None:
        """Test compare handles errors."""
        mock_agent = MagicMock()
        mock_agent.compare.side_effect = RuntimeError("Failed to fetch data")
        mock_agent_cls.return_value = mock_agent

        runner = CliRunner()
        result = runner.invoke(cli, ["compare", "AAPL"])
        assert result.exit_code == 1
        assert "Error: Failed to fetch data" in result.output


class TestCliGroup:
    """Tests for the CLI group."""

    def test_help(self) -> None:
        """Test CLI help shows overview."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Orchestration Agent" in result.output
        assert "news" in result.output.lower()
        assert "sec" in result.output.lower()

    def test_subcommand_help(self) -> None:
        """Test subcommand help works."""
        runner = CliRunner()
        for cmd in ["chat", "query", "compare"]:
            result = runner.invoke(cli, [cmd, "--help"])
            assert result.exit_code == 0

    def test_chat_help_content(self) -> None:
        """Test chat help has meaningful content."""
        runner = CliRunner()
        result = runner.invoke(cli, ["chat", "--help"])
        assert result.exit_code == 0
        assert "interactive" in result.output.lower()

    def test_query_help_content(self) -> None:
        """Test query help has meaningful content."""
        runner = CliRunner()
        result = runner.invoke(cli, ["query", "--help"])
        assert result.exit_code == 0
        assert "--ticker" in result.output
        assert "--source" in result.output
        assert "--json-output" in result.output

    def test_compare_help_content(self) -> None:
        """Test compare help has meaningful content."""
        runner = CliRunner()
        result = runner.invoke(cli, ["compare", "--help"])
        assert result.exit_code == 0
        assert "TICKER" in result.output
        assert "--json-output" in result.output


class TestOutputFormatting:
    """Tests for output formatting functions."""

    @patch("src.orchestrator.cli.main.OrchestrationAgent")
    def test_high_confidence_green(self, mock_agent_cls: MagicMock) -> None:
        """Test high confidence shows green color."""
        mock_agent = MagicMock()
        mock_agent.query.return_value = _mock_result(confidence=0.9)
        mock_agent_cls.return_value = mock_agent

        runner = CliRunner()
        result = runner.invoke(cli, ["query", "test"])
        assert result.exit_code == 0
        assert "90%" in result.output

    @patch("src.orchestrator.cli.main.OrchestrationAgent")
    def test_medium_confidence_yellow(self, mock_agent_cls: MagicMock) -> None:
        """Test medium confidence formatted correctly."""
        mock_agent = MagicMock()
        mock_agent.query.return_value = _mock_result(confidence=0.5)
        mock_agent_cls.return_value = mock_agent

        runner = CliRunner()
        result = runner.invoke(cli, ["query", "test"])
        assert result.exit_code == 0
        assert "50%" in result.output

    @patch("src.orchestrator.cli.main.OrchestrationAgent")
    def test_low_confidence_red(self, mock_agent_cls: MagicMock) -> None:
        """Test low confidence formatted correctly."""
        mock_agent = MagicMock()
        mock_agent.query.return_value = _mock_result(confidence=0.2)
        mock_agent_cls.return_value = mock_agent

        runner = CliRunner()
        result = runner.invoke(cli, ["query", "test"])
        assert result.exit_code == 0
        assert "20%" in result.output

    @patch("src.orchestrator.cli.main.OrchestrationAgent")
    def test_execution_time_displayed(self, mock_agent_cls: MagicMock) -> None:
        """Test execution time is displayed."""
        mock_agent = MagicMock()
        mock_agent.query.return_value = _mock_result(execution_time_ms=3456.78)
        mock_agent_cls.return_value = mock_agent

        runner = CliRunner()
        result = runner.invoke(cli, ["query", "test"])
        assert result.exit_code == 0
        assert "3457ms" in result.output


class TestEdgeCases:
    """Tests for edge cases."""

    @patch("src.orchestrator.cli.main.OrchestrationAgent")
    def test_empty_agents_used(self, mock_agent_cls: MagicMock) -> None:
        """Test handling of empty agents list."""
        mock_agent = MagicMock()
        # Create mock result with empty agents list
        mock_result = MagicMock()
        mock_result.response = "Test response"
        mock_result.route_type = "both"
        mock_result.agents_used = []  # Explicitly empty
        mock_result.had_discrepancies = False
        mock_result.confidence = 0.8
        mock_result.execution_time_ms = 1500.0
        mock_result.query_id = "test-query-id"
        mock_result.comparison = None
        mock_agent.query.return_value = mock_result
        mock_agent_cls.return_value = mock_agent

        runner = CliRunner()
        result = runner.invoke(cli, ["query", "test"])
        assert result.exit_code == 0
        assert "none" in result.output

    @patch("src.orchestrator.cli.main.OrchestrationAgent")
    def test_history_empty(self, mock_agent_cls: MagicMock) -> None:
        """Test history with no queries."""
        mock_agent = MagicMock()
        mock_agent.get_recent_queries.return_value = []
        mock_agent_cls.return_value = mock_agent

        runner = CliRunner()
        result = runner.invoke(cli, ["chat"], input="history\nexit\n")
        assert result.exit_code == 0
        assert "No queries in this session" in result.output

    @patch("src.orchestrator.cli.main.OrchestrationAgent")
    def test_summary_empty_tickers(self, mock_agent_cls: MagicMock) -> None:
        """Test summary with no tickers analyzed."""
        mock_agent = MagicMock()
        mock_agent.get_session_summary.return_value = {
            "session_id": "test",
            "query_count": 0,
            "tickers_analyzed": [],
            "discrepancy_count": 0,
            "average_confidence": 0,
        }
        mock_agent_cls.return_value = mock_agent

        runner = CliRunner()
        result = runner.invoke(cli, ["chat"], input="summary\nexit\n")
        assert result.exit_code == 0
        assert "none" in result.output

    @patch("src.orchestrator.cli.main.OrchestrationAgent")
    def test_long_query_truncated_in_history(self, mock_agent_cls: MagicMock) -> None:
        """Test long queries are truncated in history display."""
        mock_agent = MagicMock()
        long_query = "A" * 100
        mock_agent.get_recent_queries.return_value = [
            {"ticker": "AAPL", "user_query": long_query, "confidence": 0.8, "had_discrepancies": False},
        ]
        mock_agent_cls.return_value = mock_agent

        runner = CliRunner()
        result = runner.invoke(cli, ["chat"], input="history\nexit\n")
        assert result.exit_code == 0
        assert "..." in result.output
        # Should be truncated to 50 chars
        assert "A" * 50 in result.output
