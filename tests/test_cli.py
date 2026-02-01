"""Tests for the CLI interface."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from sec_agent.cli.main import cli


class TestChatCommand:
    @patch("sec_agent.cli.main.SECFilingsAgent")
    def test_chat_exit(self, mock_agent_cls: MagicMock) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["chat"], input="exit\n")
        assert result.exit_code == 0
        assert "interactive mode" in result.output
        assert "Goodbye" in result.output

    @patch("sec_agent.cli.main.SECFilingsAgent")
    def test_chat_quit(self, mock_agent_cls: MagicMock) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["chat"], input="quit\n")
        assert result.exit_code == 0
        assert "Goodbye" in result.output

    @patch("sec_agent.cli.main.SECFilingsAgent")
    def test_chat_sends_query(self, mock_agent_cls: MagicMock) -> None:
        mock_agent = MagicMock()
        mock_agent.query.return_value = "Apple has supply chain risks."
        mock_agent_cls.return_value = mock_agent

        runner = CliRunner()
        result = runner.invoke(cli, ["chat"], input="What are Apple's risks?\nexit\n")
        assert result.exit_code == 0
        mock_agent.query.assert_called_once_with("What are Apple's risks?")
        assert "Apple has supply chain risks." in result.output

    @patch("sec_agent.cli.main.SECFilingsAgent")
    def test_chat_handles_error(self, mock_agent_cls: MagicMock) -> None:
        mock_agent = MagicMock()
        mock_agent.query.side_effect = RuntimeError("Connection failed")
        mock_agent_cls.return_value = mock_agent

        runner = CliRunner()
        result = runner.invoke(cli, ["chat"], input="test\nexit\n")
        assert "Error: Connection failed" in result.output

    @patch("sec_agent.cli.main.SECFilingsAgent")
    def test_chat_skips_empty_input(self, mock_agent_cls: MagicMock) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["chat"], input="\nexit\n")
        assert result.exit_code == 0
        assert "Goodbye" in result.output


class TestQueryCommand:
    @patch("sec_agent.cli.main.SECFilingsAgent")
    def test_query_basic(self, mock_agent_cls: MagicMock) -> None:
        mock_agent = MagicMock()
        mock_agent.query.return_value = "Risk factors include..."
        mock_agent_cls.return_value = mock_agent

        runner = CliRunner()
        result = runner.invoke(
            cli, ["query", "What are the risks?", "--ticker", "AAPL", "--filing", "10-K"]
        )
        assert result.exit_code == 0
        assert "Risk factors include..." in result.output
        call_arg = mock_agent.query.call_args[0][0]
        assert "AAPL" in call_arg
        assert "10-K" in call_arg

    @patch("sec_agent.cli.main.SECFilingsAgent")
    def test_query_with_section_filter(self, mock_agent_cls: MagicMock) -> None:
        mock_agent = MagicMock()
        mock_agent.query.return_value = "Answer here."
        mock_agent_cls.return_value = mock_agent

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["query", "risks?", "-t", "AAPL", "-f", "10-K", "-s", "1A"],
        )
        assert result.exit_code == 0
        call_arg = mock_agent.query.call_args[0][0]
        assert "section 1A" in call_arg

    @patch("sec_agent.cli.main.SECFilingsAgent")
    def test_query_error(self, mock_agent_cls: MagicMock) -> None:
        mock_agent = MagicMock()
        mock_agent.query.side_effect = RuntimeError("fail")
        mock_agent_cls.return_value = mock_agent

        runner = CliRunner()
        result = runner.invoke(
            cli, ["query", "test", "--ticker", "AAPL", "--filing", "10-K"]
        )
        assert result.exit_code == 1

    def test_query_missing_ticker(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["query", "test", "--filing", "10-K"])
        assert result.exit_code != 0

    def test_query_invalid_filing_type(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["query", "test", "--ticker", "AAPL", "--filing", "13-F"])
        assert result.exit_code != 0


class TestFetchCommand:
    @patch("sec_agent.cli.main.fetch_and_parse_filing")
    def test_fetch_success(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = {
            "status": "ok",
            "source": "fetched",
            "filing_date": "2024-11-03",
            "sections_found": ["1A", "7"],
            "chunk_count": 42,
        }

        runner = CliRunner()
        result = runner.invoke(cli, ["fetch", "AAPL", "10-K"])
        assert result.exit_code == 0
        assert "fetched" in result.output
        assert "2024-11-03" in result.output
        assert "1A" in result.output
        assert "42" in result.output

    @patch("sec_agent.cli.main.fetch_and_parse_filing")
    def test_fetch_error_result(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = {"status": "error", "message": "Not found"}

        runner = CliRunner()
        result = runner.invoke(cli, ["fetch", "XYZ", "10-K"])
        assert result.exit_code == 1

    @patch("sec_agent.cli.main.fetch_and_parse_filing")
    def test_fetch_exception(self, mock_fetch: MagicMock) -> None:
        mock_fetch.side_effect = RuntimeError("Network error")

        runner = CliRunner()
        result = runner.invoke(cli, ["fetch", "AAPL", "10-K"])
        assert result.exit_code == 1

    @patch("sec_agent.cli.main.fetch_and_parse_filing")
    def test_fetch_with_index(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = {
            "status": "ok",
            "source": "fetched",
            "filing_date": "2023-11-03",
            "sections_found": [],
            "chunk_count": 0,
        }

        runner = CliRunner()
        result = runner.invoke(cli, ["fetch", "AAPL", "10-K", "--index", "1"])
        assert result.exit_code == 0
        mock_fetch.assert_called_once_with(ticker="AAPL", filing_type="10-K", filing_index=1)


class TestListFilingsCommand:
    @patch("sec_agent.cli.main.list_available_filings")
    def test_list_with_filings(self, mock_list: MagicMock) -> None:
        mock_list.return_value = {
            "status": "ok",
            "ticker": "AAPL",
            "cached_filings": [
                "filings/AAPL/10-K/000-123/chunks.json",
                "filings/AAPL/10-Q/000-456/chunks.json",
            ],
            "count": 2,
        }

        runner = CliRunner()
        result = runner.invoke(cli, ["list-filings", "AAPL"])
        assert result.exit_code == 0
        assert "AAPL" in result.output
        assert "000-123" in result.output
        assert "000-456" in result.output

    @patch("sec_agent.cli.main.list_available_filings")
    def test_list_empty(self, mock_list: MagicMock) -> None:
        mock_list.return_value = {
            "status": "ok",
            "ticker": "XYZ",
            "cached_filings": [],
            "count": 0,
        }

        runner = CliRunner()
        result = runner.invoke(cli, ["list-filings", "XYZ"])
        assert result.exit_code == 0
        assert "No cached filings" in result.output

    @patch("sec_agent.cli.main.list_available_filings")
    def test_list_error(self, mock_list: MagicMock) -> None:
        mock_list.side_effect = RuntimeError("S3 error")

        runner = CliRunner()
        result = runner.invoke(cli, ["list-filings", "AAPL"])
        assert result.exit_code == 1


class TestCliGroup:
    def test_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "SEC filings" in result.output

    def test_subcommand_help(self) -> None:
        runner = CliRunner()
        for cmd in ["chat", "query", "fetch", "list-filings"]:
            result = runner.invoke(cli, [cmd, "--help"])
            assert result.exit_code == 0
