"""Tests for the ResponseSynthesizer."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import Mock, patch

import pytest

from orchestrator.comparison.discrepancy import (
    Agreement,
    ComparisonResult,
    Discrepancy,
    DiscrepancySeverity,
    DiscrepancyType,
)
from orchestrator.comparison.signals import (
    ExtractedSignal,
    SignalDirection,
    SignalExtractionResult,
    SignalType,
)
from orchestrator.execution.result import AgentResult, AgentStatus
from orchestrator.synthesis.synthesizer import (
    ResponseSynthesizer,
    SynthesisInput,
    SynthesisResult,
)


class MockBedrockClient:
    """Mock Bedrock client for testing."""

    def __init__(self, response_text: str = "Synthesized response") -> None:
        self.response_text = response_text
        self.last_call: dict[str, Any] | None = None
        self.call_count = 0

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.last_call = kwargs
        self.call_count += 1
        return {
            "output": {
                "message": {
                    "content": [{"text": self.response_text}]
                }
            }
        }


# ---------------------------------------------------------------------------
# SynthesisInput tests
# ---------------------------------------------------------------------------


class TestSynthesisInput:
    """Tests for SynthesisInput dataclass."""

    def test_minimal_input(self) -> None:
        """Can create input with just a query."""
        input_data = SynthesisInput(user_query="What's happening with AAPL?")
        assert input_data.user_query == "What's happening with AAPL?"
        assert input_data.news_result is None
        assert input_data.sec_result is None
        assert input_data.comparison is None
        assert input_data.ticker is None

    def test_full_input(self) -> None:
        """Can create input with all fields."""
        news_result = AgentResult(
            agent_name="news_agent",
            status=AgentStatus.SUCCESS,
            response="News analysis",
        )
        sec_result = AgentResult(
            agent_name="sec_agent",
            status=AgentStatus.SUCCESS,
            response="SEC analysis",
        )
        comparison = ComparisonResult(
            news_result=None,
            sec_result=None,
            summary="Sources agree",
        )

        input_data = SynthesisInput(
            user_query="Compare AAPL news and filings",
            news_result=news_result,
            sec_result=sec_result,
            comparison=comparison,
            ticker="AAPL",
        )

        assert input_data.user_query == "Compare AAPL news and filings"
        assert input_data.news_result is news_result
        assert input_data.sec_result is sec_result
        assert input_data.comparison is comparison
        assert input_data.ticker == "AAPL"


# ---------------------------------------------------------------------------
# SynthesisResult tests
# ---------------------------------------------------------------------------


class TestSynthesisResult:
    """Tests for SynthesisResult dataclass."""

    def test_minimal_result(self) -> None:
        """Can create result with just response."""
        result = SynthesisResult(response="Analysis complete")
        assert result.response == "Analysis complete"
        assert result.confidence == 0.8
        assert result.sources_used == []
        assert result.had_discrepancies is False
        assert result.model_id == ""

    def test_full_result(self) -> None:
        """Can create result with all fields."""
        result = SynthesisResult(
            response="Analysis complete",
            confidence=0.95,
            sources_used=["news_agent", "sec_agent"],
            had_discrepancies=True,
            model_id="anthropic.claude-opus-4-5-20251101-v1:0",
        )
        assert result.response == "Analysis complete"
        assert result.confidence == 0.95
        assert result.sources_used == ["news_agent", "sec_agent"]
        assert result.had_discrepancies is True
        assert result.model_id == "anthropic.claude-opus-4-5-20251101-v1:0"

    def test_sources_used_none_becomes_empty_list(self) -> None:
        """sources_used None is converted to empty list."""
        result = SynthesisResult(response="test", sources_used=None)
        assert result.sources_used == []


# ---------------------------------------------------------------------------
# ResponseSynthesizer initialization tests
# ---------------------------------------------------------------------------


class TestResponseSynthesizerInit:
    """Tests for ResponseSynthesizer initialization."""

    def test_default_init(self) -> None:
        """Can initialize with defaults (will use boto3)."""
        with patch("boto3.client") as mock_boto:
            mock_boto.return_value = Mock()
            synthesizer = ResponseSynthesizer()
            assert synthesizer._model_id == "anthropic.claude-opus-4-5-20251101-v1:0"
            mock_boto.assert_called_once_with("bedrock-runtime")

    def test_custom_model_id(self) -> None:
        """Can initialize with custom model ID."""
        mock_client = MockBedrockClient()
        synthesizer = ResponseSynthesizer(
            model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
            bedrock_client=mock_client,
        )
        assert synthesizer._model_id == "us.anthropic.claude-sonnet-4-20250514-v1:0"

    def test_env_var_model_id(self) -> None:
        """Uses BEDROCK_MODEL_ID env var if set."""
        mock_client = MockBedrockClient()
        with patch.dict("os.environ", {"BEDROCK_MODEL_ID": "env-model-id"}):
            synthesizer = ResponseSynthesizer(bedrock_client=mock_client)
            assert synthesizer._model_id == "env-model-id"


# ---------------------------------------------------------------------------
# ResponseSynthesizer.synthesize tests
# ---------------------------------------------------------------------------


class TestResponseSynthesizerSynthesize:
    """Tests for ResponseSynthesizer.synthesize method."""

    def test_synthesize_with_both_agents_success(self) -> None:
        """Synthesize response when both agents succeed."""
        mock_client = MockBedrockClient(response_text="Combined analysis shows positive outlook.")
        synthesizer = ResponseSynthesizer(bedrock_client=mock_client)

        news_result = AgentResult(
            agent_name="news_agent",
            status=AgentStatus.SUCCESS,
            response="News shows positive sentiment",
        )
        sec_result = AgentResult(
            agent_name="sec_agent",
            status=AgentStatus.SUCCESS,
            response="SEC filings show strong financials",
        )

        input_data = SynthesisInput(
            user_query="What's the outlook for AAPL?",
            news_result=news_result,
            sec_result=sec_result,
            ticker="AAPL",
        )

        result = synthesizer.synthesize(input_data)

        assert result.response == "Combined analysis shows positive outlook."
        assert result.sources_used == ["news_agent", "sec_agent"]
        assert result.confidence == 0.9
        assert result.had_discrepancies is False
        assert mock_client.call_count == 1

    def test_synthesize_with_news_only(self) -> None:
        """Synthesize response when only news agent returns."""
        mock_client = MockBedrockClient(response_text="Based on news analysis...")
        synthesizer = ResponseSynthesizer(bedrock_client=mock_client)

        news_result = AgentResult(
            agent_name="news_agent",
            status=AgentStatus.SUCCESS,
            response="News shows positive sentiment",
        )

        input_data = SynthesisInput(
            user_query="What's in the news for AAPL?",
            news_result=news_result,
            sec_result=None,
        )

        result = synthesizer.synthesize(input_data)

        assert result.response == "Based on news analysis..."
        assert result.sources_used == ["news_agent"]
        assert result.confidence == 0.7

    def test_synthesize_with_sec_only(self) -> None:
        """Synthesize response when only SEC agent returns."""
        mock_client = MockBedrockClient(response_text="Based on SEC filings...")
        synthesizer = ResponseSynthesizer(bedrock_client=mock_client)

        sec_result = AgentResult(
            agent_name="sec_agent",
            status=AgentStatus.SUCCESS,
            response="SEC filings show strong financials",
        )

        input_data = SynthesisInput(
            user_query="What do the filings say about AAPL?",
            news_result=None,
            sec_result=sec_result,
        )

        result = synthesizer.synthesize(input_data)

        assert result.response == "Based on SEC filings..."
        assert result.sources_used == ["sec_agent"]
        assert result.confidence == 0.7

    def test_synthesize_with_no_successful_agents(self) -> None:
        """Synthesize response when both agents fail."""
        mock_client = MockBedrockClient(response_text="Unable to retrieve information...")
        synthesizer = ResponseSynthesizer(bedrock_client=mock_client)

        news_result = AgentResult(
            agent_name="news_agent",
            status=AgentStatus.ERROR,
            error_message="API error",
        )
        sec_result = AgentResult(
            agent_name="sec_agent",
            status=AgentStatus.TIMEOUT,
            error_message="Request timed out",
        )

        input_data = SynthesisInput(
            user_query="What's happening with AAPL?",
            news_result=news_result,
            sec_result=sec_result,
        )

        result = synthesizer.synthesize(input_data)

        assert result.response == "Unable to retrieve information..."
        assert result.sources_used == []
        assert result.confidence == 0.3

    def test_synthesize_with_discrepancies(self) -> None:
        """Synthesize response with comparison discrepancies."""
        mock_client = MockBedrockClient(response_text="Sources show conflicting views...")
        synthesizer = ResponseSynthesizer(bedrock_client=mock_client)

        news_result = AgentResult(
            agent_name="news_agent",
            status=AgentStatus.SUCCESS,
            response="News is very positive",
        )
        sec_result = AgentResult(
            agent_name="sec_agent",
            status=AgentStatus.SUCCESS,
            response="SEC shows concerning risks",
        )

        signal = ExtractedSignal(
            signal_type=SignalType.SENTIMENT,
            topic="outlook",
            direction=SignalDirection.POSITIVE,
            description="Positive outlook",
            confidence=0.8,
        )

        discrepancy = Discrepancy(
            discrepancy_type=DiscrepancyType.SENTIMENT_CONFLICT,
            severity=DiscrepancySeverity.HIGH,
            topic="outlook",
            news_signal=signal,
            sec_signal=signal,
            description="News positive, SEC negative",
        )

        comparison = ComparisonResult(
            news_result=None,
            sec_result=None,
            discrepancies=[discrepancy],
            summary="Found conflict",
            has_critical_discrepancies=True,
        )

        input_data = SynthesisInput(
            user_query="What's the outlook for AAPL?",
            news_result=news_result,
            sec_result=sec_result,
            comparison=comparison,
        )

        result = synthesizer.synthesize(input_data)

        assert result.had_discrepancies is True
        # Confidence reduced due to critical discrepancy
        assert result.confidence < 0.9

    def test_synthesize_with_agreements(self) -> None:
        """Synthesize response with comparison agreements."""
        mock_client = MockBedrockClient(response_text="Both sources confirm positive outlook...")
        synthesizer = ResponseSynthesizer(bedrock_client=mock_client)

        news_result = AgentResult(
            agent_name="news_agent",
            status=AgentStatus.SUCCESS,
            response="News shows positive sentiment",
        )
        sec_result = AgentResult(
            agent_name="sec_agent",
            status=AgentStatus.SUCCESS,
            response="SEC shows strong growth",
        )

        news_signal = ExtractedSignal(
            signal_type=SignalType.SENTIMENT,
            topic="outlook",
            direction=SignalDirection.POSITIVE,
            description="Positive outlook from news",
            confidence=0.9,
        )
        sec_signal = ExtractedSignal(
            signal_type=SignalType.SENTIMENT,
            topic="outlook",
            direction=SignalDirection.POSITIVE,
            description="Positive outlook from SEC",
            confidence=0.9,
        )

        agreement = Agreement(
            topic="outlook",
            direction=SignalDirection.POSITIVE,
            news_signal=news_signal,
            sec_signal=sec_signal,
            description="Both sources show positive outlook",
            confidence=0.9,
        )

        comparison = ComparisonResult(
            news_result=None,
            sec_result=None,
            agreements=[agreement],
            overall_alignment=0.8,
            summary="Sources agree",
        )

        input_data = SynthesisInput(
            user_query="What's the outlook for AAPL?",
            news_result=news_result,
            sec_result=sec_result,
            comparison=comparison,
        )

        result = synthesizer.synthesize(input_data)

        assert result.had_discrepancies is False
        # Positive alignment increases confidence
        assert result.confidence > 0.9


# ---------------------------------------------------------------------------
# _build_user_message tests
# ---------------------------------------------------------------------------


class TestBuildUserMessage:
    """Tests for ResponseSynthesizer._build_user_message."""

    def test_build_message_with_query_only(self) -> None:
        """Build message with just a query."""
        mock_client = MockBedrockClient()
        synthesizer = ResponseSynthesizer(bedrock_client=mock_client)

        input_data = SynthesisInput(user_query="What's happening?")
        message = synthesizer._build_user_message(input_data)

        assert "USER QUERY: What's happening?" in message
        assert "NEWS AGENT RESPONSE" in message
        assert "Not requested for this query." in message
        assert "SEC AGENT RESPONSE" in message

    def test_build_message_with_ticker(self) -> None:
        """Build message includes ticker."""
        mock_client = MockBedrockClient()
        synthesizer = ResponseSynthesizer(bedrock_client=mock_client)

        input_data = SynthesisInput(user_query="What's happening?", ticker="AAPL")
        message = synthesizer._build_user_message(input_data)

        assert "TICKER: AAPL" in message

    def test_build_message_with_successful_agents(self) -> None:
        """Build message with successful agent responses."""
        mock_client = MockBedrockClient()
        synthesizer = ResponseSynthesizer(bedrock_client=mock_client)

        news_result = AgentResult(
            agent_name="news_agent",
            status=AgentStatus.SUCCESS,
            response="News analysis here",
        )
        sec_result = AgentResult(
            agent_name="sec_agent",
            status=AgentStatus.SUCCESS,
            response="SEC analysis here",
        )

        input_data = SynthesisInput(
            user_query="What's happening?",
            news_result=news_result,
            sec_result=sec_result,
        )
        message = synthesizer._build_user_message(input_data)

        assert "News analysis here" in message
        assert "SEC analysis here" in message

    def test_build_message_with_timeout(self) -> None:
        """Build message shows timeout status."""
        mock_client = MockBedrockClient()
        synthesizer = ResponseSynthesizer(bedrock_client=mock_client)

        news_result = AgentResult(
            agent_name="news_agent",
            status=AgentStatus.TIMEOUT,
            error_message="Request timed out after 60s",
        )

        input_data = SynthesisInput(
            user_query="What's happening?",
            news_result=news_result,
        )
        message = synthesizer._build_user_message(input_data)

        assert "TIMEOUT:" in message
        assert "Request timed out after 60s" in message

    def test_build_message_with_error(self) -> None:
        """Build message shows error status."""
        mock_client = MockBedrockClient()
        synthesizer = ResponseSynthesizer(bedrock_client=mock_client)

        sec_result = AgentResult(
            agent_name="sec_agent",
            status=AgentStatus.ERROR,
            error_message="Connection failed",
        )

        input_data = SynthesisInput(
            user_query="What's happening?",
            sec_result=sec_result,
        )
        message = synthesizer._build_user_message(input_data)

        assert "ERROR:" in message
        assert "Connection failed" in message

    def test_build_message_with_comparison(self) -> None:
        """Build message includes comparison details."""
        mock_client = MockBedrockClient()
        synthesizer = ResponseSynthesizer(bedrock_client=mock_client)

        news_signal = ExtractedSignal(
            signal_type=SignalType.SENTIMENT,
            topic="outlook",
            direction=SignalDirection.POSITIVE,
            description="Positive outlook from news",
            confidence=0.8,
        )
        sec_signal = ExtractedSignal(
            signal_type=SignalType.SENTIMENT,
            topic="outlook",
            direction=SignalDirection.NEGATIVE,
            description="Negative outlook from SEC",
            confidence=0.8,
        )

        discrepancy = Discrepancy(
            discrepancy_type=DiscrepancyType.SENTIMENT_CONFLICT,
            severity=DiscrepancySeverity.HIGH,
            topic="outlook",
            news_signal=news_signal,
            sec_signal=sec_signal,
            description="News positive, SEC negative",
        )

        agreement = Agreement(
            topic="revenue",
            direction=SignalDirection.POSITIVE,
            news_signal=news_signal,
            sec_signal=sec_signal,
            description="Both agree on revenue growth",
        )

        comparison = ComparisonResult(
            news_result=None,
            sec_result=None,
            discrepancies=[discrepancy],
            agreements=[agreement],
            overall_alignment=-0.3,
            summary="Mixed signals between sources",
        )

        input_data = SynthesisInput(
            user_query="Compare AAPL?",
            comparison=comparison,
        )
        message = synthesizer._build_user_message(input_data)

        assert "SOURCE COMPARISON" in message
        assert "Mixed signals between sources" in message
        assert "Overall alignment: -0.3" in message
        assert "DISCREPANCIES:" in message
        assert "[HIGH]" in message
        assert "News positive, SEC negative" in message
        assert "AGREEMENTS:" in message
        assert "Both agree on revenue growth" in message


# ---------------------------------------------------------------------------
# _calculate_confidence tests
# ---------------------------------------------------------------------------


class TestCalculateConfidence:
    """Tests for ResponseSynthesizer._calculate_confidence."""

    def test_confidence_both_success(self) -> None:
        """High confidence when both agents succeed."""
        mock_client = MockBedrockClient()
        synthesizer = ResponseSynthesizer(bedrock_client=mock_client)

        news_result = AgentResult(
            agent_name="news_agent",
            status=AgentStatus.SUCCESS,
            response="News",
        )
        sec_result = AgentResult(
            agent_name="sec_agent",
            status=AgentStatus.SUCCESS,
            response="SEC",
        )

        input_data = SynthesisInput(
            user_query="test",
            news_result=news_result,
            sec_result=sec_result,
        )

        confidence = synthesizer._calculate_confidence(input_data)
        assert confidence == 0.9

    def test_confidence_one_success(self) -> None:
        """Medium confidence when one agent succeeds."""
        mock_client = MockBedrockClient()
        synthesizer = ResponseSynthesizer(bedrock_client=mock_client)

        news_result = AgentResult(
            agent_name="news_agent",
            status=AgentStatus.SUCCESS,
            response="News",
        )

        input_data = SynthesisInput(user_query="test", news_result=news_result)

        confidence = synthesizer._calculate_confidence(input_data)
        assert confidence == 0.7

    def test_confidence_no_success(self) -> None:
        """Low confidence when no agents succeed."""
        mock_client = MockBedrockClient()
        synthesizer = ResponseSynthesizer(bedrock_client=mock_client)

        news_result = AgentResult(
            agent_name="news_agent",
            status=AgentStatus.ERROR,
            error_message="Error",
        )

        input_data = SynthesisInput(user_query="test", news_result=news_result)

        confidence = synthesizer._calculate_confidence(input_data)
        assert confidence == 0.3

    def test_confidence_positive_alignment_boost(self) -> None:
        """Positive alignment increases confidence."""
        mock_client = MockBedrockClient()
        synthesizer = ResponseSynthesizer(bedrock_client=mock_client)

        news_result = AgentResult(
            agent_name="news_agent",
            status=AgentStatus.SUCCESS,
            response="News",
        )
        sec_result = AgentResult(
            agent_name="sec_agent",
            status=AgentStatus.SUCCESS,
            response="SEC",
        )

        comparison = ComparisonResult(
            news_result=None,
            sec_result=None,
            overall_alignment=1.0,
        )

        input_data = SynthesisInput(
            user_query="test",
            news_result=news_result,
            sec_result=sec_result,
            comparison=comparison,
        )

        confidence = synthesizer._calculate_confidence(input_data)
        assert confidence == 1.0  # 0.9 + 1.0 * 0.1 = 1.0

    def test_confidence_negative_alignment_penalty(self) -> None:
        """Negative alignment decreases confidence."""
        mock_client = MockBedrockClient()
        synthesizer = ResponseSynthesizer(bedrock_client=mock_client)

        news_result = AgentResult(
            agent_name="news_agent",
            status=AgentStatus.SUCCESS,
            response="News",
        )
        sec_result = AgentResult(
            agent_name="sec_agent",
            status=AgentStatus.SUCCESS,
            response="SEC",
        )

        comparison = ComparisonResult(
            news_result=None,
            sec_result=None,
            overall_alignment=-1.0,
        )

        input_data = SynthesisInput(
            user_query="test",
            news_result=news_result,
            sec_result=sec_result,
            comparison=comparison,
        )

        confidence = synthesizer._calculate_confidence(input_data)
        assert confidence == 0.7  # 0.9 + (-1.0) * 0.2 = 0.7

    def test_confidence_critical_discrepancy_penalty(self) -> None:
        """Critical discrepancies reduce confidence."""
        mock_client = MockBedrockClient()
        synthesizer = ResponseSynthesizer(bedrock_client=mock_client)

        news_result = AgentResult(
            agent_name="news_agent",
            status=AgentStatus.SUCCESS,
            response="News",
        )
        sec_result = AgentResult(
            agent_name="sec_agent",
            status=AgentStatus.SUCCESS,
            response="SEC",
        )

        comparison = ComparisonResult(
            news_result=None,
            sec_result=None,
            overall_alignment=0.0,
            has_critical_discrepancies=True,
        )

        input_data = SynthesisInput(
            user_query="test",
            news_result=news_result,
            sec_result=sec_result,
            comparison=comparison,
        )

        confidence = synthesizer._calculate_confidence(input_data)
        assert confidence == 0.75  # 0.9 - 0.15 = 0.75

    def test_confidence_minimum_bound(self) -> None:
        """Confidence has a minimum bound of 0.1."""
        mock_client = MockBedrockClient()
        synthesizer = ResponseSynthesizer(bedrock_client=mock_client)

        comparison = ComparisonResult(
            news_result=None,
            sec_result=None,
            overall_alignment=-1.0,
            has_critical_discrepancies=True,
        )

        input_data = SynthesisInput(
            user_query="test",
            comparison=comparison,
        )

        confidence = synthesizer._calculate_confidence(input_data)
        # Base 0.3 - 0.2 (alignment) - 0.15 (critical) would be -0.05, bounded to 0.1
        assert confidence == 0.1


# ---------------------------------------------------------------------------
# Bedrock API call verification
# ---------------------------------------------------------------------------


class TestBedrockApiCall:
    """Tests verifying correct Bedrock API calls."""

    def test_converse_called_with_correct_params(self) -> None:
        """Verify converse is called with correct parameters."""
        mock_client = MockBedrockClient()
        synthesizer = ResponseSynthesizer(
            model_id="test-model",
            bedrock_client=mock_client,
        )

        input_data = SynthesisInput(user_query="Test query")
        synthesizer.synthesize(input_data)

        assert mock_client.call_count == 1
        assert mock_client.last_call is not None
        assert mock_client.last_call["modelId"] == "test-model"
        assert "system" in mock_client.last_call
        assert "messages" in mock_client.last_call
        assert len(mock_client.last_call["messages"]) == 1
        assert mock_client.last_call["messages"][0]["role"] == "user"

    def test_system_prompt_included(self) -> None:
        """Verify system prompt is included in API call."""
        mock_client = MockBedrockClient()
        synthesizer = ResponseSynthesizer(bedrock_client=mock_client)

        input_data = SynthesisInput(user_query="Test query")
        synthesizer.synthesize(input_data)

        system = mock_client.last_call["system"]
        assert len(system) == 1
        assert "financial analyst" in system[0]["text"].lower()

    def test_model_id_in_result(self) -> None:
        """Verify model_id is included in result."""
        mock_client = MockBedrockClient()
        synthesizer = ResponseSynthesizer(
            model_id="my-model-id",
            bedrock_client=mock_client,
        )

        input_data = SynthesisInput(user_query="Test query")
        result = synthesizer.synthesize(input_data)

        assert result.model_id == "my-model-id"


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_response_from_agent(self) -> None:
        """Handle agent with empty response."""
        mock_client = MockBedrockClient(response_text="No data available")
        synthesizer = ResponseSynthesizer(bedrock_client=mock_client)

        news_result = AgentResult(
            agent_name="news_agent",
            status=AgentStatus.SUCCESS,
            response="",
        )

        input_data = SynthesisInput(user_query="Test", news_result=news_result)
        message = synthesizer._build_user_message(input_data)

        assert "(empty response)" in message

    def test_none_response_from_agent(self) -> None:
        """Handle agent with None response."""
        mock_client = MockBedrockClient(response_text="No data available")
        synthesizer = ResponseSynthesizer(bedrock_client=mock_client)

        news_result = AgentResult(
            agent_name="news_agent",
            status=AgentStatus.SUCCESS,
            response=None,
        )

        input_data = SynthesisInput(user_query="Test", news_result=news_result)
        message = synthesizer._build_user_message(input_data)

        assert "(empty response)" in message

    def test_none_error_message(self) -> None:
        """Handle agent with None error message."""
        mock_client = MockBedrockClient()
        synthesizer = ResponseSynthesizer(bedrock_client=mock_client)

        news_result = AgentResult(
            agent_name="news_agent",
            status=AgentStatus.ERROR,
            error_message=None,
        )

        input_data = SynthesisInput(user_query="Test", news_result=news_result)
        message = synthesizer._build_user_message(input_data)

        assert "ERROR: Unknown error" in message

    def test_timeout_without_error_message(self) -> None:
        """Handle timeout without error message."""
        mock_client = MockBedrockClient()
        synthesizer = ResponseSynthesizer(bedrock_client=mock_client)

        sec_result = AgentResult(
            agent_name="sec_agent",
            status=AgentStatus.TIMEOUT,
            error_message=None,
        )

        input_data = SynthesisInput(user_query="Test", sec_result=sec_result)
        message = synthesizer._build_user_message(input_data)

        assert "TIMEOUT: Request timed out" in message

    def test_multiline_response(self) -> None:
        """Handle multiline response from Bedrock."""
        mock_client = MockBedrockClient()
        mock_client.response_text = "Line 1\nLine 2\nLine 3"
        synthesizer = ResponseSynthesizer(bedrock_client=mock_client)

        input_data = SynthesisInput(user_query="Test")
        result = synthesizer.synthesize(input_data)

        assert result.response == "Line 1\nLine 2\nLine 3"

    def test_multiple_text_blocks_in_response(self) -> None:
        """Handle multiple text blocks in Bedrock response."""
        mock_client = Mock()
        mock_client.converse.return_value = {
            "output": {
                "message": {
                    "content": [
                        {"text": "First part"},
                        {"text": "Second part"},
                    ]
                }
            }
        }

        synthesizer = ResponseSynthesizer(bedrock_client=mock_client)
        input_data = SynthesisInput(user_query="Test")
        result = synthesizer.synthesize(input_data)

        assert result.response == "First part\nSecond part"
