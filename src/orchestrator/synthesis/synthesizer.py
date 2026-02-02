"""LLM-based response synthesis for orchestrator.

This module uses Bedrock Claude to synthesize a unified response from
news_agent and sec_agent outputs, highlighting agreements and discrepancies.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import boto3

from orchestrator.comparison.discrepancy import ComparisonResult
from orchestrator.execution.result import AgentResult

SYNTHESIS_SYSTEM_PROMPT = """You are a financial analyst synthesizing information from two \
sources: news sentiment analysis and SEC filings analysis. Your job is to provide a unified, \
coherent response to the user's query.

Guidelines:
1. Integrate insights from both sources where available
2. Clearly highlight agreements between sources (increases confidence)
3. Explicitly note discrepancies with citations to each source
4. Indicate confidence level based on source agreement
5. Prioritize factual information from SEC filings over news sentiment when they conflict
6. Be concise but thorough - focus on what matters for the user's query

When presenting information:
- Use "News reports..." or "According to news sources..." for news-based information
- Use "SEC filings indicate..." or "Per the company's filings..." for SEC-based information
- Use "Both sources agree..." when there's alignment
- Use "However, there's a discrepancy..." when sources conflict

If one source is unavailable, clearly state this and base your response on the available source.
If both sources had errors, explain what information is missing and provide what context you can."""


@dataclass
class SynthesisInput:
    """Input data for response synthesis.

    Attributes:
        user_query: The original user query.
        news_result: Result from the news_agent (or None if not called/failed).
        sec_result: Result from the sec_agent (or None if not called/failed).
        comparison: Comparison result if both agents returned data.
        ticker: Optional ticker symbol for context.
    """

    user_query: str
    news_result: AgentResult | None = None
    sec_result: AgentResult | None = None
    comparison: ComparisonResult | None = None
    ticker: str | None = None


@dataclass
class SynthesisResult:
    """Result of response synthesis.

    Attributes:
        response: The synthesized response text.
        confidence: Overall confidence in the response (0.0 to 1.0).
        sources_used: List of sources used in the response.
        had_discrepancies: Whether discrepancies were found between sources.
        model_id: The model ID used for synthesis.
    """

    response: str
    confidence: float = 0.8
    sources_used: list[str] | None = None
    had_discrepancies: bool = False
    model_id: str = ""

    def __post_init__(self) -> None:
        """Initialize sources_used if None."""
        if self.sources_used is None:
            self.sources_used = []


class ResponseSynthesizer:
    """Synthesizes unified responses from multiple agent outputs using Bedrock.

    Uses a single LLM call to combine news_agent and sec_agent responses,
    taking into account any discrepancies detected between sources.
    """

    def __init__(
        self,
        model_id: str | None = None,
        bedrock_client: Any | None = None,
    ) -> None:
        """Initialize the synthesizer.

        Args:
            model_id: Bedrock model ID. Defaults to env var or Claude Opus 4.5.
            bedrock_client: Optional boto3 bedrock-runtime client for testing.
        """
        self._model_id = model_id or os.environ.get(
            "BEDROCK_MODEL_ID", "anthropic.claude-opus-4-5-20251101-v1:0"
        )
        self._client = bedrock_client or boto3.client("bedrock-runtime")

    def synthesize(self, synthesis_input: SynthesisInput) -> SynthesisResult:
        """Synthesize a unified response from agent outputs.

        Args:
            synthesis_input: The input data containing query and agent results.

        Returns:
            SynthesisResult containing the synthesized response.
        """
        # Build the user message with all available context
        user_message = self._build_user_message(synthesis_input)

        # Call Bedrock Converse API
        response = self._client.converse(
            modelId=self._model_id,
            system=[{"text": SYNTHESIS_SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": [{"text": user_message}]}],
        )

        # Extract the response text
        output = response["output"]["message"]
        text_parts = [
            block["text"] for block in output["content"] if "text" in block
        ]
        response_text = "\n".join(text_parts)

        # Determine sources used
        sources_used = []
        if synthesis_input.news_result and synthesis_input.news_result.is_success:
            sources_used.append("news_agent")
        if synthesis_input.sec_result and synthesis_input.sec_result.is_success:
            sources_used.append("sec_agent")

        # Determine confidence based on source availability and agreement
        confidence = self._calculate_confidence(synthesis_input)

        # Check for discrepancies
        had_discrepancies = (
            synthesis_input.comparison is not None
            and synthesis_input.comparison.has_discrepancies
        )

        return SynthesisResult(
            response=response_text,
            confidence=confidence,
            sources_used=sources_used,
            had_discrepancies=had_discrepancies,
            model_id=self._model_id,
        )

    def _build_user_message(self, synthesis_input: SynthesisInput) -> str:
        """Build the user message for the synthesis LLM call.

        Args:
            synthesis_input: The input data.

        Returns:
            Formatted user message string.
        """
        parts: list[str] = []

        # Add the original query
        parts.append(f"USER QUERY: {synthesis_input.user_query}")

        if synthesis_input.ticker:
            parts.append(f"TICKER: {synthesis_input.ticker}")

        parts.append("")  # Blank line

        # Add news agent result
        parts.append("--- NEWS AGENT RESPONSE ---")
        if synthesis_input.news_result is None:
            parts.append("Not requested for this query.")
        elif synthesis_input.news_result.is_success:
            parts.append(synthesis_input.news_result.response or "(empty response)")
        elif synthesis_input.news_result.is_timeout:
            err_msg = synthesis_input.news_result.error_message or "Request timed out"
            parts.append(f"TIMEOUT: {err_msg}")
        else:
            parts.append(f"ERROR: {synthesis_input.news_result.error_message or 'Unknown error'}")

        parts.append("")  # Blank line

        # Add SEC agent result
        parts.append("--- SEC AGENT RESPONSE ---")
        if synthesis_input.sec_result is None:
            parts.append("Not requested for this query.")
        elif synthesis_input.sec_result.is_success:
            parts.append(synthesis_input.sec_result.response or "(empty response)")
        elif synthesis_input.sec_result.is_timeout:
            err_msg = synthesis_input.sec_result.error_message or "Request timed out"
            parts.append(f"TIMEOUT: {err_msg}")
        else:
            parts.append(f"ERROR: {synthesis_input.sec_result.error_message or 'Unknown error'}")

        parts.append("")  # Blank line

        # Add comparison results if available
        if synthesis_input.comparison is not None:
            parts.append("--- SOURCE COMPARISON ---")
            parts.append(f"Summary: {synthesis_input.comparison.summary}")
            parts.append(f"Overall alignment: {synthesis_input.comparison.overall_alignment}")

            if synthesis_input.comparison.has_discrepancies:
                parts.append("")
                parts.append("DISCREPANCIES:")
                for d in synthesis_input.comparison.discrepancies:
                    severity_label = f"[{d.severity.value.upper()}]"
                    parts.append(f"  {severity_label} {d.description}")

            if synthesis_input.comparison.has_agreements:
                parts.append("")
                parts.append("AGREEMENTS:")
                for a in synthesis_input.comparison.agreements:
                    parts.append(f"  - {a.description}")

        parts.append("")
        parts.append(
            "Please synthesize a unified response to the user's query "
            "based on the above information."
        )

        return "\n".join(parts)

    def _calculate_confidence(self, synthesis_input: SynthesisInput) -> float:
        """Calculate confidence score for the synthesis.

        Args:
            synthesis_input: The input data.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        news_ok = (
            synthesis_input.news_result is not None
            and synthesis_input.news_result.is_success
        )
        sec_ok = (
            synthesis_input.sec_result is not None
            and synthesis_input.sec_result.is_success
        )

        # Base confidence on source availability
        if news_ok and sec_ok:
            base_confidence = 0.9
        elif news_ok or sec_ok:
            base_confidence = 0.7
        else:
            base_confidence = 0.3

        # Adjust based on alignment if comparison available
        if synthesis_input.comparison is not None:
            alignment = synthesis_input.comparison.overall_alignment
            # Positive alignment increases confidence, negative decreases
            # Scale: alignment of 1.0 adds 0.1, -1.0 subtracts 0.2
            if alignment > 0:
                base_confidence = min(1.0, base_confidence + alignment * 0.1)
            else:
                base_confidence = max(0.1, base_confidence + alignment * 0.2)

            # Critical discrepancies reduce confidence
            if synthesis_input.comparison.has_critical_discrepancies:
                base_confidence = max(0.1, base_confidence - 0.15)

        return round(base_confidence, 2)
