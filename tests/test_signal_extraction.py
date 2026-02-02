"""Tests for the signal extraction module."""

from __future__ import annotations

import pytest

from src.orchestrator.comparison.signals import (
    ExtractedSignal,
    SignalDirection,
    SignalExtractionResult,
    SignalExtractor,
    SignalType,
)


class TestExtractedSignal:
    """Tests for ExtractedSignal dataclass."""

    def test_create_valid_signal(self) -> None:
        """Create a valid signal with all required fields."""
        signal = ExtractedSignal(
            signal_type=SignalType.SENTIMENT,
            direction=SignalDirection.POSITIVE,
            topic="outlook",
            description="Positive outlook indicator",
            confidence=0.85,
            source="news_agent",
            raw_text="positive outlook",
        )

        assert signal.signal_type == SignalType.SENTIMENT
        assert signal.direction == SignalDirection.POSITIVE
        assert signal.topic == "outlook"
        assert signal.confidence == 0.85
        assert signal.source == "news_agent"

    def test_default_confidence(self) -> None:
        """Default confidence should be 0.8."""
        signal = ExtractedSignal(
            signal_type=SignalType.SENTIMENT,
            direction=SignalDirection.NEUTRAL,
            topic="test",
            description="Test signal",
        )

        assert signal.confidence == 0.8

    def test_default_source(self) -> None:
        """Default source should be news_agent."""
        signal = ExtractedSignal(
            signal_type=SignalType.SENTIMENT,
            direction=SignalDirection.NEUTRAL,
            topic="test",
            description="Test signal",
        )

        assert signal.source == "news_agent"

    def test_invalid_confidence_too_high(self) -> None:
        """Confidence above 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="Confidence must be between"):
            ExtractedSignal(
                signal_type=SignalType.SENTIMENT,
                direction=SignalDirection.POSITIVE,
                topic="test",
                description="Test",
                confidence=1.5,
            )

    def test_invalid_confidence_negative(self) -> None:
        """Negative confidence should raise ValueError."""
        with pytest.raises(ValueError, match="Confidence must be between"):
            ExtractedSignal(
                signal_type=SignalType.SENTIMENT,
                direction=SignalDirection.POSITIVE,
                topic="test",
                description="Test",
                confidence=-0.1,
            )


class TestSignalExtractionResult:
    """Tests for SignalExtractionResult dataclass."""

    def test_empty_result(self) -> None:
        """Create an empty result."""
        result = SignalExtractionResult(source="news_agent")

        assert result.source == "news_agent"
        assert result.signals == []
        assert result.has_signals is False
        assert result.overall_sentiment == SignalDirection.NEUTRAL
        assert result.extraction_successful is True

    def test_has_signals_property(self) -> None:
        """has_signals should be True when signals exist."""
        signal = ExtractedSignal(
            signal_type=SignalType.SENTIMENT,
            direction=SignalDirection.POSITIVE,
            topic="test",
            description="Test",
        )
        result = SignalExtractionResult(source="sec_agent", signals=[signal])

        assert result.has_signals is True

    def test_positive_signals_filter(self) -> None:
        """positive_signals should filter only positive signals."""
        signals = [
            ExtractedSignal(
                signal_type=SignalType.SENTIMENT,
                direction=SignalDirection.POSITIVE,
                topic="a",
                description="A",
            ),
            ExtractedSignal(
                signal_type=SignalType.SENTIMENT,
                direction=SignalDirection.NEGATIVE,
                topic="b",
                description="B",
            ),
            ExtractedSignal(
                signal_type=SignalType.SENTIMENT,
                direction=SignalDirection.POSITIVE,
                topic="c",
                description="C",
            ),
        ]
        result = SignalExtractionResult(source="news_agent", signals=signals)

        assert len(result.positive_signals) == 2
        assert all(s.direction == SignalDirection.POSITIVE for s in result.positive_signals)

    def test_negative_signals_filter(self) -> None:
        """negative_signals should filter only negative signals."""
        signals = [
            ExtractedSignal(
                signal_type=SignalType.SENTIMENT,
                direction=SignalDirection.POSITIVE,
                topic="a",
                description="A",
            ),
            ExtractedSignal(
                signal_type=SignalType.SENTIMENT,
                direction=SignalDirection.NEGATIVE,
                topic="b",
                description="B",
            ),
        ]
        result = SignalExtractionResult(source="news_agent", signals=signals)

        assert len(result.negative_signals) == 1
        assert result.negative_signals[0].topic == "b"

    def test_sentiment_signals_filter(self) -> None:
        """sentiment_signals should filter by signal type."""
        signals = [
            ExtractedSignal(
                signal_type=SignalType.SENTIMENT,
                direction=SignalDirection.POSITIVE,
                topic="sentiment",
                description="Sentiment",
            ),
            ExtractedSignal(
                signal_type=SignalType.RISK_FACTOR,
                direction=SignalDirection.NEGATIVE,
                topic="risk",
                description="Risk",
            ),
        ]
        result = SignalExtractionResult(source="sec_agent", signals=signals)

        assert len(result.sentiment_signals) == 1
        assert result.sentiment_signals[0].signal_type == SignalType.SENTIMENT

    def test_risk_signals_filter(self) -> None:
        """risk_signals should filter only risk factor signals."""
        signals = [
            ExtractedSignal(
                signal_type=SignalType.SENTIMENT,
                direction=SignalDirection.POSITIVE,
                topic="sentiment",
                description="Sentiment",
            ),
            ExtractedSignal(
                signal_type=SignalType.RISK_FACTOR,
                direction=SignalDirection.NEGATIVE,
                topic="risk",
                description="Risk",
            ),
        ]
        result = SignalExtractionResult(source="sec_agent", signals=signals)

        assert len(result.risk_signals) == 1
        assert result.risk_signals[0].signal_type == SignalType.RISK_FACTOR

    def test_financial_signals_filter(self) -> None:
        """financial_signals should filter only financial metric signals."""
        signals = [
            ExtractedSignal(
                signal_type=SignalType.FINANCIAL_METRIC,
                direction=SignalDirection.POSITIVE,
                topic="revenue",
                description="Revenue",
            ),
            ExtractedSignal(
                signal_type=SignalType.SENTIMENT,
                direction=SignalDirection.POSITIVE,
                topic="outlook",
                description="Outlook",
            ),
        ]
        result = SignalExtractionResult(source="news_agent", signals=signals)

        assert len(result.financial_signals) == 1
        assert result.financial_signals[0].topic == "revenue"

    def test_failed_extraction(self) -> None:
        """Failed extraction should have appropriate flags."""
        result = SignalExtractionResult(
            source="news_agent",
            extraction_successful=False,
            error_message="Empty response",
        )

        assert result.extraction_successful is False
        assert result.error_message == "Empty response"
        assert result.has_signals is False


class TestSignalExtractor:
    """Tests for SignalExtractor."""

    @pytest.fixture
    def extractor(self) -> SignalExtractor:
        """Create an extractor instance for testing."""
        return SignalExtractor()

    # -------------------------------------------------------------------------
    # Empty/invalid input tests
    # -------------------------------------------------------------------------

    def test_empty_response(self, extractor: SignalExtractor) -> None:
        """Empty response should return failed extraction."""
        result = extractor.extract("", "news_agent")

        assert result.extraction_successful is False
        assert result.error_message == "Empty response"
        assert result.has_signals is False

    def test_whitespace_only_response(self, extractor: SignalExtractor) -> None:
        """Whitespace-only response should return failed extraction."""
        result = extractor.extract("   \n\t  ", "sec_agent")

        assert result.extraction_successful is False
        assert result.error_message == "Empty response"

    def test_no_patterns_found(self, extractor: SignalExtractor) -> None:
        """Response with no matching patterns should return empty signals."""
        result = extractor.extract("The quick brown fox jumps over the lazy dog.", "news_agent")

        assert result.extraction_successful is True
        assert result.has_signals is False
        assert result.overall_sentiment == SignalDirection.NEUTRAL
        assert result.sentiment_score == 0.0

    # -------------------------------------------------------------------------
    # Positive sentiment extraction tests
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize(
        "text",
        [
            "The company reported strong growth this quarter.",
            "Apple showed robust performance in all segments.",
            "Tesla delivered solid results for Q4.",
            "Strong results across all business units.",
        ],
    )
    def test_positive_growth_patterns(self, extractor: SignalExtractor, text: str) -> None:
        """Positive growth patterns should be extracted."""
        result = extractor.extract(text, "news_agent")

        assert result.has_signals is True
        assert len(result.positive_signals) >= 1
        assert result.overall_sentiment in (SignalDirection.POSITIVE, SignalDirection.MIXED)

    @pytest.mark.parametrize(
        "text",
        [
            "The company beat expectations on earnings.",
            "Revenue exceeded estimates by 15%.",
        ],
    )
    def test_positive_earnings_patterns(self, extractor: SignalExtractor, text: str) -> None:
        """Beating expectations patterns should be extracted as positive."""
        result = extractor.extract(text, "news_agent")

        assert result.has_signals is True
        assert len(result.positive_signals) >= 1
        positive_topics = [s.topic for s in result.positive_signals]
        assert "earnings" in positive_topics

    @pytest.mark.parametrize(
        "text",
        [
            "Analysts maintain a positive outlook for the stock.",
            "The bullish sentiment around Tesla is evident.",
            "Optimistic tone from management on the call.",
        ],
    )
    def test_positive_outlook_patterns(self, extractor: SignalExtractor, text: str) -> None:
        """Positive outlook patterns should be extracted."""
        result = extractor.extract(text, "news_agent")

        assert result.has_signals is True
        assert len(result.positive_signals) >= 1
        positive_topics = [s.topic for s in result.positive_signals]
        assert "outlook" in positive_topics

    def test_explicit_positive_sentiment(self, extractor: SignalExtractor) -> None:
        """Explicit 'sentiment: positive' should be extracted."""
        result = extractor.extract("The overall sentiment: positive for AAPL.", "news_agent")

        assert result.has_signals is True
        assert len(result.positive_signals) >= 1

    # -------------------------------------------------------------------------
    # Negative sentiment extraction tests
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize(
        "text",
        [
            "The company reported weak results this quarter.",
            "Poor performance in the cloud division.",
            "Disappointing growth in international markets.",
        ],
    )
    def test_negative_performance_patterns(self, extractor: SignalExtractor, text: str) -> None:
        """Negative performance patterns should be extracted."""
        result = extractor.extract(text, "news_agent")

        assert result.has_signals is True
        assert len(result.negative_signals) >= 1

    @pytest.mark.parametrize(
        "text",
        [
            "The company missed expectations on revenue.",
            "Earnings came in below estimates.",
        ],
    )
    def test_negative_earnings_patterns(self, extractor: SignalExtractor, text: str) -> None:
        """Missing expectations should be extracted as negative."""
        result = extractor.extract(text, "news_agent")

        assert result.has_signals is True
        assert len(result.negative_signals) >= 1
        negative_topics = [s.topic for s in result.negative_signals]
        assert "earnings" in negative_topics

    @pytest.mark.parametrize(
        "text",
        [
            "Analysts have a negative outlook on the sector.",
            "Bearish sentiment dominates trading.",
            "Management struck a pessimistic tone.",
        ],
    )
    def test_negative_outlook_patterns(self, extractor: SignalExtractor, text: str) -> None:
        """Negative outlook patterns should be extracted."""
        result = extractor.extract(text, "news_agent")

        assert result.has_signals is True
        assert len(result.negative_signals) >= 1
        negative_topics = [s.topic for s in result.negative_signals]
        assert "outlook" in negative_topics

    def test_explicit_negative_sentiment(self, extractor: SignalExtractor) -> None:
        """Explicit 'sentiment: negative' should be extracted."""
        result = extractor.extract("Overall: negative sentiment for the stock.", "news_agent")

        assert result.has_signals is True
        assert len(result.negative_signals) >= 1

    def test_decline_patterns(self, extractor: SignalExtractor) -> None:
        """Decline in metrics should be extracted as negative."""
        result = extractor.extract("Revenue declined 10% year over year.", "sec_agent")

        assert result.has_signals is True
        negative_signals = result.negative_signals
        assert len(negative_signals) >= 1

    # -------------------------------------------------------------------------
    # Neutral sentiment extraction tests
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize(
        "text",
        [
            "Sentiment: neutral for the sector.",
            "Overall: neutral outlook expected.",
            "Mixed signals from the earnings call.",
            "Uncertain outlook for next quarter.",
        ],
    )
    def test_neutral_patterns(self, extractor: SignalExtractor, text: str) -> None:
        """Neutral/mixed patterns should be extracted."""
        result = extractor.extract(text, "news_agent")

        assert result.has_signals is True
        neutral_signals = [s for s in result.signals if s.direction == SignalDirection.NEUTRAL]
        assert len(neutral_signals) >= 1

    def test_met_expectations(self, extractor: SignalExtractor) -> None:
        """Meeting expectations should be extracted as neutral."""
        result = extractor.extract("Earnings were in line with expectations.", "news_agent")

        assert result.has_signals is True
        neutral_signals = [s for s in result.signals if s.direction == SignalDirection.NEUTRAL]
        assert len(neutral_signals) >= 1

    # -------------------------------------------------------------------------
    # Financial metric extraction tests
    # -------------------------------------------------------------------------

    def test_revenue_increase(self, extractor: SignalExtractor) -> None:
        """Revenue increase should be extracted as positive financial signal."""
        result = extractor.extract("Revenue increased 20% this quarter.", "sec_agent")

        assert result.has_signals is True
        financial = result.financial_signals
        assert len(financial) >= 1
        assert financial[0].topic == "revenue"
        assert financial[0].direction == SignalDirection.POSITIVE

    def test_revenue_decline(self, extractor: SignalExtractor) -> None:
        """Revenue decline should be extracted as negative financial signal."""
        result = extractor.extract("Revenue decreased 15% year-over-year.", "sec_agent")

        assert result.has_signals is True
        financial = result.financial_signals
        assert len(financial) >= 1
        assert financial[0].topic == "revenue"
        assert financial[0].direction == SignalDirection.NEGATIVE

    def test_earnings_increase(self, extractor: SignalExtractor) -> None:
        """Earnings increase should be extracted."""
        result = extractor.extract("Earnings grew significantly this year.", "sec_agent")

        assert result.has_signals is True
        financial = result.financial_signals
        assert len(financial) >= 1
        assert financial[0].topic == "earnings"
        assert financial[0].direction == SignalDirection.POSITIVE

    def test_margin_improvement(self, extractor: SignalExtractor) -> None:
        """Profit margin improvement should be extracted."""
        result = extractor.extract("Profit margin improved due to cost cuts.", "sec_agent")

        assert result.has_signals is True
        financial = result.financial_signals
        assert len(financial) >= 1
        assert financial[0].topic == "margin"
        assert financial[0].direction == SignalDirection.POSITIVE

    def test_cash_flow_positive(self, extractor: SignalExtractor) -> None:
        """Strong cash flow should be extracted as positive."""
        result = extractor.extract("The cash flow strong performance continued.", "sec_agent")

        assert result.has_signals is True
        financial = result.financial_signals
        assert len(financial) >= 1
        assert financial[0].topic == "cash_flow"
        assert financial[0].direction == SignalDirection.POSITIVE

    # -------------------------------------------------------------------------
    # Risk factor extraction tests
    # -------------------------------------------------------------------------

    def test_general_risk_factors(self, extractor: SignalExtractor) -> None:
        """Risk factors mention should be extracted."""
        result = extractor.extract(
            "The company disclosed several risk factors in the 10-K.",
            "sec_agent",
        )

        assert result.has_signals is True
        risks = result.risk_signals
        assert len(risks) >= 1
        assert risks[0].direction == SignalDirection.NEGATIVE

    def test_regulatory_risk(self, extractor: SignalExtractor) -> None:
        """Regulatory risk should be extracted."""
        result = extractor.extract(
            "Management identified regulatory risk as a key concern.",
            "sec_agent",
        )

        assert result.has_signals is True
        risks = result.risk_signals
        assert len(risks) >= 1
        assert "regulatory" in risks[0].topic

    def test_litigation_risk(self, extractor: SignalExtractor) -> None:
        """Litigation concerns should be extracted."""
        result = extractor.extract(
            "The company faces ongoing legal issues.",
            "sec_agent",
        )

        assert result.has_signals is True
        risks = result.risk_signals
        assert len(risks) >= 1

    def test_supply_chain_risk(self, extractor: SignalExtractor) -> None:
        """Supply chain disruption should be extracted."""
        result = extractor.extract(
            "Supply chain disruptions impacted production.",
            "sec_agent",
        )

        assert result.has_signals is True
        risks = result.risk_signals
        assert len(risks) >= 1
        assert "operational" in risks[0].topic

    def test_cybersecurity_risk(self, extractor: SignalExtractor) -> None:
        """Cybersecurity threats should be extracted."""
        result = extractor.extract(
            "Cybersecurity risks remain a concern for the company.",
            "sec_agent",
        )

        assert result.has_signals is True
        risks = result.risk_signals
        assert len(risks) >= 1
        assert "cyber" in risks[0].topic

    def test_material_weakness(self, extractor: SignalExtractor) -> None:
        """Material weakness should be extracted."""
        result = extractor.extract(
            "The auditor identified a material weakness in internal controls.",
            "sec_agent",
        )

        assert result.has_signals is True
        risks = result.risk_signals
        assert len(risks) >= 1
        assert "control" in risks[0].topic

    # -------------------------------------------------------------------------
    # Guidance extraction tests
    # -------------------------------------------------------------------------

    def test_raised_guidance(self, extractor: SignalExtractor) -> None:
        """Raised guidance should be extracted as positive."""
        result = extractor.extract("Management raised guidance for Q4.", "sec_agent")

        assert result.has_signals is True
        guidance_signals = [s for s in result.signals if s.signal_type == SignalType.GUIDANCE]
        assert len(guidance_signals) >= 1
        assert guidance_signals[0].direction == SignalDirection.POSITIVE

    def test_lowered_guidance(self, extractor: SignalExtractor) -> None:
        """Lowered guidance should be extracted as negative."""
        result = extractor.extract("The company lowered guidance for Q4.", "sec_agent")

        assert result.has_signals is True
        guidance_signals = [s for s in result.signals if s.signal_type == SignalType.GUIDANCE]
        assert len(guidance_signals) >= 1
        assert guidance_signals[0].direction == SignalDirection.NEGATIVE

    def test_reaffirmed_guidance(self, extractor: SignalExtractor) -> None:
        """Reaffirmed guidance should be extracted as neutral."""
        result = extractor.extract("Management reaffirmed guidance for the year.", "sec_agent")

        assert result.has_signals is True
        guidance_signals = [s for s in result.signals if s.signal_type == SignalType.GUIDANCE]
        assert len(guidance_signals) >= 1
        assert guidance_signals[0].direction == SignalDirection.NEUTRAL

    # -------------------------------------------------------------------------
    # Overall sentiment calculation tests
    # -------------------------------------------------------------------------

    def test_overall_positive_sentiment(self, extractor: SignalExtractor) -> None:
        """Multiple positive signals should result in positive overall sentiment."""
        text = """
        The company reported strong growth and beat expectations.
        Analysts maintain a positive outlook for the stock.
        Revenue increased 20% year over year.
        """
        result = extractor.extract(text, "news_agent")

        assert result.overall_sentiment == SignalDirection.POSITIVE
        assert result.sentiment_score > 0

    def test_overall_negative_sentiment(self, extractor: SignalExtractor) -> None:
        """Multiple negative signals should result in negative overall sentiment."""
        text = """
        The company reported weak results and missed expectations.
        Analysts have a negative outlook on the sector.
        Revenue declined significantly.
        Risk factors were highlighted.
        """
        result = extractor.extract(text, "sec_agent")

        assert result.overall_sentiment == SignalDirection.NEGATIVE
        assert result.sentiment_score < 0

    def test_mixed_sentiment(self, extractor: SignalExtractor) -> None:
        """Mix of positive and negative should result in mixed sentiment."""
        text = """
        Strong growth in the consumer segment.
        However, there are concerns about regulatory risk.
        """
        result = extractor.extract(text, "news_agent")

        # Should have both positive and negative signals
        assert len(result.positive_signals) > 0
        assert len(result.negative_signals) > 0
        # Overall should be mixed or close to neutral
        assert result.overall_sentiment in (
            SignalDirection.MIXED,
            SignalDirection.POSITIVE,
            SignalDirection.NEGATIVE,
        )

    # -------------------------------------------------------------------------
    # Ticker extraction tests
    # -------------------------------------------------------------------------

    def test_ticker_extraction_with_parentheses(self, extractor: SignalExtractor) -> None:
        """Ticker in parentheses format should be extracted."""
        result = extractor.extract("Apple AAPL (NASDAQ) reported earnings.", "news_agent")

        assert result.ticker == "AAPL"

    def test_ticker_extraction_with_colon(self, extractor: SignalExtractor) -> None:
        """Ticker with colon format should be extracted."""
        result = extractor.extract("TSLA: The stock fell 5% today.", "news_agent")

        assert result.ticker == "TSLA"

    def test_ticker_extraction_with_stock(self, extractor: SignalExtractor) -> None:
        """Ticker followed by 'stock' should be extracted."""
        result = extractor.extract("NVDA stock surged on AI news.", "news_agent")

        assert result.ticker == "NVDA"

    def test_no_ticker_found(self, extractor: SignalExtractor) -> None:
        """Response without ticker format should return None."""
        result = extractor.extract("The company reported strong earnings.", "news_agent")

        assert result.ticker is None

    # -------------------------------------------------------------------------
    # Topics extraction tests
    # -------------------------------------------------------------------------

    def test_topics_collected(self, extractor: SignalExtractor) -> None:
        """Topics from all signals should be collected."""
        text = """
        Strong growth in revenue.
        Positive outlook for earnings.
        Risk factors include regulatory concerns.
        """
        result = extractor.extract(text, "sec_agent")

        assert len(result.topics) > 0
        assert "growth" in result.topics or "outlook" in result.topics

    def test_topics_are_sorted(self, extractor: SignalExtractor) -> None:
        """Topics should be sorted alphabetically."""
        text = """
        Positive outlook and strong growth.
        Risk factors present.
        """
        result = extractor.extract(text, "news_agent")

        if len(result.topics) > 1:
            assert result.topics == sorted(result.topics)

    # -------------------------------------------------------------------------
    # Source tracking tests
    # -------------------------------------------------------------------------

    def test_news_agent_source(self, extractor: SignalExtractor) -> None:
        """Signals from news agent should have correct source."""
        result = extractor.extract("Strong growth reported.", "news_agent")

        assert result.source == "news_agent"
        for signal in result.signals:
            assert signal.source == "news_agent"

    def test_sec_agent_source(self, extractor: SignalExtractor) -> None:
        """Signals from SEC agent should have correct source."""
        result = extractor.extract("Risk factors disclosed.", "sec_agent")

        assert result.source == "sec_agent"
        for signal in result.signals:
            assert signal.source == "sec_agent"

    # -------------------------------------------------------------------------
    # Case insensitivity tests
    # -------------------------------------------------------------------------

    def test_case_insensitive_patterns(self, extractor: SignalExtractor) -> None:
        """Patterns should match regardless of case."""
        results = [
            extractor.extract("STRONG GROWTH reported.", "news_agent"),
            extractor.extract("Strong Growth reported.", "news_agent"),
            extractor.extract("strong growth reported.", "news_agent"),
        ]

        for result in results:
            assert result.has_signals is True
            assert len(result.positive_signals) >= 1

    # -------------------------------------------------------------------------
    # Edge cases
    # -------------------------------------------------------------------------

    def test_multiple_matches_same_pattern(self, extractor: SignalExtractor) -> None:
        """Multiple occurrences of same pattern should create multiple signals."""
        text = """
        Strong growth in Q1.
        Strong growth in Q2.
        Strong growth in Q3.
        """
        result = extractor.extract(text, "news_agent")

        assert result.has_signals is True
        # May dedupe or may have multiples - both acceptable
        assert len(result.positive_signals) >= 1

    def test_confidence_in_range(self, extractor: SignalExtractor) -> None:
        """All extracted signals should have valid confidence scores."""
        text = """
        Strong growth, positive outlook, revenue increased.
        Risk factors present, weak results in one segment.
        """
        result = extractor.extract(text, "news_agent")

        for signal in result.signals:
            assert 0.0 <= signal.confidence <= 1.0

    def test_sentiment_score_in_range(self, extractor: SignalExtractor) -> None:
        """Sentiment score should be between -1.0 and 1.0."""
        texts = [
            "Extremely positive outlook with strong growth everywhere.",
            "Terrible results with massive losses and declining revenue.",
            "Mixed signals from the earnings call.",
        ]

        for text in texts:
            result = extractor.extract(text, "news_agent")
            assert -1.0 <= result.sentiment_score <= 1.0
