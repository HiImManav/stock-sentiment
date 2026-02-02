"""Tests for the discrepancy detection module."""

from __future__ import annotations

import pytest

from orchestrator.comparison.discrepancy import (
    Agreement,
    ComparisonResult,
    Discrepancy,
    DiscrepancyDetector,
    DiscrepancySeverity,
    DiscrepancyType,
    _directions_agree,
    _directions_conflict,
    _topics_are_related,
)
from orchestrator.comparison.signals import (
    ExtractedSignal,
    SignalDirection,
    SignalExtractionResult,
    SignalType,
)


class TestDiscrepancy:
    """Tests for Discrepancy dataclass."""

    def test_create_valid_discrepancy(self) -> None:
        """Create a valid discrepancy with all required fields."""
        discrepancy = Discrepancy(
            discrepancy_type=DiscrepancyType.SENTIMENT_CONFLICT,
            severity=DiscrepancySeverity.HIGH,
            topic="outlook",
            news_signal=None,
            sec_signal=None,
            description="Test discrepancy",
            confidence=0.85,
        )

        assert discrepancy.discrepancy_type == DiscrepancyType.SENTIMENT_CONFLICT
        assert discrepancy.severity == DiscrepancySeverity.HIGH
        assert discrepancy.topic == "outlook"
        assert discrepancy.confidence == 0.85

    def test_default_confidence(self) -> None:
        """Default confidence should be 0.8."""
        discrepancy = Discrepancy(
            discrepancy_type=DiscrepancyType.SENTIMENT_CONFLICT,
            severity=DiscrepancySeverity.LOW,
            topic="test",
            news_signal=None,
            sec_signal=None,
            description="Test",
        )

        assert discrepancy.confidence == 0.8

    def test_invalid_confidence_too_high(self) -> None:
        """Confidence above 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="Confidence must be between"):
            Discrepancy(
                discrepancy_type=DiscrepancyType.SENTIMENT_CONFLICT,
                severity=DiscrepancySeverity.LOW,
                topic="test",
                news_signal=None,
                sec_signal=None,
                description="Test",
                confidence=1.5,
            )

    def test_invalid_confidence_negative(self) -> None:
        """Negative confidence should raise ValueError."""
        with pytest.raises(ValueError, match="Confidence must be between"):
            Discrepancy(
                discrepancy_type=DiscrepancyType.SENTIMENT_CONFLICT,
                severity=DiscrepancySeverity.LOW,
                topic="test",
                news_signal=None,
                sec_signal=None,
                description="Test",
                confidence=-0.1,
            )


class TestAgreement:
    """Tests for Agreement dataclass."""

    def test_create_valid_agreement(self) -> None:
        """Create a valid agreement with all required fields."""
        news_signal = ExtractedSignal(
            signal_type=SignalType.SENTIMENT,
            direction=SignalDirection.POSITIVE,
            topic="outlook",
            description="Positive outlook",
            source="news_agent",
        )
        sec_signal = ExtractedSignal(
            signal_type=SignalType.SENTIMENT,
            direction=SignalDirection.POSITIVE,
            topic="outlook",
            description="Positive outlook",
            source="sec_agent",
        )

        agreement = Agreement(
            topic="outlook",
            direction=SignalDirection.POSITIVE,
            news_signal=news_signal,
            sec_signal=sec_signal,
            description="Both agree on positive outlook",
            confidence=0.9,
        )

        assert agreement.topic == "outlook"
        assert agreement.direction == SignalDirection.POSITIVE
        assert agreement.confidence == 0.9


class TestComparisonResult:
    """Tests for ComparisonResult dataclass."""

    def test_empty_result(self) -> None:
        """Create an empty comparison result."""
        result = ComparisonResult(
            news_result=None,
            sec_result=None,
        )

        assert result.news_result is None
        assert result.sec_result is None
        assert result.has_discrepancies is False
        assert result.has_agreements is False
        assert result.discrepancy_count == 0
        assert result.agreement_count == 0

    def test_has_discrepancies_property(self) -> None:
        """has_discrepancies should be True when discrepancies exist."""
        discrepancy = Discrepancy(
            discrepancy_type=DiscrepancyType.SENTIMENT_CONFLICT,
            severity=DiscrepancySeverity.LOW,
            topic="test",
            news_signal=None,
            sec_signal=None,
            description="Test",
        )
        result = ComparisonResult(
            news_result=None,
            sec_result=None,
            discrepancies=[discrepancy],
        )

        assert result.has_discrepancies is True
        assert result.discrepancy_count == 1

    def test_high_severity_discrepancies_filter(self) -> None:
        """high_severity_discrepancies should filter only HIGH severity."""
        discrepancies = [
            Discrepancy(
                discrepancy_type=DiscrepancyType.SENTIMENT_CONFLICT,
                severity=DiscrepancySeverity.HIGH,
                topic="a",
                news_signal=None,
                sec_signal=None,
                description="A",
            ),
            Discrepancy(
                discrepancy_type=DiscrepancyType.SENTIMENT_CONFLICT,
                severity=DiscrepancySeverity.LOW,
                topic="b",
                news_signal=None,
                sec_signal=None,
                description="B",
            ),
            Discrepancy(
                discrepancy_type=DiscrepancyType.FINANCIAL_CONFLICT,
                severity=DiscrepancySeverity.HIGH,
                topic="c",
                news_signal=None,
                sec_signal=None,
                description="C",
            ),
        ]
        result = ComparisonResult(
            news_result=None,
            sec_result=None,
            discrepancies=discrepancies,
        )

        high = result.high_severity_discrepancies
        assert len(high) == 2
        assert all(d.severity == DiscrepancySeverity.HIGH for d in high)

    def test_sentiment_discrepancies_filter(self) -> None:
        """sentiment_discrepancies should filter by type."""
        discrepancies = [
            Discrepancy(
                discrepancy_type=DiscrepancyType.SENTIMENT_CONFLICT,
                severity=DiscrepancySeverity.LOW,
                topic="a",
                news_signal=None,
                sec_signal=None,
                description="A",
            ),
            Discrepancy(
                discrepancy_type=DiscrepancyType.FINANCIAL_CONFLICT,
                severity=DiscrepancySeverity.HIGH,
                topic="b",
                news_signal=None,
                sec_signal=None,
                description="B",
            ),
        ]
        result = ComparisonResult(
            news_result=None,
            sec_result=None,
            discrepancies=discrepancies,
        )

        sentiment = result.sentiment_discrepancies
        assert len(sentiment) == 1
        assert sentiment[0].discrepancy_type == DiscrepancyType.SENTIMENT_CONFLICT

    def test_financial_discrepancies_filter(self) -> None:
        """financial_discrepancies should filter by type."""
        discrepancies = [
            Discrepancy(
                discrepancy_type=DiscrepancyType.SENTIMENT_CONFLICT,
                severity=DiscrepancySeverity.LOW,
                topic="a",
                news_signal=None,
                sec_signal=None,
                description="A",
            ),
            Discrepancy(
                discrepancy_type=DiscrepancyType.FINANCIAL_CONFLICT,
                severity=DiscrepancySeverity.HIGH,
                topic="b",
                news_signal=None,
                sec_signal=None,
                description="B",
            ),
        ]
        result = ComparisonResult(
            news_result=None,
            sec_result=None,
            discrepancies=discrepancies,
        )

        financial = result.financial_discrepancies
        assert len(financial) == 1
        assert financial[0].discrepancy_type == DiscrepancyType.FINANCIAL_CONFLICT


class TestHelperFunctions:
    """Tests for helper functions."""

    # -------------------------------------------------------------------------
    # _topics_are_related tests
    # -------------------------------------------------------------------------

    def test_exact_topic_match(self) -> None:
        """Exact topic match should be related."""
        assert _topics_are_related("revenue", "revenue") is True
        assert _topics_are_related("outlook", "outlook") is True

    def test_sentiment_topics_related(self) -> None:
        """Topics in sentiment category should be related."""
        assert _topics_are_related("sentiment", "outlook") is True
        assert _topics_are_related("rating", "growth") is True
        assert _topics_are_related("performance", "sentiment") is True

    def test_financial_topics_related(self) -> None:
        """Topics in financial category should be related."""
        assert _topics_are_related("revenue", "earnings") is True
        assert _topics_are_related("margin", "cash_flow") is True

    def test_guidance_topics_related(self) -> None:
        """Topics in guidance category should be related."""
        assert _topics_are_related("guidance", "forecast") is True
        assert _topics_are_related("forward_looking", "guidance") is True

    def test_risk_topics_related(self) -> None:
        """Topics in risk category should be related."""
        assert _topics_are_related("regulatory_risk", "market_risk") is True
        assert _topics_are_related("cyber_risk", "operational_risk") is True

    def test_unrelated_topics(self) -> None:
        """Topics from different categories should not be related."""
        assert _topics_are_related("revenue", "outlook") is False
        assert _topics_are_related("guidance", "cyber_risk") is False
        assert _topics_are_related("sentiment", "margin") is False

    # -------------------------------------------------------------------------
    # _directions_conflict tests
    # -------------------------------------------------------------------------

    def test_positive_negative_conflict(self) -> None:
        """Positive vs negative should conflict."""
        assert _directions_conflict(SignalDirection.POSITIVE, SignalDirection.NEGATIVE) is True
        assert _directions_conflict(SignalDirection.NEGATIVE, SignalDirection.POSITIVE) is True

    def test_same_direction_no_conflict(self) -> None:
        """Same direction should not conflict."""
        assert _directions_conflict(SignalDirection.POSITIVE, SignalDirection.POSITIVE) is False
        assert _directions_conflict(SignalDirection.NEGATIVE, SignalDirection.NEGATIVE) is False

    def test_neutral_no_conflict(self) -> None:
        """Neutral with anything should not conflict."""
        assert _directions_conflict(SignalDirection.NEUTRAL, SignalDirection.POSITIVE) is False
        assert _directions_conflict(SignalDirection.NEUTRAL, SignalDirection.NEGATIVE) is False
        assert _directions_conflict(SignalDirection.POSITIVE, SignalDirection.NEUTRAL) is False

    def test_mixed_no_conflict(self) -> None:
        """Mixed with anything should not conflict."""
        assert _directions_conflict(SignalDirection.MIXED, SignalDirection.POSITIVE) is False
        assert _directions_conflict(SignalDirection.MIXED, SignalDirection.NEGATIVE) is False

    # -------------------------------------------------------------------------
    # _directions_agree tests
    # -------------------------------------------------------------------------

    def test_same_direction_agrees(self) -> None:
        """Same direction should agree."""
        assert _directions_agree(SignalDirection.POSITIVE, SignalDirection.POSITIVE) is True
        assert _directions_agree(SignalDirection.NEGATIVE, SignalDirection.NEGATIVE) is True
        assert _directions_agree(SignalDirection.NEUTRAL, SignalDirection.NEUTRAL) is True

    def test_different_directions_no_agreement(self) -> None:
        """Different directions should not agree."""
        assert _directions_agree(SignalDirection.POSITIVE, SignalDirection.NEGATIVE) is False
        assert _directions_agree(SignalDirection.POSITIVE, SignalDirection.NEUTRAL) is False

    def test_mixed_no_agreement(self) -> None:
        """Mixed with itself should not count as agreement."""
        assert _directions_agree(SignalDirection.MIXED, SignalDirection.MIXED) is False


class TestDiscrepancyDetector:
    """Tests for DiscrepancyDetector."""

    @pytest.fixture
    def detector(self) -> DiscrepancyDetector:
        """Create a detector instance for testing."""
        return DiscrepancyDetector()

    # -------------------------------------------------------------------------
    # Null/empty input tests
    # -------------------------------------------------------------------------

    def test_both_results_none(self, detector: DiscrepancyDetector) -> None:
        """Both results None should return empty comparison."""
        result = detector.compare(None, None)

        assert result.news_result is None
        assert result.sec_result is None
        assert result.has_discrepancies is False
        assert "No signals available" in result.summary

    def test_news_result_none(self, detector: DiscrepancyDetector) -> None:
        """Only news result None should indicate limited comparison."""
        sec_result = SignalExtractionResult(source="sec_agent")
        result = detector.compare(None, sec_result)

        assert result.news_result is None
        assert result.sec_result is sec_result
        assert "News agent signals unavailable" in result.summary

    def test_sec_result_none(self, detector: DiscrepancyDetector) -> None:
        """Only SEC result None should indicate limited comparison."""
        news_result = SignalExtractionResult(source="news_agent")
        result = detector.compare(news_result, None)

        assert result.news_result is news_result
        assert result.sec_result is None
        assert "SEC agent signals unavailable" in result.summary

    def test_news_extraction_failed(self, detector: DiscrepancyDetector) -> None:
        """Failed news extraction should indicate limited comparison."""
        news_result = SignalExtractionResult(
            source="news_agent",
            extraction_successful=False,
            error_message="Empty response",
        )
        sec_result = SignalExtractionResult(source="sec_agent")
        result = detector.compare(news_result, sec_result)

        assert "News agent signals unavailable" in result.summary

    def test_sec_extraction_failed(self, detector: DiscrepancyDetector) -> None:
        """Failed SEC extraction should indicate limited comparison."""
        news_result = SignalExtractionResult(source="news_agent")
        sec_result = SignalExtractionResult(
            source="sec_agent",
            extraction_successful=False,
            error_message="Empty response",
        )
        result = detector.compare(news_result, sec_result)

        assert "SEC agent signals unavailable" in result.summary

    # -------------------------------------------------------------------------
    # Sentiment conflict detection tests
    # -------------------------------------------------------------------------

    def test_detect_overall_sentiment_conflict(self, detector: DiscrepancyDetector) -> None:
        """Overall sentiment conflict should be detected."""
        news_result = SignalExtractionResult(
            source="news_agent",
            overall_sentiment=SignalDirection.POSITIVE,
            signals=[
                ExtractedSignal(
                    signal_type=SignalType.SENTIMENT,
                    direction=SignalDirection.POSITIVE,
                    topic="outlook",
                    description="Positive outlook",
                    source="news_agent",
                )
            ],
        )
        sec_result = SignalExtractionResult(
            source="sec_agent",
            overall_sentiment=SignalDirection.NEGATIVE,
            signals=[
                ExtractedSignal(
                    signal_type=SignalType.SENTIMENT,
                    direction=SignalDirection.NEGATIVE,
                    topic="outlook",
                    description="Negative outlook",
                    source="sec_agent",
                )
            ],
        )

        result = detector.compare(news_result, sec_result)

        assert result.has_discrepancies is True
        sentiment_conflicts = result.sentiment_discrepancies
        assert len(sentiment_conflicts) >= 1
        assert any(d.topic == "overall_sentiment" for d in sentiment_conflicts)

    def test_detect_topic_sentiment_conflict(self, detector: DiscrepancyDetector) -> None:
        """Sentiment conflict on specific topic should be detected."""
        news_result = SignalExtractionResult(
            source="news_agent",
            overall_sentiment=SignalDirection.POSITIVE,
            signals=[
                ExtractedSignal(
                    signal_type=SignalType.SENTIMENT,
                    direction=SignalDirection.POSITIVE,
                    topic="growth",
                    description="Strong growth",
                    source="news_agent",
                )
            ],
        )
        sec_result = SignalExtractionResult(
            source="sec_agent",
            overall_sentiment=SignalDirection.NEGATIVE,
            signals=[
                ExtractedSignal(
                    signal_type=SignalType.SENTIMENT,
                    direction=SignalDirection.NEGATIVE,
                    topic="performance",  # Related to growth
                    description="Weak performance",
                    source="sec_agent",
                )
            ],
        )

        result = detector.compare(news_result, sec_result)

        assert result.has_discrepancies is True
        # Should detect conflict on related topics (growth/performance)
        sentiment_conflicts = result.sentiment_discrepancies
        assert len(sentiment_conflicts) >= 1

    def test_no_sentiment_conflict_same_direction(self, detector: DiscrepancyDetector) -> None:
        """Same sentiment direction should not create conflict."""
        news_result = SignalExtractionResult(
            source="news_agent",
            overall_sentiment=SignalDirection.POSITIVE,
            signals=[
                ExtractedSignal(
                    signal_type=SignalType.SENTIMENT,
                    direction=SignalDirection.POSITIVE,
                    topic="outlook",
                    description="Positive outlook",
                    source="news_agent",
                )
            ],
        )
        sec_result = SignalExtractionResult(
            source="sec_agent",
            overall_sentiment=SignalDirection.POSITIVE,
            signals=[
                ExtractedSignal(
                    signal_type=SignalType.SENTIMENT,
                    direction=SignalDirection.POSITIVE,
                    topic="outlook",
                    description="Positive outlook",
                    source="sec_agent",
                )
            ],
        )

        result = detector.compare(news_result, sec_result)

        sentiment_conflicts = result.sentiment_discrepancies
        assert len(sentiment_conflicts) == 0

    # -------------------------------------------------------------------------
    # Financial conflict detection tests
    # -------------------------------------------------------------------------

    def test_detect_financial_conflict(self, detector: DiscrepancyDetector) -> None:
        """Financial metric conflict should be detected."""
        news_result = SignalExtractionResult(
            source="news_agent",
            overall_sentiment=SignalDirection.POSITIVE,
            signals=[
                ExtractedSignal(
                    signal_type=SignalType.FINANCIAL_METRIC,
                    direction=SignalDirection.POSITIVE,
                    topic="revenue",
                    description="Revenue increased",
                    source="news_agent",
                )
            ],
        )
        sec_result = SignalExtractionResult(
            source="sec_agent",
            overall_sentiment=SignalDirection.NEGATIVE,
            signals=[
                ExtractedSignal(
                    signal_type=SignalType.FINANCIAL_METRIC,
                    direction=SignalDirection.NEGATIVE,
                    topic="revenue",
                    description="Revenue decreased",
                    source="sec_agent",
                )
            ],
        )

        result = detector.compare(news_result, sec_result)

        assert result.has_discrepancies is True
        financial_conflicts = result.financial_discrepancies
        assert len(financial_conflicts) >= 1
        assert financial_conflicts[0].severity == DiscrepancySeverity.HIGH

    def test_detect_related_financial_conflict(self, detector: DiscrepancyDetector) -> None:
        """Financial conflict on related metrics should be detected."""
        news_result = SignalExtractionResult(
            source="news_agent",
            overall_sentiment=SignalDirection.POSITIVE,
            signals=[
                ExtractedSignal(
                    signal_type=SignalType.FINANCIAL_METRIC,
                    direction=SignalDirection.POSITIVE,
                    topic="earnings",
                    description="Earnings up",
                    source="news_agent",
                )
            ],
        )
        sec_result = SignalExtractionResult(
            source="sec_agent",
            overall_sentiment=SignalDirection.NEGATIVE,
            signals=[
                ExtractedSignal(
                    signal_type=SignalType.FINANCIAL_METRIC,
                    direction=SignalDirection.NEGATIVE,
                    topic="margin",  # Related to earnings
                    description="Margin compressed",
                    source="sec_agent",
                )
            ],
        )

        result = detector.compare(news_result, sec_result)

        assert result.has_discrepancies is True
        financial_conflicts = result.financial_discrepancies
        assert len(financial_conflicts) >= 1

    # -------------------------------------------------------------------------
    # Guidance conflict detection tests
    # -------------------------------------------------------------------------

    def test_detect_guidance_conflict(self, detector: DiscrepancyDetector) -> None:
        """Guidance conflict should be detected."""
        news_result = SignalExtractionResult(
            source="news_agent",
            overall_sentiment=SignalDirection.POSITIVE,
            signals=[
                ExtractedSignal(
                    signal_type=SignalType.GUIDANCE,
                    direction=SignalDirection.POSITIVE,
                    topic="guidance",
                    description="Raised guidance",
                    source="news_agent",
                )
            ],
        )
        sec_result = SignalExtractionResult(
            source="sec_agent",
            overall_sentiment=SignalDirection.NEGATIVE,
            signals=[
                ExtractedSignal(
                    signal_type=SignalType.GUIDANCE,
                    direction=SignalDirection.NEGATIVE,
                    topic="guidance",
                    description="Lowered guidance",
                    source="sec_agent",
                )
            ],
        )

        result = detector.compare(news_result, sec_result)

        assert result.has_discrepancies is True
        # Guidance conflicts should be high severity
        high_severity = result.high_severity_discrepancies
        assert len(high_severity) >= 1

    # -------------------------------------------------------------------------
    # Risk mismatch detection tests
    # -------------------------------------------------------------------------

    def test_detect_risk_mismatch(self, detector: DiscrepancyDetector) -> None:
        """Risk mismatch should be detected when SEC shows risks but news is positive."""
        news_result = SignalExtractionResult(
            source="news_agent",
            overall_sentiment=SignalDirection.POSITIVE,
            signals=[
                ExtractedSignal(
                    signal_type=SignalType.SENTIMENT,
                    direction=SignalDirection.POSITIVE,
                    topic="outlook",
                    description="Positive",
                    source="news_agent",
                ),
                ExtractedSignal(
                    signal_type=SignalType.SENTIMENT,
                    direction=SignalDirection.POSITIVE,
                    topic="growth",
                    description="Strong growth",
                    source="news_agent",
                ),
                ExtractedSignal(
                    signal_type=SignalType.SENTIMENT,
                    direction=SignalDirection.POSITIVE,
                    topic="earnings",
                    description="Beat earnings",
                    source="news_agent",
                ),
            ],
        )
        sec_result = SignalExtractionResult(
            source="sec_agent",
            overall_sentiment=SignalDirection.NEUTRAL,
            signals=[
                ExtractedSignal(
                    signal_type=SignalType.RISK_FACTOR,
                    direction=SignalDirection.NEGATIVE,
                    topic="regulatory_risk",
                    description="Regulatory risk",
                    source="sec_agent",
                ),
                ExtractedSignal(
                    signal_type=SignalType.RISK_FACTOR,
                    direction=SignalDirection.NEGATIVE,
                    topic="market_risk",
                    description="Market risk",
                    source="sec_agent",
                ),
            ],
        )

        result = detector.compare(news_result, sec_result)

        assert result.has_discrepancies is True
        risk_mismatches = [
            d for d in result.discrepancies
            if d.discrepancy_type == DiscrepancyType.RISK_MISMATCH
        ]
        assert len(risk_mismatches) >= 1

    def test_no_risk_mismatch_when_news_balanced(self, detector: DiscrepancyDetector) -> None:
        """No risk mismatch when news sentiment is balanced."""
        news_result = SignalExtractionResult(
            source="news_agent",
            overall_sentiment=SignalDirection.MIXED,
            signals=[
                ExtractedSignal(
                    signal_type=SignalType.SENTIMENT,
                    direction=SignalDirection.POSITIVE,
                    topic="outlook",
                    description="Positive",
                    source="news_agent",
                ),
                ExtractedSignal(
                    signal_type=SignalType.SENTIMENT,
                    direction=SignalDirection.NEGATIVE,
                    topic="concerns",
                    description="Concerns noted",
                    source="news_agent",
                ),
            ],
        )
        sec_result = SignalExtractionResult(
            source="sec_agent",
            overall_sentiment=SignalDirection.NEUTRAL,
            signals=[
                ExtractedSignal(
                    signal_type=SignalType.RISK_FACTOR,
                    direction=SignalDirection.NEGATIVE,
                    topic="regulatory_risk",
                    description="Regulatory risk",
                    source="sec_agent",
                ),
                ExtractedSignal(
                    signal_type=SignalType.RISK_FACTOR,
                    direction=SignalDirection.NEGATIVE,
                    topic="market_risk",
                    description="Market risk",
                    source="sec_agent",
                ),
            ],
        )

        result = detector.compare(news_result, sec_result)

        risk_mismatches = [
            d for d in result.discrepancies
            if d.discrepancy_type == DiscrepancyType.RISK_MISMATCH
        ]
        assert len(risk_mismatches) == 0

    # -------------------------------------------------------------------------
    # Agreement detection tests
    # -------------------------------------------------------------------------

    def test_detect_overall_sentiment_agreement(self, detector: DiscrepancyDetector) -> None:
        """Overall sentiment agreement should be detected."""
        news_result = SignalExtractionResult(
            source="news_agent",
            overall_sentiment=SignalDirection.POSITIVE,
            signals=[
                ExtractedSignal(
                    signal_type=SignalType.SENTIMENT,
                    direction=SignalDirection.POSITIVE,
                    topic="outlook",
                    description="Positive outlook",
                    source="news_agent",
                )
            ],
        )
        sec_result = SignalExtractionResult(
            source="sec_agent",
            overall_sentiment=SignalDirection.POSITIVE,
            signals=[
                ExtractedSignal(
                    signal_type=SignalType.SENTIMENT,
                    direction=SignalDirection.POSITIVE,
                    topic="outlook",
                    description="Positive outlook",
                    source="sec_agent",
                )
            ],
        )

        result = detector.compare(news_result, sec_result)

        assert result.has_agreements is True
        assert result.agreement_count >= 1

    def test_detect_topic_agreement(self, detector: DiscrepancyDetector) -> None:
        """Agreement on specific topic should be detected."""
        news_result = SignalExtractionResult(
            source="news_agent",
            overall_sentiment=SignalDirection.POSITIVE,
            signals=[
                ExtractedSignal(
                    signal_type=SignalType.FINANCIAL_METRIC,
                    direction=SignalDirection.POSITIVE,
                    topic="revenue",
                    description="Revenue up",
                    source="news_agent",
                )
            ],
        )
        sec_result = SignalExtractionResult(
            source="sec_agent",
            overall_sentiment=SignalDirection.POSITIVE,
            signals=[
                ExtractedSignal(
                    signal_type=SignalType.FINANCIAL_METRIC,
                    direction=SignalDirection.POSITIVE,
                    topic="revenue",
                    description="Revenue increased",
                    source="sec_agent",
                )
            ],
        )

        result = detector.compare(news_result, sec_result)

        assert result.has_agreements is True
        agreements = result.agreements
        assert any(a.topic == "revenue" for a in agreements)

    # -------------------------------------------------------------------------
    # Alignment score tests
    # -------------------------------------------------------------------------

    def test_positive_alignment_score(self, detector: DiscrepancyDetector) -> None:
        """Agreement-heavy results should have positive alignment."""
        news_result = SignalExtractionResult(
            source="news_agent",
            overall_sentiment=SignalDirection.POSITIVE,
            signals=[
                ExtractedSignal(
                    signal_type=SignalType.SENTIMENT,
                    direction=SignalDirection.POSITIVE,
                    topic="outlook",
                    description="Positive",
                    source="news_agent",
                )
            ],
        )
        sec_result = SignalExtractionResult(
            source="sec_agent",
            overall_sentiment=SignalDirection.POSITIVE,
            signals=[
                ExtractedSignal(
                    signal_type=SignalType.SENTIMENT,
                    direction=SignalDirection.POSITIVE,
                    topic="outlook",
                    description="Positive",
                    source="sec_agent",
                )
            ],
        )

        result = detector.compare(news_result, sec_result)

        assert result.overall_alignment > 0

    def test_negative_alignment_score(self, detector: DiscrepancyDetector) -> None:
        """Discrepancy-heavy results should have negative alignment."""
        news_result = SignalExtractionResult(
            source="news_agent",
            overall_sentiment=SignalDirection.POSITIVE,
            signals=[
                ExtractedSignal(
                    signal_type=SignalType.SENTIMENT,
                    direction=SignalDirection.POSITIVE,
                    topic="outlook",
                    description="Positive",
                    source="news_agent",
                )
            ],
        )
        sec_result = SignalExtractionResult(
            source="sec_agent",
            overall_sentiment=SignalDirection.NEGATIVE,
            signals=[
                ExtractedSignal(
                    signal_type=SignalType.SENTIMENT,
                    direction=SignalDirection.NEGATIVE,
                    topic="outlook",
                    description="Negative",
                    source="sec_agent",
                )
            ],
        )

        result = detector.compare(news_result, sec_result)

        assert result.overall_alignment < 0

    def test_alignment_score_in_range(self, detector: DiscrepancyDetector) -> None:
        """Alignment score should always be between -1.0 and 1.0."""
        news_result = SignalExtractionResult(
            source="news_agent",
            overall_sentiment=SignalDirection.POSITIVE,
            signals=[
                ExtractedSignal(
                    signal_type=SignalType.SENTIMENT,
                    direction=SignalDirection.POSITIVE,
                    topic="outlook",
                    description="Positive",
                    source="news_agent",
                    confidence=0.95,
                )
            ] * 5,  # Multiple positive signals
        )
        sec_result = SignalExtractionResult(
            source="sec_agent",
            overall_sentiment=SignalDirection.NEGATIVE,
            signals=[
                ExtractedSignal(
                    signal_type=SignalType.SENTIMENT,
                    direction=SignalDirection.NEGATIVE,
                    topic="outlook",
                    description="Negative",
                    source="sec_agent",
                    confidence=0.95,
                )
            ] * 5,  # Multiple negative signals
        )

        result = detector.compare(news_result, sec_result)

        assert -1.0 <= result.overall_alignment <= 1.0

    # -------------------------------------------------------------------------
    # Critical discrepancy flag tests
    # -------------------------------------------------------------------------

    def test_has_critical_discrepancies_true(self, detector: DiscrepancyDetector) -> None:
        """has_critical_discrepancies should be True when HIGH severity exists."""
        news_result = SignalExtractionResult(
            source="news_agent",
            overall_sentiment=SignalDirection.POSITIVE,
            signals=[
                ExtractedSignal(
                    signal_type=SignalType.FINANCIAL_METRIC,
                    direction=SignalDirection.POSITIVE,
                    topic="revenue",
                    description="Revenue up",
                    source="news_agent",
                )
            ],
        )
        sec_result = SignalExtractionResult(
            source="sec_agent",
            overall_sentiment=SignalDirection.NEGATIVE,
            signals=[
                ExtractedSignal(
                    signal_type=SignalType.FINANCIAL_METRIC,
                    direction=SignalDirection.NEGATIVE,
                    topic="revenue",
                    description="Revenue down",
                    source="sec_agent",
                )
            ],
        )

        result = detector.compare(news_result, sec_result)

        # Financial conflicts are HIGH severity
        assert result.has_critical_discrepancies is True

    def test_has_critical_discrepancies_false(self, detector: DiscrepancyDetector) -> None:
        """has_critical_discrepancies should be False when no HIGH severity."""
        news_result = SignalExtractionResult(
            source="news_agent",
            overall_sentiment=SignalDirection.POSITIVE,
            signals=[
                ExtractedSignal(
                    signal_type=SignalType.SENTIMENT,
                    direction=SignalDirection.POSITIVE,
                    topic="outlook",
                    description="Positive",
                    source="news_agent",
                )
            ],
        )
        sec_result = SignalExtractionResult(
            source="sec_agent",
            overall_sentiment=SignalDirection.POSITIVE,
            signals=[
                ExtractedSignal(
                    signal_type=SignalType.SENTIMENT,
                    direction=SignalDirection.POSITIVE,
                    topic="outlook",
                    description="Positive",
                    source="sec_agent",
                )
            ],
        )

        result = detector.compare(news_result, sec_result)

        assert result.has_critical_discrepancies is False

    # -------------------------------------------------------------------------
    # Summary generation tests
    # -------------------------------------------------------------------------

    def test_summary_includes_sentiment(self, detector: DiscrepancyDetector) -> None:
        """Summary should include sentiment comparison."""
        news_result = SignalExtractionResult(
            source="news_agent",
            overall_sentiment=SignalDirection.POSITIVE,
        )
        sec_result = SignalExtractionResult(
            source="sec_agent",
            overall_sentiment=SignalDirection.NEGATIVE,
        )

        result = detector.compare(news_result, sec_result)

        assert "positive" in result.summary.lower()
        assert "negative" in result.summary.lower()

    def test_summary_includes_discrepancy_count(self, detector: DiscrepancyDetector) -> None:
        """Summary should mention discrepancies when present."""
        news_result = SignalExtractionResult(
            source="news_agent",
            overall_sentiment=SignalDirection.POSITIVE,
            signals=[
                ExtractedSignal(
                    signal_type=SignalType.FINANCIAL_METRIC,
                    direction=SignalDirection.POSITIVE,
                    topic="revenue",
                    description="Up",
                    source="news_agent",
                )
            ],
        )
        sec_result = SignalExtractionResult(
            source="sec_agent",
            overall_sentiment=SignalDirection.NEGATIVE,
            signals=[
                ExtractedSignal(
                    signal_type=SignalType.FINANCIAL_METRIC,
                    direction=SignalDirection.NEGATIVE,
                    topic="revenue",
                    description="Down",
                    source="sec_agent",
                )
            ],
        )

        result = detector.compare(news_result, sec_result)

        assert "discrepancy" in result.summary.lower() or "discrepancies" in result.summary.lower()

    def test_summary_no_discrepancies(self, detector: DiscrepancyDetector) -> None:
        """Summary should indicate no discrepancies when none found."""
        news_result = SignalExtractionResult(
            source="news_agent",
            overall_sentiment=SignalDirection.POSITIVE,
        )
        sec_result = SignalExtractionResult(
            source="sec_agent",
            overall_sentiment=SignalDirection.POSITIVE,
        )

        result = detector.compare(news_result, sec_result)

        assert "No discrepancies detected" in result.summary

    # -------------------------------------------------------------------------
    # Edge cases
    # -------------------------------------------------------------------------

    def test_empty_signals_both_sources(self, detector: DiscrepancyDetector) -> None:
        """Both sources with no signals should result in no discrepancies."""
        news_result = SignalExtractionResult(source="news_agent")
        sec_result = SignalExtractionResult(source="sec_agent")

        result = detector.compare(news_result, sec_result)

        assert result.has_discrepancies is False
        assert result.has_agreements is False
        assert result.overall_alignment == 0.0

    def test_multiple_discrepancies_same_type(self, detector: DiscrepancyDetector) -> None:
        """Multiple discrepancies of same type should all be captured."""
        news_result = SignalExtractionResult(
            source="news_agent",
            overall_sentiment=SignalDirection.POSITIVE,
            signals=[
                ExtractedSignal(
                    signal_type=SignalType.FINANCIAL_METRIC,
                    direction=SignalDirection.POSITIVE,
                    topic="revenue",
                    description="Revenue up",
                    source="news_agent",
                ),
                ExtractedSignal(
                    signal_type=SignalType.FINANCIAL_METRIC,
                    direction=SignalDirection.POSITIVE,
                    topic="earnings",
                    description="Earnings up",
                    source="news_agent",
                ),
            ],
        )
        sec_result = SignalExtractionResult(
            source="sec_agent",
            overall_sentiment=SignalDirection.NEGATIVE,
            signals=[
                ExtractedSignal(
                    signal_type=SignalType.FINANCIAL_METRIC,
                    direction=SignalDirection.NEGATIVE,
                    topic="revenue",
                    description="Revenue down",
                    source="sec_agent",
                ),
                ExtractedSignal(
                    signal_type=SignalType.FINANCIAL_METRIC,
                    direction=SignalDirection.NEGATIVE,
                    topic="earnings",
                    description="Earnings down",
                    source="sec_agent",
                ),
            ],
        )

        result = detector.compare(news_result, sec_result)

        financial_conflicts = result.financial_discrepancies
        # Should detect conflicts on both revenue and earnings
        assert len(financial_conflicts) >= 2

    def test_mixed_discrepancies_and_agreements(self, detector: DiscrepancyDetector) -> None:
        """Results with both discrepancies and agreements should track both."""
        news_result = SignalExtractionResult(
            source="news_agent",
            overall_sentiment=SignalDirection.MIXED,
            signals=[
                ExtractedSignal(
                    signal_type=SignalType.FINANCIAL_METRIC,
                    direction=SignalDirection.POSITIVE,
                    topic="revenue",
                    description="Revenue up",
                    source="news_agent",
                ),
                ExtractedSignal(
                    signal_type=SignalType.SENTIMENT,
                    direction=SignalDirection.NEGATIVE,
                    topic="outlook",
                    description="Cautious outlook",
                    source="news_agent",
                ),
            ],
        )
        sec_result = SignalExtractionResult(
            source="sec_agent",
            overall_sentiment=SignalDirection.MIXED,
            signals=[
                ExtractedSignal(
                    signal_type=SignalType.FINANCIAL_METRIC,
                    direction=SignalDirection.POSITIVE,
                    topic="revenue",
                    description="Revenue increased",
                    source="sec_agent",
                ),
                ExtractedSignal(
                    signal_type=SignalType.SENTIMENT,
                    direction=SignalDirection.POSITIVE,
                    topic="outlook",
                    description="Positive outlook",
                    source="sec_agent",
                ),
            ],
        )

        result = detector.compare(news_result, sec_result)

        # Should have agreement on revenue
        assert result.has_agreements is True
        # Should have discrepancy on outlook
        assert result.has_discrepancies is True
