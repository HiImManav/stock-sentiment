"""Discrepancy detection between news and SEC agent signals.

This module compares signals extracted from news_agent and sec_agent
to identify agreements, discrepancies, and conflicts between sources.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from src.orchestrator.comparison.signals import (
    ExtractedSignal,
    SignalDirection,
    SignalExtractionResult,
    SignalType,
)


class DiscrepancyType(Enum):
    """Type of discrepancy between sources."""

    SENTIMENT_CONFLICT = "sentiment_conflict"
    FINANCIAL_CONFLICT = "financial_conflict"
    GUIDANCE_CONFLICT = "guidance_conflict"
    RISK_MISMATCH = "risk_mismatch"
    TOPIC_DISAGREEMENT = "topic_disagreement"


class DiscrepancySeverity(Enum):
    """Severity level of a discrepancy."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class Discrepancy:
    """A detected discrepancy between news and SEC signals.

    Attributes:
        discrepancy_type: The category of discrepancy.
        severity: How significant the discrepancy is.
        topic: The topic where discrepancy was found.
        news_signal: The signal from news_agent (if any).
        sec_signal: The signal from sec_agent (if any).
        description: Human-readable description of the discrepancy.
        confidence: Confidence in the discrepancy detection (0.0 to 1.0).
    """

    discrepancy_type: DiscrepancyType
    severity: DiscrepancySeverity
    topic: str
    news_signal: ExtractedSignal | None
    sec_signal: ExtractedSignal | None
    description: str
    confidence: float = 0.8

    def __post_init__(self) -> None:
        """Validate confidence is in range."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")


@dataclass
class Agreement:
    """An agreement between news and SEC signals.

    Attributes:
        topic: The topic where agreement was found.
        direction: The shared direction/sentiment.
        news_signal: The signal from news_agent.
        sec_signal: The signal from sec_agent.
        description: Human-readable description of the agreement.
        confidence: Combined confidence from both signals.
    """

    topic: str
    direction: SignalDirection
    news_signal: ExtractedSignal
    sec_signal: ExtractedSignal
    description: str
    confidence: float = 0.8


@dataclass
class ComparisonResult:
    """Result of comparing news and SEC agent signals.

    Attributes:
        news_result: The signal extraction result from news_agent.
        sec_result: The signal extraction result from sec_agent.
        discrepancies: List of detected discrepancies.
        agreements: List of detected agreements.
        overall_alignment: How aligned the sources are (-1.0 to 1.0).
        has_critical_discrepancies: Whether any high-severity discrepancies exist.
        summary: Brief summary of the comparison.
    """

    news_result: SignalExtractionResult | None
    sec_result: SignalExtractionResult | None
    discrepancies: list[Discrepancy] = field(default_factory=list)
    agreements: list[Agreement] = field(default_factory=list)
    overall_alignment: float = 0.0
    has_critical_discrepancies: bool = False
    summary: str = ""

    @property
    def has_discrepancies(self) -> bool:
        """Check if any discrepancies were found."""
        return len(self.discrepancies) > 0

    @property
    def has_agreements(self) -> bool:
        """Check if any agreements were found."""
        return len(self.agreements) > 0

    @property
    def discrepancy_count(self) -> int:
        """Return the number of discrepancies."""
        return len(self.discrepancies)

    @property
    def agreement_count(self) -> int:
        """Return the number of agreements."""
        return len(self.agreements)

    @property
    def high_severity_discrepancies(self) -> list[Discrepancy]:
        """Return only high-severity discrepancies."""
        return [d for d in self.discrepancies if d.severity == DiscrepancySeverity.HIGH]

    @property
    def sentiment_discrepancies(self) -> list[Discrepancy]:
        """Return only sentiment-related discrepancies."""
        return [d for d in self.discrepancies if d.discrepancy_type == DiscrepancyType.SENTIMENT_CONFLICT]

    @property
    def financial_discrepancies(self) -> list[Discrepancy]:
        """Return only financial-related discrepancies."""
        return [d for d in self.discrepancies if d.discrepancy_type == DiscrepancyType.FINANCIAL_CONFLICT]


# Topic categories for grouping related topics
SENTIMENT_TOPICS = {"sentiment", "outlook", "rating", "growth", "performance"}
FINANCIAL_TOPICS = {"revenue", "earnings", "margin", "cash_flow"}
GUIDANCE_TOPICS = {"guidance", "forward_looking", "forecast"}
RISK_TOPICS = {"general_risk", "regulatory_risk", "market_risk", "operational_risk", "cyber_risk", "control_risk", "risk"}


def _topics_are_related(topic1: str, topic2: str) -> bool:
    """Check if two topics are related and comparable."""
    # Exact match
    if topic1 == topic2:
        return True

    # Check if both in same category
    for category in [SENTIMENT_TOPICS, FINANCIAL_TOPICS, GUIDANCE_TOPICS, RISK_TOPICS]:
        if topic1 in category and topic2 in category:
            return True

    return False


def _directions_conflict(dir1: SignalDirection, dir2: SignalDirection) -> bool:
    """Check if two signal directions are in conflict."""
    conflicting_pairs = {
        (SignalDirection.POSITIVE, SignalDirection.NEGATIVE),
        (SignalDirection.NEGATIVE, SignalDirection.POSITIVE),
    }
    return (dir1, dir2) in conflicting_pairs


def _directions_agree(dir1: SignalDirection, dir2: SignalDirection) -> bool:
    """Check if two signal directions agree."""
    return dir1 == dir2 and dir1 in (SignalDirection.POSITIVE, SignalDirection.NEGATIVE, SignalDirection.NEUTRAL)


def _calculate_discrepancy_severity(
    news_signal: ExtractedSignal | None,
    sec_signal: ExtractedSignal | None,
    discrepancy_type: DiscrepancyType,
) -> DiscrepancySeverity:
    """Determine the severity of a discrepancy."""
    # Financial and guidance conflicts are more severe
    if discrepancy_type in (DiscrepancyType.FINANCIAL_CONFLICT, DiscrepancyType.GUIDANCE_CONFLICT):
        return DiscrepancySeverity.HIGH

    # If both signals have high confidence, the conflict is more severe
    if news_signal and sec_signal:
        avg_confidence = (news_signal.confidence + sec_signal.confidence) / 2
        if avg_confidence >= 0.85:
            return DiscrepancySeverity.HIGH
        elif avg_confidence >= 0.7:
            return DiscrepancySeverity.MEDIUM

    # Sentiment conflicts on key topics are medium severity
    if discrepancy_type == DiscrepancyType.SENTIMENT_CONFLICT:
        return DiscrepancySeverity.MEDIUM

    return DiscrepancySeverity.LOW


class DiscrepancyDetector:
    """Detects discrepancies and agreements between news and SEC signals.

    Compares signals extracted from both sources to identify:
    - Sentiment conflicts (news positive, SEC negative or vice versa)
    - Financial metric conflicts (different views on revenue/earnings)
    - Guidance conflicts (optimistic news vs cautious SEC filings)
    - Risk mismatches (risks mentioned in one source but not reflected in the other)
    """

    def compare(
        self,
        news_result: SignalExtractionResult | None,
        sec_result: SignalExtractionResult | None,
    ) -> ComparisonResult:
        """Compare signals from news and SEC agents.

        Args:
            news_result: Signal extraction result from news_agent (or None if not available).
            sec_result: Signal extraction result from sec_agent (or None if not available).

        Returns:
            ComparisonResult containing discrepancies and agreements.
        """
        # Handle cases where one or both results are missing
        if news_result is None and sec_result is None:
            return ComparisonResult(
                news_result=None,
                sec_result=None,
                summary="No signals available from either source.",
            )

        if news_result is None or not news_result.extraction_successful:
            return ComparisonResult(
                news_result=news_result,
                sec_result=sec_result,
                overall_alignment=0.0,
                summary="News agent signals unavailable; comparison limited to SEC data only.",
            )

        if sec_result is None or not sec_result.extraction_successful:
            return ComparisonResult(
                news_result=news_result,
                sec_result=sec_result,
                overall_alignment=0.0,
                summary="SEC agent signals unavailable; comparison limited to news data only.",
            )

        discrepancies: list[Discrepancy] = []
        agreements: list[Agreement] = []

        # Detect overall sentiment conflict
        discrepancies.extend(self._detect_sentiment_conflicts(news_result, sec_result))

        # Detect financial metric conflicts
        discrepancies.extend(self._detect_financial_conflicts(news_result, sec_result))

        # Detect guidance conflicts
        discrepancies.extend(self._detect_guidance_conflicts(news_result, sec_result))

        # Detect risk mismatches
        discrepancies.extend(self._detect_risk_mismatches(news_result, sec_result))

        # Find agreements
        agreements.extend(self._find_agreements(news_result, sec_result))

        # Calculate overall alignment score
        overall_alignment = self._calculate_alignment(discrepancies, agreements)

        # Check for critical discrepancies
        has_critical = any(d.severity == DiscrepancySeverity.HIGH for d in discrepancies)

        # Generate summary
        summary = self._generate_summary(discrepancies, agreements, news_result, sec_result)

        return ComparisonResult(
            news_result=news_result,
            sec_result=sec_result,
            discrepancies=discrepancies,
            agreements=agreements,
            overall_alignment=overall_alignment,
            has_critical_discrepancies=has_critical,
            summary=summary,
        )

    def _detect_sentiment_conflicts(
        self,
        news_result: SignalExtractionResult,
        sec_result: SignalExtractionResult,
    ) -> list[Discrepancy]:
        """Detect conflicts in overall sentiment between sources."""
        discrepancies: list[Discrepancy] = []

        # Check overall sentiment direction conflict
        if _directions_conflict(news_result.overall_sentiment, sec_result.overall_sentiment):
            # Find representative signals for each
            news_signal = news_result.sentiment_signals[0] if news_result.sentiment_signals else None
            sec_signal = sec_result.sentiment_signals[0] if sec_result.sentiment_signals else None

            severity = _calculate_discrepancy_severity(
                news_signal, sec_signal, DiscrepancyType.SENTIMENT_CONFLICT
            )

            discrepancies.append(
                Discrepancy(
                    discrepancy_type=DiscrepancyType.SENTIMENT_CONFLICT,
                    severity=severity,
                    topic="overall_sentiment",
                    news_signal=news_signal,
                    sec_signal=sec_signal,
                    description=(
                        f"News sentiment is {news_result.overall_sentiment.value} "
                        f"while SEC sentiment is {sec_result.overall_sentiment.value}."
                    ),
                    confidence=0.9,
                )
            )

        # Check for signal-level sentiment conflicts on related topics
        for news_signal in news_result.sentiment_signals:
            for sec_signal in sec_result.sentiment_signals:
                if _topics_are_related(news_signal.topic, sec_signal.topic):
                    if _directions_conflict(news_signal.direction, sec_signal.direction):
                        severity = _calculate_discrepancy_severity(
                            news_signal, sec_signal, DiscrepancyType.SENTIMENT_CONFLICT
                        )
                        discrepancies.append(
                            Discrepancy(
                                discrepancy_type=DiscrepancyType.SENTIMENT_CONFLICT,
                                severity=severity,
                                topic=news_signal.topic,
                                news_signal=news_signal,
                                sec_signal=sec_signal,
                                description=(
                                    f"News indicates {news_signal.direction.value} {news_signal.topic}, "
                                    f"but SEC indicates {sec_signal.direction.value} {sec_signal.topic}."
                                ),
                                confidence=(news_signal.confidence + sec_signal.confidence) / 2,
                            )
                        )

        return discrepancies

    def _detect_financial_conflicts(
        self,
        news_result: SignalExtractionResult,
        sec_result: SignalExtractionResult,
    ) -> list[Discrepancy]:
        """Detect conflicts in financial metrics between sources."""
        discrepancies: list[Discrepancy] = []

        news_financial = news_result.financial_signals
        sec_financial = sec_result.financial_signals

        for news_signal in news_financial:
            for sec_signal in sec_financial:
                if _topics_are_related(news_signal.topic, sec_signal.topic):
                    if _directions_conflict(news_signal.direction, sec_signal.direction):
                        discrepancies.append(
                            Discrepancy(
                                discrepancy_type=DiscrepancyType.FINANCIAL_CONFLICT,
                                severity=DiscrepancySeverity.HIGH,
                                topic=news_signal.topic,
                                news_signal=news_signal,
                                sec_signal=sec_signal,
                                description=(
                                    f"News reports {news_signal.direction.value} {news_signal.topic}, "
                                    f"but SEC filings show {sec_signal.direction.value} {sec_signal.topic}."
                                ),
                                confidence=(news_signal.confidence + sec_signal.confidence) / 2,
                            )
                        )

        return discrepancies

    def _detect_guidance_conflicts(
        self,
        news_result: SignalExtractionResult,
        sec_result: SignalExtractionResult,
    ) -> list[Discrepancy]:
        """Detect conflicts in forward guidance between sources."""
        discrepancies: list[Discrepancy] = []

        news_guidance = [s for s in news_result.signals if s.signal_type == SignalType.GUIDANCE]
        sec_guidance = [s for s in sec_result.signals if s.signal_type == SignalType.GUIDANCE]

        for news_signal in news_guidance:
            for sec_signal in sec_guidance:
                if _topics_are_related(news_signal.topic, sec_signal.topic):
                    if _directions_conflict(news_signal.direction, sec_signal.direction):
                        discrepancies.append(
                            Discrepancy(
                                discrepancy_type=DiscrepancyType.GUIDANCE_CONFLICT,
                                severity=DiscrepancySeverity.HIGH,
                                topic=news_signal.topic,
                                news_signal=news_signal,
                                sec_signal=sec_signal,
                                description=(
                                    f"News reports {news_signal.direction.value} guidance, "
                                    f"but SEC filings indicate {sec_signal.direction.value} guidance."
                                ),
                                confidence=(news_signal.confidence + sec_signal.confidence) / 2,
                            )
                        )

        return discrepancies

    def _detect_risk_mismatches(
        self,
        news_result: SignalExtractionResult,
        sec_result: SignalExtractionResult,
    ) -> list[Discrepancy]:
        """Detect risk factor mismatches between sources.

        Flags cases where SEC mentions significant risks that are not
        reflected in the news sentiment (e.g., SEC has many risks but
        news is overwhelmingly positive).
        """
        discrepancies: list[Discrepancy] = []

        sec_risks = sec_result.risk_signals
        news_positive_count = len(news_result.positive_signals)
        news_negative_count = len(news_result.negative_signals)

        # If SEC shows significant risks but news is overwhelmingly positive
        if len(sec_risks) >= 2 and news_positive_count > news_negative_count * 2:
            # Get most significant risk signal
            risk_signal = sec_risks[0] if sec_risks else None
            news_signal = news_result.positive_signals[0] if news_result.positive_signals else None

            discrepancies.append(
                Discrepancy(
                    discrepancy_type=DiscrepancyType.RISK_MISMATCH,
                    severity=DiscrepancySeverity.MEDIUM,
                    topic="risk_sentiment_gap",
                    news_signal=news_signal,
                    sec_signal=risk_signal,
                    description=(
                        f"SEC filings highlight {len(sec_risks)} risk factors, "
                        f"but news sentiment is predominantly positive ({news_positive_count} positive vs {news_negative_count} negative signals)."
                    ),
                    confidence=0.75,
                )
            )

        return discrepancies

    def _find_agreements(
        self,
        news_result: SignalExtractionResult,
        sec_result: SignalExtractionResult,
    ) -> list[Agreement]:
        """Find agreements between news and SEC signals."""
        agreements: list[Agreement] = []

        # Check overall sentiment agreement
        if _directions_agree(news_result.overall_sentiment, sec_result.overall_sentiment):
            news_signal = news_result.sentiment_signals[0] if news_result.sentiment_signals else None
            sec_signal = sec_result.sentiment_signals[0] if sec_result.sentiment_signals else None

            if news_signal and sec_signal:
                agreements.append(
                    Agreement(
                        topic="overall_sentiment",
                        direction=news_result.overall_sentiment,
                        news_signal=news_signal,
                        sec_signal=sec_signal,
                        description=(
                            f"Both sources show {news_result.overall_sentiment.value} sentiment."
                        ),
                        confidence=(news_signal.confidence + sec_signal.confidence) / 2,
                    )
                )

        # Check for signal-level agreements
        for news_signal in news_result.signals:
            for sec_signal in sec_result.signals:
                if _topics_are_related(news_signal.topic, sec_signal.topic):
                    if _directions_agree(news_signal.direction, sec_signal.direction):
                        # Avoid duplicating overall sentiment agreement
                        if news_signal.topic == "sentiment" and sec_signal.topic == "sentiment":
                            continue
                        agreements.append(
                            Agreement(
                                topic=news_signal.topic,
                                direction=news_signal.direction,
                                news_signal=news_signal,
                                sec_signal=sec_signal,
                                description=(
                                    f"Both sources agree on {news_signal.direction.value} {news_signal.topic}."
                                ),
                                confidence=(news_signal.confidence + sec_signal.confidence) / 2,
                            )
                        )

        return agreements

    def _calculate_alignment(
        self,
        discrepancies: list[Discrepancy],
        agreements: list[Agreement],
    ) -> float:
        """Calculate overall alignment score between -1.0 and 1.0.

        Returns:
            Score where 1.0 = perfect agreement, -1.0 = complete conflict,
            0.0 = neutral/no overlap.
        """
        if not discrepancies and not agreements:
            return 0.0

        # Weight agreements positively, discrepancies negatively
        agreement_score = sum(a.confidence for a in agreements)
        discrepancy_score = sum(d.confidence for d in discrepancies)

        # Extra penalty for high-severity discrepancies
        for d in discrepancies:
            if d.severity == DiscrepancySeverity.HIGH:
                discrepancy_score += 0.5

        total = agreement_score + discrepancy_score
        if total == 0:
            return 0.0

        # Calculate normalized score
        alignment = (agreement_score - discrepancy_score) / total

        # Clamp to [-1.0, 1.0]
        return max(-1.0, min(1.0, round(alignment, 2)))

    def _generate_summary(
        self,
        discrepancies: list[Discrepancy],
        agreements: list[Agreement],
        news_result: SignalExtractionResult,
        sec_result: SignalExtractionResult,
    ) -> str:
        """Generate a brief summary of the comparison."""
        parts: list[str] = []

        # Overall sentiment comparison
        if news_result.overall_sentiment == sec_result.overall_sentiment:
            parts.append(
                f"Both sources show {news_result.overall_sentiment.value} sentiment."
            )
        else:
            parts.append(
                f"News is {news_result.overall_sentiment.value}, SEC is {sec_result.overall_sentiment.value}."
            )

        # Discrepancy summary
        if discrepancies:
            high_count = len([d for d in discrepancies if d.severity == DiscrepancySeverity.HIGH])
            if high_count > 0:
                parts.append(f"Found {high_count} high-severity discrepancy(ies).")
            else:
                parts.append(f"Found {len(discrepancies)} discrepancy(ies).")
        else:
            parts.append("No discrepancies detected.")

        # Agreement summary
        if agreements:
            parts.append(f"Found {len(agreements)} agreement(s).")

        return " ".join(parts)
