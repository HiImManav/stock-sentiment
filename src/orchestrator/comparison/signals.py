"""Signal extraction from agent responses.

This module extracts structured signals from news and SEC agent responses
to enable comparison and discrepancy detection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class SignalType(Enum):
    """Type of extracted signal."""

    SENTIMENT = "sentiment"
    FINANCIAL_METRIC = "financial_metric"
    RISK_FACTOR = "risk_factor"
    GUIDANCE = "guidance"
    EVENT = "event"


class SignalDirection(Enum):
    """Direction or valence of a signal."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


@dataclass
class ExtractedSignal:
    """A signal extracted from an agent response.

    Attributes:
        signal_type: The category of this signal.
        direction: The positive/negative/neutral valence.
        topic: The subject of the signal (e.g., "revenue", "risk", "outlook").
        description: Brief description of the signal.
        confidence: Confidence in the extraction (0.0 to 1.0).
        source: Which agent produced this signal.
        raw_text: The original text from which this was extracted.
    """

    signal_type: SignalType
    direction: SignalDirection
    topic: str
    description: str
    confidence: float = 0.8
    source: Literal["news_agent", "sec_agent"] = "news_agent"
    raw_text: str = ""

    def __post_init__(self) -> None:
        """Validate confidence is in range."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")


@dataclass
class SignalExtractionResult:
    """Result of signal extraction from an agent response.

    Attributes:
        source: Which agent produced the response.
        signals: List of extracted signals.
        overall_sentiment: The dominant sentiment direction.
        sentiment_score: Numeric sentiment score (-1.0 to 1.0).
        topics: List of topics mentioned.
        ticker: Company ticker if detected.
        extraction_successful: Whether extraction was successful.
        error_message: Error message if extraction failed.
    """

    source: Literal["news_agent", "sec_agent"]
    signals: list[ExtractedSignal] = field(default_factory=list)
    overall_sentiment: SignalDirection = SignalDirection.NEUTRAL
    sentiment_score: float = 0.0
    topics: list[str] = field(default_factory=list)
    ticker: str | None = None
    extraction_successful: bool = True
    error_message: str | None = None

    @property
    def has_signals(self) -> bool:
        """Check if any signals were extracted."""
        return len(self.signals) > 0

    @property
    def positive_signals(self) -> list[ExtractedSignal]:
        """Return only positive signals."""
        return [s for s in self.signals if s.direction == SignalDirection.POSITIVE]

    @property
    def negative_signals(self) -> list[ExtractedSignal]:
        """Return only negative signals."""
        return [s for s in self.signals if s.direction == SignalDirection.NEGATIVE]

    @property
    def sentiment_signals(self) -> list[ExtractedSignal]:
        """Return only sentiment-type signals."""
        return [s for s in self.signals if s.signal_type == SignalType.SENTIMENT]

    @property
    def risk_signals(self) -> list[ExtractedSignal]:
        """Return only risk factor signals."""
        return [s for s in self.signals if s.signal_type == SignalType.RISK_FACTOR]

    @property
    def financial_signals(self) -> list[ExtractedSignal]:
        """Return only financial metric signals."""
        return [s for s in self.signals if s.signal_type == SignalType.FINANCIAL_METRIC]


class SignalExtractor:
    """Extracts structured signals from agent responses.

    Uses pattern matching to identify sentiment indicators, financial metrics,
    risk factors, and forward guidance from agent text responses.
    """

    # Sentiment patterns
    POSITIVE_PATTERNS: list[tuple[str, str]] = [
        (r"\b(strong|robust|solid)\s+(growth|performance|results?)\b", "growth"),
        (r"\b(beat|exceeded?|surpass(?:ed)?)\s+(expectations?|estimates?)\b", "earnings"),
        (r"\b(positive|bullish|optimistic)\s+(outlook|sentiment|tone)\b", "outlook"),
        (r"\bsentiment[:\s]+positive\b", "sentiment"),
        (r"\boverall[:\s]+positive\b", "sentiment"),
        (r"\b(upside|outperform|upgrade[ds]?)\b", "rating"),
        (r"\b(record|all-time)?\s*high\s+(revenue|earnings|profits?)\b", "earnings"),
        (r"\b(significant|substantial)\s+(increase|growth|gains?)\b", "growth"),
    ]

    NEGATIVE_PATTERNS: list[tuple[str, str]] = [
        (r"\b(weak|poor|disappointing)\s+(results?|performance|growth)\b", "performance"),
        (r"\b(missed?|below|under)\s+(expectations?|estimates?)\b", "earnings"),
        (r"\b(negative|bearish|pessimistic)\s+(outlook|sentiment|tone)\b", "outlook"),
        (r"\bsentiment[:\s]+negative\b", "sentiment"),
        (r"\boverall[:\s]+negative\b", "sentiment"),
        (r"\b(downside|underperform|downgrade[ds]?)\b", "rating"),
        (r"\b(decline[ds]?|drop(?:ped)?|fell?|decrease[ds]?)\s+(?:in\s+)?(revenue|earnings|profits?)\b", "earnings"),
        (r"\b(significant|substantial)\s+(decline|decrease|loss(?:es)?)\b", "performance"),
        (r"\b(concerns?|worries?|risks?)\s+(?:about|regarding|over)\b", "risk"),
    ]

    NEUTRAL_PATTERNS: list[tuple[str, str]] = [
        (r"\bsentiment[:\s]+neutral\b", "sentiment"),
        (r"\boverall[:\s]+neutral\b", "sentiment"),
        (r"\b(mixed|uncertain|unclear)\s+(outlook|sentiment|signals?)\b", "outlook"),
        (r"\b(in\s+line\s+with|met)\s+(expectations?|estimates?)\b", "earnings"),
        (r"\b(stable|steady|unchanged)\b", "performance"),
    ]

    # Financial metric patterns
    FINANCIAL_PATTERNS: list[tuple[str, str, str]] = [
        (r"\brevenue\s+(increased?|grew?|rose|up)\s+(\d+(?:\.\d+)?%?)", "revenue", "positive"),
        (r"\brevenue\s+(decreased?|declined?|fell?|down)\s+(\d+(?:\.\d+)?%?)", "revenue", "negative"),
        (r"\bearnings?\s+(increased?|grew?|rose|up)", "earnings", "positive"),
        (r"\bearnings?\s+(decreased?|declined?|fell?|down)", "earnings", "negative"),
        (r"\bprofit\s+margin\s+(improved?|increased?|expanded?)", "margin", "positive"),
        (r"\bprofit\s+margin\s+(declined?|decreased?|compressed?)", "margin", "negative"),
        (r"\bcash\s+flow\s+(strong|positive|improved?)", "cash_flow", "positive"),
        (r"\bcash\s+flow\s+(weak|negative|declined?)", "cash_flow", "negative"),
    ]

    # Risk factor patterns (primarily from SEC filings)
    RISK_PATTERNS: list[tuple[str, str]] = [
        (r"\brisk\s+factors?\b", "general_risk"),
        (r"\b(regulatory|legal|litigation)\s+(risk|concern|issue)s?\b", "regulatory_risk"),
        (r"\b(market|competition|competitive)\s+(risk|pressure)s?\b", "market_risk"),
        (r"\b(supply\s+chain|operational)\s+(risk|disruption)s?\b", "operational_risk"),
        (r"\b(cybersecurity|data\s+breach|security)\s+(risk|threat)s?\b", "cyber_risk"),
        (r"\b(material\s+weakness|internal\s+control)\s+(issue|deficiency)?\b", "control_risk"),
    ]

    # Guidance patterns
    GUIDANCE_PATTERNS: list[tuple[str, str, str]] = [
        (r"\b(raised?|increased?|upgraded?)\s+(guidance|forecast|outlook)\b", "guidance", "positive"),
        (r"\b(lowered?|reduced?|cut)\s+(guidance|forecast|outlook)\b", "guidance", "negative"),
        (r"\b(reaffirmed?|maintained?|confirmed?)\s+(guidance|forecast|outlook)\b", "guidance", "neutral"),
        (r"\bforward[- ]looking\s+(positive|optimistic)\b", "forward_looking", "positive"),
        (r"\bforward[- ]looking\s+(cautious|conservative|negative)\b", "forward_looking", "negative"),
    ]

    # Ticker pattern
    TICKER_PATTERN = re.compile(r"\b([A-Z]{1,5})\b(?:\s+\(|:|\s+stock|\s+shares)")

    def __init__(self) -> None:
        """Initialize the signal extractor with compiled patterns."""
        self._positive_compiled = [
            (re.compile(p, re.IGNORECASE), t) for p, t in self.POSITIVE_PATTERNS
        ]
        self._negative_compiled = [
            (re.compile(p, re.IGNORECASE), t) for p, t in self.NEGATIVE_PATTERNS
        ]
        self._neutral_compiled = [
            (re.compile(p, re.IGNORECASE), t) for p, t in self.NEUTRAL_PATTERNS
        ]
        self._financial_compiled = [
            (re.compile(p, re.IGNORECASE), t, d) for p, t, d in self.FINANCIAL_PATTERNS
        ]
        self._risk_compiled = [
            (re.compile(p, re.IGNORECASE), t) for p, t in self.RISK_PATTERNS
        ]
        self._guidance_compiled = [
            (re.compile(p, re.IGNORECASE), t, d) for p, t, d in self.GUIDANCE_PATTERNS
        ]

    def extract(
        self,
        response: str,
        source: Literal["news_agent", "sec_agent"],
    ) -> SignalExtractionResult:
        """Extract signals from an agent response.

        Args:
            response: The text response from an agent.
            source: Which agent produced the response.

        Returns:
            SignalExtractionResult containing all extracted signals.
        """
        if not response or not response.strip():
            return SignalExtractionResult(
                source=source,
                extraction_successful=False,
                error_message="Empty response",
            )

        signals: list[ExtractedSignal] = []
        topics: set[str] = set()

        # Extract sentiment signals
        signals.extend(self._extract_sentiment_signals(response, source))

        # Extract financial signals
        signals.extend(self._extract_financial_signals(response, source))

        # Extract risk signals
        signals.extend(self._extract_risk_signals(response, source))

        # Extract guidance signals
        signals.extend(self._extract_guidance_signals(response, source))

        # Collect all topics
        for signal in signals:
            topics.add(signal.topic)

        # Calculate overall sentiment
        overall_sentiment, sentiment_score = self._calculate_overall_sentiment(signals)

        # Try to extract ticker
        ticker = self._extract_ticker(response)

        return SignalExtractionResult(
            source=source,
            signals=signals,
            overall_sentiment=overall_sentiment,
            sentiment_score=sentiment_score,
            topics=sorted(topics),
            ticker=ticker,
            extraction_successful=True,
        )

    def _extract_sentiment_signals(
        self,
        response: str,
        source: Literal["news_agent", "sec_agent"],
    ) -> list[ExtractedSignal]:
        """Extract sentiment-related signals."""
        signals: list[ExtractedSignal] = []

        # Check positive patterns
        for pattern, topic in self._positive_compiled:
            matches = pattern.findall(response)
            for match in matches:
                raw_text = match if isinstance(match, str) else " ".join(match)
                signals.append(
                    ExtractedSignal(
                        signal_type=SignalType.SENTIMENT,
                        direction=SignalDirection.POSITIVE,
                        topic=topic,
                        description=f"Positive {topic} indicator",
                        confidence=0.85,
                        source=source,
                        raw_text=raw_text,
                    )
                )

        # Check negative patterns
        for pattern, topic in self._negative_compiled:
            matches = pattern.findall(response)
            for match in matches:
                raw_text = match if isinstance(match, str) else " ".join(match)
                signals.append(
                    ExtractedSignal(
                        signal_type=SignalType.SENTIMENT,
                        direction=SignalDirection.NEGATIVE,
                        topic=topic,
                        description=f"Negative {topic} indicator",
                        confidence=0.85,
                        source=source,
                        raw_text=raw_text,
                    )
                )

        # Check neutral patterns
        for pattern, topic in self._neutral_compiled:
            matches = pattern.findall(response)
            for match in matches:
                raw_text = match if isinstance(match, str) else " ".join(match)
                signals.append(
                    ExtractedSignal(
                        signal_type=SignalType.SENTIMENT,
                        direction=SignalDirection.NEUTRAL,
                        topic=topic,
                        description=f"Neutral {topic} indicator",
                        confidence=0.75,
                        source=source,
                        raw_text=raw_text,
                    )
                )

        return signals

    def _extract_financial_signals(
        self,
        response: str,
        source: Literal["news_agent", "sec_agent"],
    ) -> list[ExtractedSignal]:
        """Extract financial metric signals."""
        signals: list[ExtractedSignal] = []

        for pattern, topic, direction_str in self._financial_compiled:
            matches = pattern.findall(response)
            for match in matches:
                raw_text = match if isinstance(match, str) else " ".join(match)
                direction = (
                    SignalDirection.POSITIVE
                    if direction_str == "positive"
                    else SignalDirection.NEGATIVE
                )
                signals.append(
                    ExtractedSignal(
                        signal_type=SignalType.FINANCIAL_METRIC,
                        direction=direction,
                        topic=topic,
                        description=f"{topic.replace('_', ' ').title()} {direction_str}",
                        confidence=0.9,
                        source=source,
                        raw_text=raw_text,
                    )
                )

        return signals

    def _extract_risk_signals(
        self,
        response: str,
        source: Literal["news_agent", "sec_agent"],
    ) -> list[ExtractedSignal]:
        """Extract risk factor signals."""
        signals: list[ExtractedSignal] = []

        for pattern, topic in self._risk_compiled:
            matches = pattern.findall(response)
            if matches:
                # Risk factors are inherently negative signals
                signals.append(
                    ExtractedSignal(
                        signal_type=SignalType.RISK_FACTOR,
                        direction=SignalDirection.NEGATIVE,
                        topic=topic,
                        description=f"{topic.replace('_', ' ').title()} identified",
                        confidence=0.8,
                        source=source,
                        raw_text=matches[0] if isinstance(matches[0], str) else " ".join(matches[0]),
                    )
                )

        return signals

    def _extract_guidance_signals(
        self,
        response: str,
        source: Literal["news_agent", "sec_agent"],
    ) -> list[ExtractedSignal]:
        """Extract forward guidance signals."""
        signals: list[ExtractedSignal] = []

        for pattern, topic, direction_str in self._guidance_compiled:
            matches = pattern.findall(response)
            for match in matches:
                raw_text = match if isinstance(match, str) else " ".join(match)
                if direction_str == "positive":
                    direction = SignalDirection.POSITIVE
                elif direction_str == "negative":
                    direction = SignalDirection.NEGATIVE
                else:
                    direction = SignalDirection.NEUTRAL

                signals.append(
                    ExtractedSignal(
                        signal_type=SignalType.GUIDANCE,
                        direction=direction,
                        topic=topic,
                        description=f"{direction_str.title()} {topic.replace('_', ' ')}",
                        confidence=0.85,
                        source=source,
                        raw_text=raw_text,
                    )
                )

        return signals

    def _calculate_overall_sentiment(
        self,
        signals: list[ExtractedSignal],
    ) -> tuple[SignalDirection, float]:
        """Calculate overall sentiment from extracted signals.

        Returns:
            Tuple of (overall direction, sentiment score from -1.0 to 1.0).
        """
        if not signals:
            return SignalDirection.NEUTRAL, 0.0

        positive_count = sum(1 for s in signals if s.direction == SignalDirection.POSITIVE)
        negative_count = sum(1 for s in signals if s.direction == SignalDirection.NEGATIVE)
        total = len(signals)

        # Weight by confidence
        positive_weight = sum(
            s.confidence for s in signals if s.direction == SignalDirection.POSITIVE
        )
        negative_weight = sum(
            s.confidence for s in signals if s.direction == SignalDirection.NEGATIVE
        )

        total_weight = positive_weight + negative_weight
        if total_weight == 0:
            return SignalDirection.NEUTRAL, 0.0

        # Calculate score from -1.0 to 1.0
        score = (positive_weight - negative_weight) / total_weight

        # Determine overall direction
        if positive_count > negative_count * 1.5:
            direction = SignalDirection.POSITIVE
        elif negative_count > positive_count * 1.5:
            direction = SignalDirection.NEGATIVE
        elif positive_count > 0 and negative_count > 0:
            direction = SignalDirection.MIXED
        else:
            direction = SignalDirection.NEUTRAL

        return direction, round(score, 2)

    def _extract_ticker(self, response: str) -> str | None:
        """Try to extract a ticker symbol from the response."""
        match = self.TICKER_PATTERN.search(response)
        return match.group(1) if match else None
