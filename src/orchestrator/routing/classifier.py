"""Query classification for routing to appropriate agents."""

from __future__ import annotations

import re
from typing import Pattern

from orchestrator.execution.result import QueryClassification, RouteType


class QueryClassifier:
    """Classifies user queries to determine which agent(s) to route to.

    Uses rule-based pattern matching to classify queries as:
    - NEWS_ONLY: Query is specifically about news/sentiment
    - SEC_ONLY: Query is specifically about SEC filings
    - BOTH: Query requires comparison or is ambiguous

    Attributes:
        news_patterns: Compiled regex patterns for news-related queries.
        sec_patterns: Compiled regex patterns for SEC-related queries.
        comparison_patterns: Compiled regex patterns for comparison queries.
    """

    # Pattern strings for news-related queries
    NEWS_PATTERN_STRINGS: list[str] = [
        r"\bnews\b",
        r"\bsentiment\b",
        r"\bheadlines?\b",
        r"\brecent\b",
        r"\blatest\b",
        r"\bbreaking\b",
        r"\bpress\s+release\b",
        r"\bmedia\b",
        r"\breport(?:s|ed|ing)?\b",
        r"\banalyst(?:s)?\b",
        r"\bmarket\s+reaction\b",
        r"\bstock\s+price\b",
        r"\btrading\b",
    ]

    # Pattern strings for SEC-related queries
    SEC_PATTERN_STRINGS: list[str] = [
        r"\b10-?[KQ]\b",
        r"\b8-?K\b",
        r"\bfiling(?:s)?\b",
        r"\brisk\s+factors?\b",
        r"\bsec\b",
        r"\bannual\s+report\b",
        r"\bquarterly\s+report\b",
        r"\bfinancial\s+statements?\b",
        r"\bearnings\b",
        r"\brevenue\b",
        r"\bbalance\s+sheet\b",
        r"\bcash\s+flow\b",
        r"\bmd&a\b",
        r"\bmanagement\s+discussion\b",
        r"\bforward[- ]looking\b",
        r"\bregulatory\b",
    ]

    # Pattern strings for comparison queries
    COMPARISON_PATTERN_STRINGS: list[str] = [
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bvs\.?\b",
        r"\bversus\b",
        r"\bdiscrepanc(?:y|ies)\b",
        r"\bdifference(?:s)?\b",
        r"\bcontrast\b",
        r"\bmatch(?:es)?\b",
        r"\balign(?:s|ed|ment)?\b",
        r"\bconsistent\b",
        r"\binconsistent\b",
        r"\bboth\s+(?:sources?|agents?)\b",
        r"\bnews\s+(?:and|&)\s+(?:sec|filings?)\b",
        r"\bfilings?\s+(?:and|&)\s+news\b",
    ]

    def __init__(self) -> None:
        """Initialize the classifier with compiled regex patterns."""
        self.news_patterns: list[Pattern[str]] = [
            re.compile(p, re.IGNORECASE) for p in self.NEWS_PATTERN_STRINGS
        ]
        self.sec_patterns: list[Pattern[str]] = [
            re.compile(p, re.IGNORECASE) for p in self.SEC_PATTERN_STRINGS
        ]
        self.comparison_patterns: list[Pattern[str]] = [
            re.compile(p, re.IGNORECASE) for p in self.COMPARISON_PATTERN_STRINGS
        ]

    def classify(self, query: str) -> QueryClassification:
        """Classify a user query to determine routing.

        Classification logic:
        1. Check for explicit comparison keywords -> BOTH agents
        2. Check for SEC-specific terms -> SEC_ONLY
        3. Check for news-specific terms -> NEWS_ONLY
        4. Default/ambiguous -> BOTH agents

        Args:
            query: The user's query string.

        Returns:
            QueryClassification with route_type, confidence, and matched patterns.
        """
        matched_patterns: list[str] = []

        # Check for comparison patterns first (highest priority)
        comparison_matches = self._find_matches(query, self.comparison_patterns)
        if comparison_matches:
            matched_patterns.extend(comparison_matches)
            return QueryClassification(
                route_type=RouteType.BOTH,
                confidence=1.0,
                matched_patterns=matched_patterns,
                reasoning="Query contains comparison keywords",
            )

        # Check for SEC and news patterns
        sec_matches = self._find_matches(query, self.sec_patterns)
        news_matches = self._find_matches(query, self.news_patterns)

        # If both have matches, route to both
        if sec_matches and news_matches:
            matched_patterns.extend(sec_matches)
            matched_patterns.extend(news_matches)
            return QueryClassification(
                route_type=RouteType.BOTH,
                confidence=0.9,
                matched_patterns=matched_patterns,
                reasoning="Query contains both SEC and news keywords",
            )

        # SEC-only patterns
        if sec_matches:
            matched_patterns.extend(sec_matches)
            return QueryClassification(
                route_type=RouteType.SEC_ONLY,
                confidence=self._calculate_confidence(len(sec_matches)),
                matched_patterns=matched_patterns,
                reasoning="Query contains SEC-specific keywords",
            )

        # News-only patterns
        if news_matches:
            matched_patterns.extend(news_matches)
            return QueryClassification(
                route_type=RouteType.NEWS_ONLY,
                confidence=self._calculate_confidence(len(news_matches)),
                matched_patterns=matched_patterns,
                reasoning="Query contains news-specific keywords",
            )

        # Default: ambiguous query, route to both agents
        return QueryClassification(
            route_type=RouteType.BOTH,
            confidence=0.5,
            matched_patterns=[],
            reasoning="No specific patterns matched; defaulting to both agents",
        )

    def _find_matches(
        self, query: str, patterns: list[Pattern[str]]
    ) -> list[str]:
        """Find all matching patterns in the query.

        Args:
            query: The query string to search.
            patterns: List of compiled regex patterns.

        Returns:
            List of matched pattern strings.
        """
        matches: list[str] = []
        for pattern in patterns:
            if pattern.search(query):
                matches.append(pattern.pattern)
        return matches

    def _calculate_confidence(self, match_count: int) -> float:
        """Calculate confidence score based on number of matches.

        Args:
            match_count: Number of pattern matches found.

        Returns:
            Confidence score between 0.6 and 1.0.
        """
        # More matches = higher confidence, capped at 1.0
        base_confidence = 0.6
        confidence_per_match = 0.1
        return min(1.0, base_confidence + (match_count * confidence_per_match))
