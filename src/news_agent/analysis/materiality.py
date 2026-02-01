"""Three-layer materiality filter for news articles."""

from __future__ import annotations

import re
from typing import Literal

from news_agent.models import NewsArticle

# Source tiers for reputation filtering
TIER_1_SOURCES: set[str] = {
    # Major wire services
    "reuters.com",
    "apnews.com",
    "afp.com",
    # Financial news
    "bloomberg.com",
    "wsj.com",
    "ft.com",
    "barrons.com",
    "cnbc.com",
    "finance.yahoo.com",
    # Business publications
    "forbes.com",
    "businessinsider.com",
    "fortune.com",
    # Tech/Industry
    "techcrunch.com",
    "wired.com",
    "theverge.com",
    # Official sources
    "sec.gov",
    "prnewswire.com",
    "businesswire.com",
    "globenewswire.com",
}

TIER_2_SOURCES: set[str] = {
    "marketwatch.com",
    "seekingalpha.com",
    "investopedia.com",
    "thestreet.com",
    "fool.com",
    "benzinga.com",
    "zacks.com",
    "tipranks.com",
    "investors.com",
    "morningstar.com",
}

# Keywords that indicate potentially material news
MATERIAL_KEYWORDS: dict[str, list[str]] = {
    "earnings": [
        "earnings",
        "revenue",
        "profit",
        "loss",
        "guidance",
        "quarterly results",
        "eps",
        "beat expectations",
        "miss expectations",
        "fiscal",
    ],
    "legal": [
        "lawsuit",
        "settlement",
        "sec investigation",
        "fraud",
        "litigation",
        "indictment",
        "subpoena",
        "class action",
        "regulatory fine",
    ],
    "regulatory": [
        "fda approval",
        "fda rejection",
        "antitrust",
        "regulation",
        "compliance",
        "ftc",
        "doj investigation",
        "eu commission",
    ],
    "leadership": [
        "ceo",
        "cfo",
        "cto",
        "coo",
        "resignation",
        "appointed",
        "board of directors",
        "executive departure",
        "succession",
    ],
    "m_and_a": [
        "acquisition",
        "merger",
        "buyout",
        "takeover",
        "spin-off",
        "divestiture",
        "ipo",
        "spac",
        "deal",
    ],
    "operations": [
        "layoffs",
        "restructuring",
        "plant closure",
        "expansion",
        "hiring freeze",
        "workforce reduction",
        "cost cutting",
    ],
    "product": [
        "product launch",
        "recall",
        "patent",
        "product failure",
        "new product",
        "innovation",
        "breakthrough",
    ],
    "financial": [
        "debt",
        "credit rating",
        "bankruptcy",
        "default",
        "refinancing",
        "dividend",
        "buyback",
        "share repurchase",
    ],
}


class MaterialityFilter:
    """Three-layer filter to identify material news articles."""

    def __init__(self) -> None:
        # Compile keyword patterns for efficient matching
        self._keyword_patterns: dict[str, re.Pattern] = {}
        for category, keywords in MATERIAL_KEYWORDS.items():
            pattern = "|".join(re.escape(kw) for kw in keywords)
            self._keyword_patterns[category] = re.compile(pattern, re.IGNORECASE)

    def get_source_tier(self, domain: str) -> Literal[1, 2, 3]:
        """Determine the reputation tier of a news source.

        Args:
            domain: Source domain (e.g., 'reuters.com')

        Returns:
            Tier 1 (high credibility), 2 (medium), or 3 (low)
        """
        domain_lower = domain.lower().replace("www.", "")

        if domain_lower in TIER_1_SOURCES:
            return 1
        elif domain_lower in TIER_2_SOURCES:
            return 2
        else:
            return 3

    def get_source_weight(self, domain: str) -> float:
        """Get the weight multiplier for a source based on tier.

        Args:
            domain: Source domain

        Returns:
            Weight multiplier (1.0, 0.7, or 0.3)
        """
        tier = self.get_source_tier(domain)
        weights = {1: 1.0, 2: 0.7, 3: 0.3}
        return weights[tier]

    def detect_keywords(self, text: str) -> dict[str, bool]:
        """Detect material keywords in text.

        Args:
            text: Text to analyze

        Returns:
            Dict mapping category names to whether keywords were found
        """
        results: dict[str, bool] = {}
        for category, pattern in self._keyword_patterns.items():
            results[category] = bool(pattern.search(text))
        return results

    def get_detected_topics(self, text: str) -> list[str]:
        """Get list of material topics detected in text.

        Args:
            text: Text to analyze

        Returns:
            List of detected topic categories
        """
        keywords = self.detect_keywords(text)
        return [cat for cat, found in keywords.items() if found]

    def passes_layer_1(self, article: NewsArticle) -> tuple[bool, str]:
        """Layer 1: Source reputation check.

        Args:
            article: News article to check

        Returns:
            Tuple of (passes, reason)
        """
        tier = self.get_source_tier(article.source_domain)
        if tier <= 2:
            return True, f"Source tier {tier} ({article.source_name})"
        else:
            return False, f"Low-tier source ({article.source_name})"

    def passes_layer_2(self, article: NewsArticle) -> tuple[bool, str]:
        """Layer 2: Keyword detection.

        Args:
            article: News article to check

        Returns:
            Tuple of (passes, reason)
        """
        # Combine title, description, and content for analysis
        text = f"{article.title} {article.description} {article.content}"
        topics = self.get_detected_topics(text)

        if topics:
            return True, f"Material keywords detected: {', '.join(topics)}"
        else:
            return False, "No material keywords detected"

    def filter_articles(
        self, articles: list[NewsArticle], min_tier: int = 3
    ) -> list[tuple[NewsArticle, float, list[str]]]:
        """Filter articles by materiality (Layers 1 & 2 only).

        Layer 3 (LLM classification) is handled separately in sentiment analysis.

        Args:
            articles: List of articles to filter
            min_tier: Minimum source tier to include (1-3)

        Returns:
            List of tuples: (article, weight, detected_topics)
        """
        results: list[tuple[NewsArticle, float, list[str]]] = []

        for article in articles:
            tier = self.get_source_tier(article.source_domain)

            # Skip if below minimum tier
            if tier > min_tier:
                continue

            weight = self.get_source_weight(article.source_domain)
            text = f"{article.title} {article.description} {article.content}"
            topics = self.get_detected_topics(text)

            # Boost weight if material keywords found
            if topics:
                weight = min(weight * 1.2, 1.0)

            results.append((article, weight, topics))

        return results

    def is_clickbait(self, title: str) -> bool:
        """Check if a headline appears to be clickbait.

        Args:
            title: Article headline

        Returns:
            True if headline shows clickbait patterns
        """
        clickbait_patterns = [
            r"you won't believe",
            r"this one trick",
            r"shocking",
            r"breaking:",
            r"\d+ reasons",
            r"what happens next",
            r"find out",
            r"jaw-dropping",
        ]

        title_lower = title.lower()
        for pattern in clickbait_patterns:
            if re.search(pattern, title_lower):
                return True
        return False
