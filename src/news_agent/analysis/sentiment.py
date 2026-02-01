"""LLM-based sentiment classification for news articles."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Literal

import boto3

from news_agent.models import NewsArticle, SentimentAnalysis

_DEFAULT_MODEL = "us.anthropic.claude-sonnet-4-20250514-v1:0"

SENTIMENT_ANALYSIS_PROMPT = """Analyze this news article about {company_name} ({ticker}) for sentiment and financial materiality.

Article Title: {title}
Source: {source}
Date: {date}
Content: {content}

Analyze the article and respond with a JSON object containing:

{{
    "sentiment": "positive" | "negative" | "neutral",
    "confidence": 0.0-1.0,
    "magnitude": 0.0-1.0,
    "is_material": true | false,
    "materiality_reason": "Brief explanation of why this is/isn't material",
    "key_claims": ["List of specific factual claims from the article"],
    "topics": ["List from: earnings, legal, regulatory, leadership, m_and_a, operations, product, financial, other"]
}}

Guidelines for materiality classification - mark as MATERIAL if the news could reasonably:
1. Impact the company's stock price by >1%
2. Affect quarterly revenue/earnings estimates
3. Change the company's risk profile
4. Influence investor perception significantly

Be specific in key_claims - extract verifiable facts, not opinions.
Return only the JSON object, no other text."""


BATCH_SENTIMENT_PROMPT = """Analyze these news articles about {company_name} ({ticker}) for sentiment and materiality.

{articles_text}

For each article, provide analysis in this JSON format:

{{
    "analyses": [
        {{
            "article_index": 0,
            "sentiment": "positive" | "negative" | "neutral",
            "confidence": 0.0-1.0,
            "magnitude": 0.0-1.0,
            "is_material": true | false,
            "materiality_reason": "Brief explanation",
            "key_claims": ["Specific factual claims"],
            "topics": ["Categories from: earnings, legal, regulatory, leadership, m_and_a, operations, product, financial, other"]
        }}
    ]
}}

Return only the JSON object."""


class SentimentAnalyzer:
    """Analyzes news article sentiment using Claude via Bedrock."""

    def __init__(
        self,
        model_id: str | None = None,
        bedrock_client: Any | None = None,
    ) -> None:
        self._model_id = model_id or os.environ.get("BEDROCK_MODEL_ID", _DEFAULT_MODEL)
        self._bedrock = bedrock_client or boto3.client("bedrock-runtime")

    def _call_bedrock(self, prompt: str, max_tokens: int = 4000) -> str:
        """Make a Bedrock Converse API call."""
        response = self._bedrock.converse(
            modelId=self._model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0},
        )

        output = response.get("output", {})
        message = output.get("message", {})
        content = message.get("content", [])

        if content and content[0].get("text"):
            return content[0]["text"]
        return "{}"

    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON from LLM response, handling markdown code blocks."""
        response = response.strip()

        # Remove markdown code blocks if present
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(l for l in lines if not l.strip().startswith("```"))

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {}

    def analyze_article(self, article: NewsArticle) -> SentimentAnalysis:
        """Analyze sentiment of a single article.

        Args:
            article: News article to analyze

        Returns:
            SentimentAnalysis result
        """
        prompt = SENTIMENT_ANALYSIS_PROMPT.format(
            company_name=article.company_name,
            ticker=article.ticker,
            title=article.title,
            source=article.source_name,
            date=article.published_at.strftime("%Y-%m-%d"),
            content=article.content or article.description,
        )

        response_text = self._call_bedrock(prompt)
        data = self._parse_json_response(response_text)

        # Extract and validate fields with defaults
        sentiment: Literal["positive", "negative", "neutral"] = data.get(
            "sentiment", "neutral"
        )
        if sentiment not in ("positive", "negative", "neutral"):
            sentiment = "neutral"

        return SentimentAnalysis(
            article_id=article.article_id,
            sentiment=sentiment,
            confidence=float(data.get("confidence", 0.5)),
            magnitude=float(data.get("magnitude", 0.5)),
            key_claims=data.get("key_claims", []),
            topics=data.get("topics", []),
            is_material=bool(data.get("is_material", False)),
            materiality_reason=data.get("materiality_reason", ""),
            analysis_timestamp=datetime.now(timezone.utc),
        )

    def analyze_articles_batch(
        self, articles: list[NewsArticle], batch_size: int = 5
    ) -> list[SentimentAnalysis]:
        """Analyze sentiment of multiple articles efficiently.

        Args:
            articles: List of articles to analyze
            batch_size: Number of articles per LLM call

        Returns:
            List of SentimentAnalysis results
        """
        if not articles:
            return []

        results: list[SentimentAnalysis] = []

        # Process in batches
        for i in range(0, len(articles), batch_size):
            batch = articles[i : i + batch_size]

            if len(batch) == 1:
                # Single article - use individual analysis
                results.append(self.analyze_article(batch[0]))
                continue

            # Build batch prompt
            articles_text = "\n\n".join(
                f"--- Article {j} ---\n"
                f"Title: {a.title}\n"
                f"Source: {a.source_name}\n"
                f"Date: {a.published_at.strftime('%Y-%m-%d')}\n"
                f"Content: {a.content or a.description}"
                for j, a in enumerate(batch)
            )

            prompt = BATCH_SENTIMENT_PROMPT.format(
                company_name=batch[0].company_name,
                ticker=batch[0].ticker,
                articles_text=articles_text,
            )

            response_text = self._call_bedrock(prompt, max_tokens=8000)
            data = self._parse_json_response(response_text)

            analyses = data.get("analyses", [])

            # Map responses back to articles
            for j, article in enumerate(batch):
                # Find matching analysis
                analysis_data = None
                for a in analyses:
                    if a.get("article_index") == j:
                        analysis_data = a
                        break

                if not analysis_data:
                    # Fallback to individual analysis
                    results.append(self.analyze_article(article))
                    continue

                sentiment: Literal["positive", "negative", "neutral"] = analysis_data.get(
                    "sentiment", "neutral"
                )
                if sentiment not in ("positive", "negative", "neutral"):
                    sentiment = "neutral"

                results.append(
                    SentimentAnalysis(
                        article_id=article.article_id,
                        sentiment=sentiment,
                        confidence=float(analysis_data.get("confidence", 0.5)),
                        magnitude=float(analysis_data.get("magnitude", 0.5)),
                        key_claims=analysis_data.get("key_claims", []),
                        topics=analysis_data.get("topics", []),
                        is_material=bool(analysis_data.get("is_material", False)),
                        materiality_reason=analysis_data.get("materiality_reason", ""),
                        analysis_timestamp=datetime.now(timezone.utc),
                    )
                )

        return results

    def calculate_aggregate_sentiment(
        self, analyses: list[SentimentAnalysis]
    ) -> dict:
        """Calculate aggregate sentiment metrics from multiple analyses.

        Args:
            analyses: List of sentiment analysis results

        Returns:
            Dict with overall_sentiment, sentiment_score, confidence, etc.
        """
        if not analyses:
            return {
                "overall_sentiment": "neutral",
                "sentiment_score": 0.0,
                "confidence": 0.0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
            }

        # Count sentiments
        positive_count = sum(1 for a in analyses if a.sentiment == "positive")
        negative_count = sum(1 for a in analyses if a.sentiment == "negative")
        neutral_count = sum(1 for a in analyses if a.sentiment == "neutral")

        # Calculate weighted sentiment score (-1 to 1)
        # Weight by confidence and magnitude
        total_weight = 0.0
        weighted_score = 0.0

        for analysis in analyses:
            weight = analysis.confidence * analysis.magnitude
            if analysis.sentiment == "positive":
                weighted_score += weight
            elif analysis.sentiment == "negative":
                weighted_score -= weight
            total_weight += weight

        sentiment_score = weighted_score / total_weight if total_weight > 0 else 0.0

        # Determine overall sentiment
        if abs(sentiment_score) < 0.15:
            overall_sentiment = "neutral"
        elif sentiment_score >= 0.15:
            if negative_count > positive_count * 0.5:
                overall_sentiment = "mixed"
            else:
                overall_sentiment = "positive"
        else:
            if positive_count > negative_count * 0.5:
                overall_sentiment = "mixed"
            else:
                overall_sentiment = "negative"

        # Average confidence
        avg_confidence = sum(a.confidence for a in analyses) / len(analyses)

        return {
            "overall_sentiment": overall_sentiment,
            "sentiment_score": round(sentiment_score, 3),
            "confidence": round(avg_confidence, 3),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
        }
