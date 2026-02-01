"""Extract key claims from news articles using LLM."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import boto3

from news_agent.models import NewsArticle

_DEFAULT_MODEL = "us.anthropic.claude-sonnet-4-20250514-v1:0"

CLAIMS_EXTRACTION_PROMPT = """Extract the key factual claims from this news article about {company_name} ({ticker}).

Article Title: {title}
Source: {source}
Date: {date}
Content: {content}

Extract specific, verifiable factual claims. Focus on:
- Financial metrics (revenue, profit, growth percentages)
- Specific events (announcements, lawsuits, regulatory actions)
- Quotes from executives or officials
- Analyst opinions with attribution

Return a JSON object with this structure:
{{
    "claims": [
        {{
            "claim": "The specific factual claim",
            "type": "metric|event|quote|opinion",
            "confidence": 0.0-1.0
        }}
    ]
}}

Only include claims that are explicitly stated in the article. Do not infer or speculate.
Return only the JSON object, no other text."""


class ClaimsExtractor:
    """Extracts key claims from news articles using Claude."""

    def __init__(
        self,
        model_id: str | None = None,
        bedrock_client: Any | None = None,
    ) -> None:
        self._model_id = model_id or os.environ.get("BEDROCK_MODEL_ID", _DEFAULT_MODEL)
        self._bedrock = bedrock_client or boto3.client("bedrock-runtime")

    def _call_bedrock(self, prompt: str) -> str:
        """Make a Bedrock API call."""
        response = self._bedrock.converse(
            modelId=self._model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 2000, "temperature": 0.0},
        )

        output = response.get("output", {})
        message = output.get("message", {})
        content = message.get("content", [])

        if content and content[0].get("text"):
            return content[0]["text"]
        return "{}"

    def extract_claims(self, article: NewsArticle) -> list[dict]:
        """Extract key claims from a single article.

        Args:
            article: News article to analyze

        Returns:
            List of claim dicts with 'claim', 'type', and 'confidence' fields
        """
        prompt = CLAIMS_EXTRACTION_PROMPT.format(
            company_name=article.company_name,
            ticker=article.ticker,
            title=article.title,
            source=article.source_name,
            date=article.published_at.strftime("%Y-%m-%d"),
            content=article.content or article.description,
        )

        response_text = self._call_bedrock(prompt)

        try:
            # Try to extract JSON from response
            response_text = response_text.strip()
            if response_text.startswith("```"):
                # Remove markdown code blocks
                lines = response_text.split("\n")
                response_text = "\n".join(
                    l for l in lines if not l.strip().startswith("```")
                )

            data = json.loads(response_text)
            claims = data.get("claims", [])

            # Validate and clean claims
            valid_claims = []
            for claim in claims:
                if isinstance(claim, dict) and claim.get("claim"):
                    valid_claims.append(
                        {
                            "claim": str(claim["claim"]),
                            "type": claim.get("type", "event"),
                            "confidence": float(claim.get("confidence", 0.8)),
                        }
                    )
            return valid_claims

        except (json.JSONDecodeError, KeyError, TypeError):
            return []

    def extract_claims_batch(
        self, articles: list[NewsArticle]
    ) -> dict[str, list[dict]]:
        """Extract claims from multiple articles.

        Args:
            articles: List of articles to analyze

        Returns:
            Dict mapping article_id to list of claims
        """
        results: dict[str, list[dict]] = {}

        for article in articles:
            claims = self.extract_claims(article)
            results[article.article_id] = claims

        return results

    def format_claims_with_source(
        self, article: NewsArticle, claims: list[dict]
    ) -> list[dict]:
        """Format claims with source attribution.

        Args:
            article: Source article
            claims: List of claims from the article

        Returns:
            List of claims with source, date, and article info
        """
        return [
            {
                "claim": c["claim"],
                "type": c["type"],
                "confidence": c["confidence"],
                "source": article.source_name,
                "date": article.published_at.strftime("%Y-%m-%d"),
                "article_title": article.title,
                "article_url": article.url,
            }
            for c in claims
        ]
