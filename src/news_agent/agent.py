"""News Sentiment Agent using Bedrock Converse API with tool use."""

from __future__ import annotations

import json
import os
from typing import Any

import boto3

from news_agent.memory.agent_memory import AgentMemory
from news_agent.tools.analyze import analyze_sentiment
from news_agent.tools.fetch_news import fetch_news
from news_agent.tools.trends import get_trends

# ---------------------------------------------------------------------------
# Tool schemas for Bedrock Converse API
# ---------------------------------------------------------------------------

TOOL_CONFIG: dict[str, Any] = {
    "tools": [
        {
            "toolSpec": {
                "name": "fetch_news",
                "description": (
                    "Fetch news articles from NewsAPI for a given company. "
                    "This must be called before analyze_sentiment. "
                    "Articles are cached for 24 hours."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "Stock ticker symbol (e.g., AAPL, TSLA)",
                            },
                            "days_back": {
                                "type": "integer",
                                "description": "Number of days to look back (default: 30, max: 30)",
                            },
                            "force_refresh": {
                                "type": "boolean",
                                "description": "Force refresh from API even if cached",
                            },
                        },
                        "required": ["ticker"],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "analyze_sentiment",
                "description": (
                    "Analyze sentiment and extract key claims from cached news articles. "
                    "The articles must be fetched first using fetch_news. "
                    "Returns overall sentiment, key claims, material events, and narrative summary."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "Stock ticker symbol",
                            },
                            "question": {
                                "type": "string",
                                "description": (
                                    "Optional question to focus the analysis "
                                    "(e.g., 'What is the narrative around iPhone sales?')"
                                ),
                            },
                            "filter_material_only": {
                                "type": "boolean",
                                "description": "Only analyze material/significant news (default: true)",
                            },
                            "max_articles": {
                                "type": "integer",
                                "description": "Maximum articles to analyze (default: 20)",
                            },
                        },
                        "required": ["ticker"],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "get_trends",
                "description": (
                    "Get sentiment trend data over time for a company. "
                    "Shows how sentiment has changed day-over-day and identifies inflection points."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "Stock ticker symbol",
                            },
                            "days_back": {
                                "type": "integer",
                                "description": "Number of days to analyze (default: 30)",
                            },
                        },
                        "required": ["ticker"],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "get_memory",
                "description": (
                    "Retrieve previous analysis results from session memory. "
                    "Use this to check if a company was already analyzed."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "Optional: filter by ticker symbol",
                            },
                        },
                        "required": [],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "save_memory",
                "description": (
                    "Save analysis results to session memory for future reference."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "Stock ticker symbol",
                            },
                            "company_name": {
                                "type": "string",
                                "description": "Company name",
                            },
                            "summary": {
                                "type": "string",
                                "description": "Key findings summary",
                            },
                            "sentiment": {
                                "type": "string",
                                "enum": ["positive", "negative", "neutral", "mixed"],
                                "description": "Overall sentiment",
                            },
                            "sentiment_score": {
                                "type": "number",
                                "description": "Sentiment score (-1.0 to 1.0)",
                            },
                            "material_events": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of material events identified",
                            },
                            "articles_analyzed": {
                                "type": "integer",
                                "description": "Number of articles analyzed",
                            },
                        },
                        "required": [
                            "ticker",
                            "company_name",
                            "summary",
                            "sentiment",
                            "sentiment_score",
                        ],
                    }
                },
            }
        },
    ]
}

SYSTEM_PROMPT = """You are a news sentiment analyst. You help users understand the recent news narrative, sentiment, and material events for publicly traded companies.

When a user asks about a company:
1. Check memory for recent analysis of the same company
2. Fetch news articles if not already cached (use fetch_news)
3. Analyze sentiment and extract key claims (use analyze_sentiment)
4. Optionally check sentiment trends over time (use get_trends)
5. Synthesize a clear answer with specific details and source citations
6. Save key findings to memory for future reference

Always provide:
- Overall sentiment assessment with confidence level
- Key material events or claims with sources
- Trend direction if available (improving, worsening, stable)
- Specific citations from news sources

If the user asks a vague question like "what's happening with X?", interpret it as asking for recent news narrative and sentiment.

When reporting findings:
- Be specific about dates and sources
- Distinguish between facts and opinions
- Note when there's insufficient data for confident analysis
- Highlight any conflicting narratives in the coverage"""

# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

_TOOL_HANDLERS: dict[str, Any] = {
    "fetch_news": fetch_news,
    "analyze_sentiment": analyze_sentiment,
    "get_trends": get_trends,
}

_MEMORY_TOOLS = {"get_memory", "save_memory"}


def _execute_tool(
    name: str, input_data: dict, memory: AgentMemory | None = None
) -> str:
    """Execute a tool by name and return JSON result string."""
    if name in _MEMORY_TOOLS:
        if memory is None:
            return json.dumps({"status": "error", "message": "Memory not available"})
        return _execute_memory_tool(name, input_data, memory)

    handler = _TOOL_HANDLERS.get(name)
    if handler is None:
        return json.dumps({"status": "error", "message": f"Unknown tool: {name}"})

    # Filter out None values from input
    filtered_input = {k: v for k, v in input_data.items() if v is not None}

    result = handler(**filtered_input)
    return json.dumps(result, default=str)


def _execute_memory_tool(name: str, input_data: dict, memory: AgentMemory) -> str:
    """Execute a memory-related tool."""
    if name == "get_memory":
        ticker = input_data.get("ticker")
        if ticker:
            entries = memory.get_recent_analyses(ticker=ticker)
            return json.dumps({"status": "ok", "analyses": entries}, default=str)
        else:
            context = memory.get_session_context()
            return json.dumps({"status": "ok", **context}, default=str)

    if name == "save_memory":
        memory.store_analysis(
            ticker=input_data["ticker"],
            company_name=input_data["company_name"],
            summary=input_data["summary"],
            sentiment=input_data["sentiment"],
            sentiment_score=input_data.get("sentiment_score", 0.0),
            material_events=input_data.get("material_events", []),
            articles_analyzed=input_data.get("articles_analyzed", 0),
        )
        return json.dumps({"status": "ok", "message": "Analysis saved to memory"})

    return json.dumps({"status": "error", "message": f"Unknown memory tool: {name}"})


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class NewsSentimentAgent:
    """Bedrock Converse-based agent for news sentiment analysis."""

    def __init__(
        self,
        model_id: str | None = None,
        bedrock_client: Any | None = None,
        max_turns: int = 10,
        memory: AgentMemory | None = None,
    ) -> None:
        self._model_id = model_id or os.environ.get(
            "BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0"
        )
        self._client = bedrock_client or boto3.client("bedrock-runtime")
        self._max_turns = max_turns
        self._messages: list[dict] = []
        self._memory = memory or AgentMemory()

    def _converse(self) -> dict:
        """Call Bedrock Converse API with current messages."""
        return self._client.converse(
            modelId=self._model_id,
            system=[{"text": SYSTEM_PROMPT}],
            messages=self._messages,
            toolConfig=TOOL_CONFIG,
        )

    def _process_tool_uses(self, content: list[dict]) -> list[dict]:
        """Execute tool calls from assistant response and return tool result blocks."""
        tool_results = []
        for block in content:
            if "toolUse" not in block:
                continue
            tool_use = block["toolUse"]
            result_str = _execute_tool(
                tool_use["name"], tool_use["input"], memory=self._memory
            )
            tool_results.append(
                {
                    "toolResult": {
                        "toolUseId": tool_use["toolUseId"],
                        "content": [{"json": json.loads(result_str)}],
                    }
                }
            )
        return tool_results

    def query(self, user_message: str) -> str:
        """Send a user message and run the agent loop until a final text response.

        Returns the assistant's final text response.
        """
        self._messages.append({"role": "user", "content": [{"text": user_message}]})

        for _ in range(self._max_turns):
            response = self._converse()
            output = response["output"]["message"]
            self._messages.append(output)

            stop_reason = response.get("stopReason", "end_turn")

            if stop_reason == "tool_use":
                tool_results = self._process_tool_uses(output["content"])
                self._messages.append({"role": "user", "content": tool_results})
            else:
                # Extract final text
                text_parts = [
                    block["text"]
                    for block in output["content"]
                    if "text" in block
                ]
                return "\n".join(text_parts)

        return "I was unable to complete the analysis within the allowed number of steps."

    @property
    def memory(self) -> AgentMemory:
        """Return the agent's memory instance."""
        return self._memory

    def reset(self) -> None:
        """Clear conversation history and memory."""
        self._messages = []
        self._memory.clear()
