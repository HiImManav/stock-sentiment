"""AgentCore Bedrock agent for SEC filings analysis.

Uses the Bedrock Converse API with tool use to orchestrate filing
fetching, parsing, and question answering.
"""

from __future__ import annotations

import json
import os
from typing import Any

import boto3

from sec_agent.memory.agent_memory import AgentMemory
from sec_agent.tools.fetch_filing import fetch_and_parse_filing
from sec_agent.tools.query_section import list_available_filings, query_filing

# ---------------------------------------------------------------------------
# Tool schemas for Bedrock Converse API
# ---------------------------------------------------------------------------

TOOL_CONFIG: dict[str, Any] = {
    "tools": [
        {
            "toolSpec": {
                "name": "fetch_and_parse_filing",
                "description": (
                    "Fetch a SEC filing from EDGAR, parse it into sections, and cache it. "
                    "Call this before querying a filing."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "Stock ticker symbol (e.g., AAPL)",
                            },
                            "filing_type": {
                                "type": "string",
                                "enum": ["10-K", "10-Q", "8-K"],
                                "description": "Type of SEC filing",
                            },
                            "filing_index": {
                                "type": "integer",
                                "description": (
                                    "0 = most recent, 1 = second most recent, etc."
                                ),
                            },
                        },
                        "required": ["ticker", "filing_type"],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "query_filing",
                "description": (
                    "Search a cached SEC filing and return relevant chunks to answer "
                    "a question. The filing must be fetched first."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "Stock ticker symbol",
                            },
                            "filing_type": {
                                "type": "string",
                                "enum": ["10-K", "10-Q", "8-K"],
                                "description": "Type of SEC filing",
                            },
                            "question": {
                                "type": "string",
                                "description": "Natural language question about the filing",
                            },
                            "section_filter": {
                                "type": "string",
                                "description": (
                                    "Optional: limit search to a specific section "
                                    "(e.g., '1A' for Risk Factors)"
                                ),
                            },
                        },
                        "required": ["ticker", "filing_type", "question"],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "list_available_filings",
                "description": "List cached filings available for a given ticker.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "Stock ticker symbol",
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
                    "Retrieve previous analysis results from memory. "
                    "Use this to check if a company was already analyzed in this session."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": (
                                    "Optional: filter by ticker symbol. "
                                    "Omit to get all session memory."
                                ),
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
                    "Save analysis results to memory for future reference within this session."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "Stock ticker symbol",
                            },
                            "filing_type": {
                                "type": "string",
                                "enum": ["10-K", "10-Q", "8-K"],
                                "description": "Type of SEC filing analyzed",
                            },
                            "summary": {
                                "type": "string",
                                "description": "Key findings summary to remember",
                            },
                            "sections_analyzed": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of section item numbers analyzed (e.g., ['1A', '7'])",
                            },
                        },
                        "required": ["ticker", "filing_type", "summary", "sections_analyzed"],
                    }
                },
            }
        },
    ]
}

SYSTEM_PROMPT = (
    "You are an SEC filings analyst. You help users understand company risks, "
    "management discussion, material events, and financial statements from "
    "SEC filings (10-K, 10-Q, 8-K).\n\n"
    "When a user asks about a company:\n"
    "1. Check memory for recent analysis of the same company\n"
    "2. Fetch the relevant filing if not already cached\n"
    "3. Query the appropriate sections to find relevant information\n"
    "4. Synthesize a clear answer with specific citations (section, item number)\n"
    "5. Save key findings to memory for future reference\n\n"
    "Always cite the specific filing type, date, and section you are referencing.\n"
    "If the user asks a vague question, infer the most relevant filing type and section.\n"
    "If the user asks what companies have been analyzed, use get_memory to check."
)

# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

_TOOL_HANDLERS: dict[str, Any] = {
    "fetch_and_parse_filing": fetch_and_parse_filing,
    "query_filing": query_filing,
    "list_available_filings": list_available_filings,
}

# Memory tools are handled separately because they need the agent's memory instance.
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
    result = handler(**input_data)
    return json.dumps(result, default=str)


def _execute_memory_tool(
    name: str, input_data: dict, memory: AgentMemory
) -> str:
    """Execute a memory-related tool."""
    if name == "get_memory":
        ticker = input_data.get("ticker")
        if ticker:
            entries = memory.get_recent_analyses(ticker=ticker)
        else:
            context = memory.get_session_context()
            return json.dumps({"status": "ok", **context}, default=str)
        return json.dumps({"status": "ok", "analyses": entries}, default=str)

    if name == "save_memory":
        memory.store_analysis(
            ticker=input_data["ticker"],
            filing_type=input_data["filing_type"],
            summary=input_data["summary"],
            sections_analyzed=input_data["sections_analyzed"],
        )
        return json.dumps({"status": "ok", "message": "Analysis saved to memory"})

    return json.dumps({"status": "error", "message": f"Unknown memory tool: {name}"})


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class SECFilingsAgent:
    """Bedrock Converse-based agent for SEC filings analysis."""

    def __init__(
        self,
        model_id: str | None = None,
        bedrock_client: Any | None = None,
        max_turns: int = 10,
        memory: AgentMemory | None = None,
    ) -> None:
        self._model_id = model_id or os.environ.get(
            "BEDROCK_MODEL_ID", "anthropic.claude-opus-4-5-20251101-v1:0"
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
