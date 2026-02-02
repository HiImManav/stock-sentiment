# Orchestration Agent PRD

## Overview
The Orchestration Agent coordinates between the `news_agent` (news sentiment analysis) and `sec_agent` (SEC filings analysis) to provide unified company intelligence. It intelligently routes queries, executes agents in parallel, handles failures gracefully, and compares information from both sources.

## Goals
1. **Smart Routing**: Determine which agent(s) to call based on user query
2. **Parallel Execution**: Run multiple agents asynchronously for performance
3. **Graceful Failure Handling**: Return partial results if one agent fails
4. **Data Pooling**: Combine results into a unified response
5. **Discrepancy Detection**: Compare news sentiment vs SEC filing data

## Workflow
```
User Query
    |
    v
[Query Classifier] --> Determine: news_only | sec_only | both
    |
    v
[Async Executor] --> Run agents in parallel with timeout
    |
    +---> NewsSentimentAgent.query()
    +---> SECFilingsAgent.query()
    |
    v
[Result Comparator] --> Identify agreements/discrepancies
    |
    v
[Response Synthesizer] --> Generate unified response (LLM)
    |
    v
Final Response to User
```

## Architecture

### Lightweight Coordinator (Not Another Bedrock Agent)
The orchestrator is NOT a Bedrock tool-use agent. This avoids:
- Nested LLM calls (orchestrator LLM -> sub-agent LLM)
- Excessive latency and cost
- Complex debugging

Instead, it's a Python coordinator that:
1. Uses rule-based query classification (with optional LLM fallback)
2. Executes sub-agents via `asyncio.to_thread()`
3. Makes a single LLM call for final synthesis

### Sub-Agent Integration
Both sub-agents have identical interfaces:
```python
class Agent:
    def __init__(self, model_id=None, bedrock_client=None, max_turns=10, memory=None)
    def query(user_message: str) -> str
    def reset() -> None
    @property memory -> AgentMemory
```

## File Structure
```
src/orchestrator/
    __init__.py                    # Package exports
    agent.py                       # Main OrchestrationAgent class
    routing/
        __init__.py
        classifier.py              # Query classification logic
    execution/
        __init__.py
        async_executor.py          # Parallel agent execution
        result.py                  # AgentResult, QueryClassification dataclasses
    comparison/
        __init__.py
        comparator.py              # News vs SEC result comparison
    synthesis/
        __init__.py
        synthesizer.py             # LLM-based response synthesis
    memory/
        __init__.py
        agent_memory.py            # Unified orchestrator memory
    cli/
        __init__.py
        main.py                    # CLI interface
    api/
        __init__.py
        server.py                  # FastAPI endpoints
    mangum_handler.py              # AWS Lambda handler
```

## Implementation Tasks

### Phase 1: Core Infrastructure
| Task | Description | Status |
|------|-------------|--------|
| 1.1 | Create module structure and `__init__.py` files | Done |
| 1.2 | Implement `AgentResult` and `QueryClassification` dataclasses | Done |
| 1.3 | Implement `QueryClassifier` with rule-based patterns | Done |
| 1.4 | Write unit tests for classifier | Done |

### Phase 2: Async Execution
| Task | Description | Status |
|------|-------------|--------|
| 2.1 | Implement `AsyncAgentExecutor` with timeout handling | Done |
| 2.2 | Add graceful error handling for timeouts and exceptions | Done |
| 2.3 | Write unit tests with mocked agents | Done |

### Phase 3: Comparison Logic
| Task | Description | Status |
|------|-------------|--------|
| 3.1 | Implement signal extraction from agent responses | Done |
| 3.2 | Implement discrepancy detection rules | Done |
| 3.3 | Implement `ComparisonResult` generation | Done |

### Phase 4: Synthesis & Main Agent
| Task | Description | Status |
|------|-------------|--------|
| 4.1 | Implement `ResponseSynthesizer` with Bedrock | Done |
| 4.2 | Implement `OrchestratorMemory` | Done |
| 4.3 | Implement main `OrchestrationAgent` class | Done |
| 4.4 | End-to-end integration tests | Done |

### Phase 5: CLI & API
| Task | Description | Status |
|------|-------------|--------|
| 5.1 | Implement CLI with `chat`, `query`, `compare` commands | Done |
| 5.2 | Implement FastAPI endpoints | Done |
| 5.3 | Add `orchestrator` entry point to `pyproject.toml` | Done |
| 5.4 | Implement Lambda handler | Pending |

### Phase 6: Testing & Polish
| Task | Description | Status |
|------|-------------|--------|
| 6.1 | Write comprehensive unit tests | Pending |
| 6.2 | Write integration tests | Pending |
| 6.3 | Add type hints and run mypy | Pending |

## Key Components

### Query Classifier (`routing/classifier.py`)
```python
NEWS_PATTERNS = [r"\bnews\b", r"\bsentiment\b", r"\bheadlines?\b", r"\brecent\b"]
SEC_PATTERNS = [r"\b10-?[KQ]\b", r"\b8-?K\b", r"\bfiling\b", r"\brisk factors?\b"]
COMPARISON_PATTERNS = [r"\bcompare\b", r"\bvs\b", r"\bdiscrepancy\b"]
```

Classification logic:
1. Check for explicit comparison keywords -> both agents
2. Check for SEC-specific terms -> sec_agent only
3. Check for news-specific terms -> news_agent only
4. Default/ambiguous -> both agents

### Async Executor (`execution/async_executor.py`)
- Uses `asyncio.to_thread()` for parallel execution of synchronous agents
- Configurable timeout (default: 60s)
- Returns `AgentResult` with status: success | timeout | error

### Result Comparator (`comparison/comparator.py`)
Identifies discrepancies such as:
- News reports "strong growth" but SEC shows declining revenue
- News sentiment is positive but SEC risk factors highlight concerns
- Forward guidance in SEC differs from news narrative

### Response Synthesizer (`synthesis/synthesizer.py`)
Single LLM call that:
1. Integrates insights from both agents
2. Highlights agreements between sources
3. Clearly notes discrepancies with citations
4. Indicates confidence level based on source agreement

## API Endpoints

### POST /query
```json
Request:
{
  "query": "What's the outlook for Apple?",
  "ticker": "AAPL",
  "sources": ["news", "sec"],
  "enable_comparison": true
}

Response:
{
  "answer": "...",
  "agents_used": ["news_agent", "sec_agent"],
  "had_discrepancies": true,
  "execution_time_ms": 3500,
  "session_id": "abc123"
}
```

### GET /health
```json
{"status": "healthy", "service": "orchestrator"}
```

## CLI Commands

```bash
# Interactive chat
orchestrator chat

# One-shot query
orchestrator query "What are the risks for Tesla?" --ticker TSLA --source both

# Explicit comparison
orchestrator compare AAPL
```

## Environment Variables
```
BEDROCK_MODEL_ID=anthropic.claude-opus-4-5-20251101-v1:0
ORCHESTRATOR_TIMEOUT_SECONDS=60
ORCHESTRATOR_ENABLE_COMPARISON=true
```

## Dependencies
Uses existing project dependencies (boto3, fastapi, click, asyncio).

## Success Criteria
1. Query routing correctly identifies agent(s) needed
2. Parallel execution reduces latency vs sequential calls
3. Graceful handling of agent failures with partial results
4. Accurate discrepancy detection between news and SEC data
5. Clear, synthesized responses with proper citations
