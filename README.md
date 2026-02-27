# Stock Sentiment

A unified AI agent platform for stock analysis that combines SEC filings intelligence with real-time news sentiment. Three coordinated agents — SEC, News, and Orchestrator — work together to deliver comprehensive company insights with discrepancy detection between regulatory filings and market news.

## Architecture

```
                        ┌─────────────────────┐
                        │  Orchestration Agent │
                        │  (Query Routing &    │
                        │   Synthesis)         │
                        └────────┬────────────┘
                                 │
                    ┌────────────┼────────────┐
                    │                         │
           ┌───────▼────────┐       ┌────────▼───────┐
           │  SEC Filings   │       │  News Sentiment │
           │  Agent         │       │  Agent          │
           └───────┬────────┘       └────────┬───────┘
                   │                         │
           ┌───────▼────────┐       ┌────────▼───────┐
           │  SEC EDGAR API │       │  NewsAPI.org    │
           └────────────────┘       └────────────────┘
```

**SEC Filings Agent** — Downloads and parses SEC EDGAR filings (10-K, 10-Q, 8-K), chunks them into sections, generates Titan embeddings, and performs semantic search via FAISS.

**News Sentiment Agent** — Fetches recent articles from NewsAPI, analyzes sentiment, extracts key claims, identifies material events, and tracks sentiment trends over time.

**Orchestration Agent** — Routes queries to the right agent(s), runs them in parallel, extracts signals from both sources, detects discrepancies, and synthesizes a unified response with confidence scoring.

## Prerequisites

- Python 3.11+
- AWS account with Bedrock access (Claude and Titan Embeddings models enabled)
- [NewsAPI](https://newsapi.org/) API key

## Setup

```bash
# Clone and set up virtual environment
git clone <repo-url> && cd stock-sentiment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Configure environment
cp .env.example .env  # then edit with your values
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `SEC_EDGAR_USER_AGENT` | SEC EDGAR contact info (required by SEC) |
| `AWS_REGION` | AWS region (default: `us-east-1`) |
| `SEC_FILINGS_BUCKET` | S3 bucket for filing cache |
| `NEWS_SENTIMENT_BUCKET` | S3 bucket for news cache |
| `BEDROCK_MODEL_ID` | Bedrock Claude model ID |
| `EMBEDDING_MODEL_ID` | Bedrock Titan embedding model ID |
| `NEWSAPI_KEY` | NewsAPI.org API key |
| `ORCHESTRATOR_TIMEOUT_SECONDS` | Agent execution timeout (default: `60`) |
| `ORCHESTRATOR_ENABLE_COMPARISON` | Enable discrepancy detection (default: `true`) |

### S3 Buckets

```bash
aws s3 mb s3://sec-filings-cache --region us-east-1
aws s3 mb s3://news-sentiment-cache-$(aws sts get-caller-identity --query Account --output text) --region us-east-1
```

## Usage

### CLI

```bash
# Orchestrator (recommended entry point)
orchestrator chat                                          # Interactive mode
orchestrator query "What's the outlook for Apple?" -t AAPL # Single query
orchestrator compare AAPL                                  # News vs SEC comparison

# SEC Agent
sec-agent chat
sec-agent query "What are Apple's risk factors?" --ticker AAPL --filing 10-K
sec-agent fetch AAPL 10-K
sec-agent list AAPL

# News Agent
news-agent chat
news-agent query "Recent news about Tesla" -t TSLA
```

### REST API

```bash
# Start the orchestrator API
uvicorn orchestrator.api.server:app --port 8000
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Execute an orchestrated query |
| `/compare` | POST | Compare news vs SEC signals |
| `/health` | GET | Health check |
| `/session/summary` | GET | Session statistics |
| `/session/history` | GET | Query history |
| `/reset` | POST | Reset session |

### Programmatic

```python
from orchestrator import OrchestrationAgent

agent = OrchestrationAgent()
result = agent.query("What's the outlook for Apple?", ticker="AAPL")

print(result.response)
print(f"Confidence: {result.confidence:.0%}")
print(f"Agents used: {result.agents_used}")
print(f"Discrepancies: {result.had_discrepancies}")
```

## Development

```bash
# Run tests
pytest

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

## Infrastructure

AWS CDK stack in `infra/` provisions S3, Lambda (via Mangum), API Gateway, and IAM roles.

```bash
cd infra && cdk deploy
```

## Project Structure

```
src/
├── sec_agent/          # SEC filings analysis agent
│   ├── parser/         #   EDGAR client, HTML parsing, chunking
│   ├── retrieval/      #   Embeddings, FAISS vector store, S3 cache
│   ├── tools/          #   Bedrock tool definitions
│   ├── memory/         #   Session memory
│   ├── api/            #   FastAPI server
│   └── cli/            #   Click CLI
├── news_agent/         # News sentiment analysis agent
│   ├── news/           #   NewsAPI client, caching, entity resolution
│   ├── analysis/       #   Sentiment, claims, materiality, trends
│   ├── tools/          #   Bedrock tool definitions
│   ├── memory/         #   Session memory
│   ├── api/            #   FastAPI server
│   └── cli/            #   Click CLI
└── orchestrator/       # Coordination layer
    ├── routing/        #   Query classification
    ├── execution/      #   Parallel agent execution
    ├── comparison/     #   Signal extraction & discrepancy detection
    ├── synthesis/      #   LLM response synthesis
    ├── memory/         #   Session memory
    ├── api/            #   FastAPI server
    └── cli/            #   Click CLI
```
