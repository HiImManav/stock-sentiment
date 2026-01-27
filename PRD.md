# SEC Filings Agent — Product Requirements Document

## Status Tracker

| # | Component | Status |
|---|-----------|--------|
| 1 | Project scaffolding (pyproject.toml, directory structure) | DONE |
| 2 | SEC EDGAR fetcher (download 10-K, 10-Q, 8-K filings) | DONE |
| 3 | Filing parser (HTML/SGML → section extraction) | DONE |
| 4 | Section-aware chunking engine | DONE |
| 5 | S3 storage layer (cache parsed chunks with metadata) | DONE |
| 6 | Embedding + FAISS retrieval at query time | DONE |
| 7 | AgentCore Bedrock agent (tool definitions, orchestration) | DONE |
| 8 | AgentCore memory integration (short-term memory) | DONE |
| 9 | CLI interface | NOT STARTED |
| 10 | REST API (FastAPI) | NOT STARTED |
| 11 | AWS CDK infrastructure stack | NOT STARTED |
| 12 | Tests | NOT STARTED |

---

## 1. Overview

An AI agent built on **AWS AgentCore with Bedrock** that fetches SEC EDGAR filings (10-K, 10-Q, 8-K), parses them into structured sections, and answers natural language questions about company risks, management discussion, material events, and financials.

### Key Design Decisions
- **Model**: Claude Opus 4.5 via Bedrock
- **Interface**: CLI + REST API (FastAPI)
- **Retrieval**: Custom S3 + section-aware chunking + Bedrock Titan embeddings + FAISS
- **Fetch mode**: On-demand (fetch at query time for any ticker)
- **Language**: Python
- **IaC**: AWS CDK
- **Memory**: AgentCore memory for short-term context (previous queries, tickers analyzed)

---

## 2. Architecture

```
User (CLI / API)
       │
       ▼
┌──────────────────────┐
│  Agent Orchestrator   │  (AgentCore Bedrock Agent)
│  - Tool routing       │
│  - Conversation mgmt  │
│  - AgentCore Memory   │
└──────┬───────────────┘
       │ calls tools
       ▼
┌──────────────────────────────────────────────┐
│                  Agent Tools                  │
│                                              │
│  ┌─────────────┐  ┌──────────────────────┐   │
│  │ fetch_filing │  │ query_filing_section │   │
│  │ (SEC EDGAR) │  │ (retrieve + answer)  │   │
│  └──────┬──────┘  └──────────┬───────────┘   │
│         │                    │               │
│         ▼                    ▼               │
│  ┌─────────────┐  ┌──────────────────────┐   │
│  │   Parser    │  │  Embedding + FAISS   │   │
│  │ (sections)  │  │  (semantic search)   │   │
│  └──────┬──────┘  └──────────┬───────────┘   │
│         │                    │               │
│         ▼                    ▼               │
│  ┌──────────────────────────────────────┐    │
│  │         S3 (chunk cache)             │    │
│  │  s3://bucket/filings/{cik}/{type}/   │    │
│  │         {accession}/chunks.json      │    │
│  └──────────────────────────────────────┘    │
└──────────────────────────────────────────────┘
```

---

## 3. Components (Detailed)

### 3.1 Project Structure

```
stock-sentiment/
├── PRD.md
├── pyproject.toml
├── src/
│   └── sec_agent/
│       ├── __init__.py
│       ├── agent.py              # AgentCore Bedrock agent definition
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── fetch_filing.py   # SEC EDGAR download tool
│       │   └── query_section.py  # Retrieval + answer tool
│       ├── parser/
│       │   ├── __init__.py
│       │   ├── edgar_client.py   # SEC EDGAR API client
│       │   ├── filing_parser.py  # HTML/SGML → sections
│       │   └── chunker.py        # Section-aware chunking
│       ├── retrieval/
│       │   ├── __init__.py
│       │   ├── embeddings.py     # Bedrock Titan embeddings
│       │   ├── vector_store.py   # FAISS wrapper
│       │   └── s3_cache.py       # S3 chunk storage
│       ├── memory/
│       │   ├── __init__.py
│       │   └── agent_memory.py   # AgentCore memory wrapper
│       ├── api/
│       │   ├── __init__.py
│       │   └── server.py         # FastAPI REST API
│       └── cli/
│           ├── __init__.py
│           └── main.py           # CLI entry point
├── infra/
│   ├── app.py                    # CDK app entry
│   └── stacks/
│       └── sec_agent_stack.py    # CDK stack
└── tests/
    ├── test_parser.py
    ├── test_chunker.py
    ├── test_retrieval.py
    └── test_agent.py
```

### 3.2 SEC EDGAR Fetcher (`edgar_client.py`)

Fetches filings from SEC EDGAR using their public REST API.

- **Endpoint**: `https://efts.sec.gov/LATEST/search-index?q=...` and full-text search API
- **CIK lookup**: `https://www.sec.gov/cgi-bin/browse-edgar?company={ticker}&CIK=&type=&dateb=&owner=include&count=40&search_text=&action=getcompany` or the company tickers JSON at `https://www.sec.gov/files/company_tickers.json`
- **Filing index**: `https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type={filing_type}&dateb=&owner=include&count=10`
- **Full filing**: Download from EDGAR archives
- **Rate limiting**: SEC requires ≤10 req/sec, must set `User-Agent` header with company name and email
- **User-Agent**: Configurable via env var `SEC_EDGAR_USER_AGENT` (e.g., `"CompanyName admin@company.com"`)

Functions:
- `get_cik(ticker: str) -> str` — resolve ticker to CIK
- `list_filings(cik: str, filing_type: str, count: int = 5) -> list[FilingMetadata]` — list recent filings
- `download_filing(accession_number: str) -> str` — download full filing HTML

### 3.3 Filing Parser (`filing_parser.py`)

Parses SEC filing HTML/SGML into structured sections.

**Target sections by filing type:**

| Filing | Item | Section Name |
|--------|------|-------------|
| 10-K | 1 | Business |
| 10-K | 1A | Risk Factors |
| 10-K | 7 | MD&A |
| 10-K | 8 | Financial Statements |
| 10-Q | 1 | Financial Statements |
| 10-Q | 1A | Risk Factors |
| 10-Q | 2 | MD&A |
| 8-K | varies | Item 1.01–9.01 (material events) |

**Parsing strategy:**
1. Use `beautifulsoup4` to parse HTML
2. Identify section headers via regex patterns:
   - `r"(?:Item|ITEM)\s*(\d+[A-Za-z]?)[\.\:\s]"` for section numbers
   - Also match common header text like "RISK FACTORS", "MANAGEMENT'S DISCUSSION"
3. Extract text between consecutive section headers
4. Handle edge cases:
   - Table of contents links (skip ToC, find actual content)
   - Nested tables (flatten or extract text)
   - Multiple documents in one filing (find the main document via `<DOCUMENT>` tags)
   - Exhibits vs. main document

Functions:
- `parse_filing(html: str, filing_type: str) -> list[Section]` — returns list of Section objects
- `Section` dataclass: `name: str, item_number: str, text: str, start_pos: int, end_pos: int`

### 3.4 Section-Aware Chunker (`chunker.py`)

Chunks sections into retrieval-friendly pieces while preserving section context.

**Strategy:**
- Chunk size: ~1500 tokens (adjustable)
- Overlap: 200 tokens between chunks
- Each chunk carries metadata: `{section_name, item_number, filing_type, ticker, accession_number, chunk_index}`
- Prefer splitting at paragraph boundaries (double newline)
- Never split mid-sentence if avoidable (fall back to sentence boundary via regex)
- Tables get their own chunks (never split a table row across chunks)

Functions:
- `chunk_section(section: Section, max_tokens: int = 1500, overlap: int = 200) -> list[Chunk]`
- `chunk_filing(sections: list[Section], ...) -> list[Chunk]`
- `Chunk` dataclass: `text: str, metadata: dict, token_count: int`

### 3.5 S3 Cache (`s3_cache.py`)

Stores and retrieves parsed/chunked filings from S3.

**Key schema**: `s3://{bucket}/filings/{ticker}/{filing_type}/{accession_number}/chunks.json`

The JSON file contains:
```json
{
  "ticker": "AAPL",
  "cik": "0000320193",
  "filing_type": "10-K",
  "accession_number": "0000320193-23-000106",
  "filing_date": "2023-11-03",
  "sections": ["1A", "7", "8"],
  "chunks": [
    {
      "text": "...",
      "section_name": "Risk Factors",
      "item_number": "1A",
      "chunk_index": 0,
      "token_count": 1487
    }
  ]
}
```

Functions:
- `get_cached_filing(ticker, filing_type, accession) -> FilingChunks | None`
- `cache_filing(filing_chunks: FilingChunks) -> None`
- `list_cached_filings(ticker: str) -> list[str]`

**S3 bucket**: Created by CDK stack, name from env var `SEC_FILINGS_BUCKET`.

### 3.6 Embeddings (`embeddings.py`)

Uses **Bedrock Titan Text Embeddings V2** (`amazon.titan-embed-text-v2:0`).

- Dimension: 1024
- Batch embed chunks for efficiency (Titan supports batch)
- Cache embeddings alongside chunks in S3 (add `embeddings` field to chunks.json as base64-encoded numpy arrays)

Functions:
- `embed_texts(texts: list[str]) -> np.ndarray` — returns (N, 1024) array
- `embed_query(query: str) -> np.ndarray` — returns (1, 1024) array

### 3.7 Vector Store (`vector_store.py`)

FAISS-based in-memory vector search, built per-query from cached embeddings.

Functions:
- `build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP` — cosine similarity index
- `search(index, query_embedding, chunks, top_k=10) -> list[Chunk]` — returns ranked chunks
- Optionally filter by section before search (e.g., only search Risk Factors chunks)

### 3.8 AgentCore Bedrock Agent (`agent.py`)

The core agent using AWS AgentCore Bedrock scaffolding.

**Agent definition:**
- Uses `bedrock-agentcore` SDK
- Model: `anthropic.claude-opus-4-5-20251101-v1:0` (Bedrock model ID for Opus 4.5)
- System prompt defines the agent's role as SEC filing analyst
- Tools registered:
  1. `fetch_and_parse_filing` — fetches a filing from EDGAR, parses, chunks, caches in S3
  2. `query_filing` — semantic search over cached chunks + LLM answer
  3. `list_available_filings` — lists what's cached for a ticker
  4. `get_memory` — retrieve previous analysis from AgentCore memory
  5. `save_memory` — store analysis results to AgentCore memory

**System prompt** (summary):
```
You are an SEC filings analyst. You help users understand company risks,
management discussion, material events, and financial statements from
SEC filings (10-K, 10-Q, 8-K).

When a user asks about a company:
1. Check memory for recent analysis of the same company
2. Fetch the relevant filing if not cached
3. Query the appropriate sections
4. Synthesize a clear answer with specific citations (section, page)
5. Save key findings to memory for future reference

Always cite the specific filing and section you're referencing.
```

**Tool schemas:**

```python
fetch_and_parse_filing_schema = {
    "name": "fetch_and_parse_filing",
    "description": "Fetch a SEC filing from EDGAR, parse it into sections, and cache it. Call this before querying.",
    "parameters": {
        "ticker": {"type": "string", "description": "Stock ticker symbol (e.g., AAPL)"},
        "filing_type": {"type": "string", "enum": ["10-K", "10-Q", "8-K"], "description": "Type of SEC filing"},
        "filing_index": {"type": "integer", "description": "0 = most recent, 1 = second most recent, etc.", "default": 0}
    },
    "required": ["ticker", "filing_type"]
}

query_filing_schema = {
    "name": "query_filing",
    "description": "Search a cached SEC filing and answer a question about it.",
    "parameters": {
        "ticker": {"type": "string"},
        "filing_type": {"type": "string", "enum": ["10-K", "10-Q", "8-K"]},
        "question": {"type": "string", "description": "Natural language question about the filing"},
        "section_filter": {"type": "string", "description": "Optional: limit search to specific section (e.g., '1A' for Risk Factors)", "default": null}
    },
    "required": ["ticker", "filing_type", "question"]
}
```

### 3.9 AgentCore Memory (`agent_memory.py`)

Uses AgentCore's memory service for short-term context.

**What to store:**
- Previous tickers analyzed in this session
- Key findings per company (risk summary, notable events)
- User preferences (e.g., focus on specific sectors)

**Memory schema:**
```json
{
  "session_id": "...",
  "entries": [
    {
      "ticker": "AAPL",
      "filing_type": "10-K",
      "timestamp": "2026-01-27T...",
      "summary": "Key risks: supply chain concentration in China, regulatory...",
      "sections_analyzed": ["1A", "7"]
    }
  ]
}
```

Functions:
- `store_analysis(ticker, filing_type, summary, sections) -> None`
- `get_recent_analyses(ticker: str = None) -> list[MemoryEntry]`
- `get_session_context() -> dict` — returns all memory for current session

### 3.10 CLI (`cli/main.py`)

Uses `click` for CLI interface.

```bash
# Interactive mode
sec-agent chat

# One-shot queries
sec-agent query "What risks has Apple disclosed?" --ticker AAPL --filing 10-K
sec-agent query "Any recent material events for Tesla?" --ticker TSLA --filing 8-K

# Utility commands
sec-agent fetch AAPL 10-K          # Pre-fetch and cache
sec-agent list-filings AAPL        # Show cached filings
```

### 3.11 REST API (`api/server.py`)

FastAPI server with these endpoints:

```
POST /query
  Body: {"ticker": "AAPL", "filing_type": "10-K", "question": "What are the risk factors?"}
  Response: {"answer": "...", "sources": [...], "session_id": "..."}

POST /fetch
  Body: {"ticker": "AAPL", "filing_type": "10-K"}
  Response: {"status": "ok", "sections_found": ["1A", "7", "8"], "chunk_count": 45}

GET /filings/{ticker}
  Response: {"cached_filings": [...]}

GET /health
  Response: {"status": "healthy"}
```

### 3.12 CDK Stack (`infra/stacks/sec_agent_stack.py`)

AWS resources:
- **S3 Bucket**: For filing chunk cache (`sec-filings-cache-{account}`)
- **Lambda Function** (or ECS Fargate): Hosts the FastAPI API
- **IAM Roles**: Bedrock invoke, S3 read/write, AgentCore access
- **Bedrock Agent**: AgentCore agent registration
- **API Gateway**: REST API in front of Lambda/Fargate
- **CloudWatch**: Logging

CDK will use `aws-cdk-lib` with Python bindings.

---

## 4. Configuration

Environment variables:
```
SEC_EDGAR_USER_AGENT=CompanyName admin@company.com
SEC_FILINGS_BUCKET=sec-filings-cache
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=anthropic.claude-opus-4-5-20251101-v1:0
EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0
AGENTCORE_MEMORY_NAMESPACE=sec-agent
```

---

## 5. Dependencies

```
boto3
beautifulsoup4
lxml
faiss-cpu
numpy
click
fastapi
uvicorn
mangum          # Lambda adapter for FastAPI
httpx           # Async HTTP for EDGAR
tiktoken        # Token counting
aws-cdk-lib     # CDK (infra only)
constructs      # CDK (infra only)
pytest
moto            # AWS mocking for tests
```

---

## 6. Key Challenges & Mitigations

| Challenge | Mitigation |
|-----------|-----------|
| Section boundaries vary across companies | Multiple regex patterns + fallback to header text matching + LLM-based section identification as last resort |
| Filings are huge (10-K can be 200+ pages) | Section-level extraction first, then chunking. Never load full filing into LLM context |
| SEC rate limiting (10 req/sec) | Implement rate limiter with backoff. Cache aggressively in S3 |
| SGML vs HTML format differences | Handle both with beautifulsoup + custom SGML preprocessing |
| Tables and financial data | Preserve table structure in chunks, use markdown table format |
| Older filings use different formats | Focus on filings from 2010+ which are mostly XHTML. Add format detection |

---

## 7. Sample Queries the Agent Should Handle

1. "What risks has Apple management disclosed?" → Fetches latest AAPL 10-K, searches Item 1A
2. "Any recent material events for Tesla?" → Fetches latest TSLA 8-K, extracts all items
3. "Compare risk factors between Google and Microsoft" → Fetches both 10-Ks, compares Item 1A
4. "What did Amazon's MD&A say about AWS growth?" → Fetches AMZN 10-K, searches Item 7
5. "Show me Meta's latest financial statements summary" → Fetches META 10-K, searches Item 8
6. "What companies have I analyzed so far?" → Reads AgentCore memory

---

## 8. Implementation Order

1. Project scaffolding
2. SEC EDGAR client (fetch filings)
3. Filing parser (HTML → sections)
4. Chunker (sections → chunks)
5. S3 cache layer
6. Embeddings + FAISS retrieval
7. AgentCore Bedrock agent with tools
8. AgentCore memory integration
9. CLI interface
10. REST API
11. CDK infrastructure stack
12. Tests
