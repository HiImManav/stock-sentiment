"""CLI interface for the SEC filings agent.

Provides interactive chat mode, one-shot queries, and utility commands.
"""

from __future__ import annotations

import json
import sys

import click

from sec_agent.agent import SECFilingsAgent
from sec_agent.tools.fetch_filing import fetch_and_parse_filing
from sec_agent.tools.query_section import list_available_filings


@click.group()
def cli() -> None:
    """SEC filings analysis agent — query SEC EDGAR filings with AI."""


@cli.command()
def chat() -> None:
    """Start an interactive chat session with the SEC filings agent."""
    click.echo("SEC Filings Agent — interactive mode")
    click.echo("Type 'exit' or 'quit' to end the session.\n")

    agent = SECFilingsAgent()

    while True:
        try:
            user_input = click.prompt("You", prompt_suffix="> ")
        except (EOFError, KeyboardInterrupt):
            click.echo("\nGoodbye.")
            break

        if user_input.strip().lower() in ("exit", "quit"):
            click.echo("Goodbye.")
            break

        if not user_input.strip():
            continue

        try:
            response = agent.query(user_input)
            click.echo(f"\nAgent: {response}\n")
        except Exception as e:
            click.echo(f"\nError: {e}\n", err=True)


@cli.command()
@click.argument("question")
@click.option("--ticker", "-t", required=True, help="Stock ticker symbol (e.g., AAPL).")
@click.option(
    "--filing",
    "-f",
    "filing_type",
    required=True,
    type=click.Choice(["10-K", "10-Q", "8-K"], case_sensitive=False),
    help="SEC filing type.",
)
@click.option("--section", "-s", default=None, help="Section filter (e.g., '1A' for Risk Factors).")
def query(question: str, ticker: str, filing_type: str, section: str | None) -> None:
    """Ask a one-shot question about a company's SEC filing.

    QUESTION is the natural language question to ask.
    """
    agent = SECFilingsAgent()

    filing_type = filing_type.upper()
    prompt = (
        f"Fetch the latest {filing_type} filing for {ticker.upper()} and answer this question"
    )
    if section:
        prompt += f" (focus on section {section})"
    prompt += f": {question}"

    try:
        response = agent.query(prompt)
        click.echo(response)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("ticker")
@click.argument("filing_type", type=click.Choice(["10-K", "10-Q", "8-K"], case_sensitive=False))
@click.option("--index", "-i", "filing_index", default=0, help="Filing index (0=most recent).")
def fetch(ticker: str, filing_type: str, filing_index: int) -> None:
    """Pre-fetch and cache a SEC filing for a ticker.

    TICKER is the stock ticker symbol (e.g., AAPL).
    FILING_TYPE is the type of filing (10-K, 10-Q, or 8-K).
    """
    click.echo(f"Fetching {filing_type.upper()} for {ticker.upper()}...")

    try:
        result = fetch_and_parse_filing(
            ticker=ticker,
            filing_type=filing_type.upper(),
            filing_index=filing_index,
        )
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if result.get("status") == "error":
        click.echo(f"Error: {result.get('message', 'Unknown error')}", err=True)
        sys.exit(1)

    source = result.get("source", "unknown")
    sections = result.get("sections_found", [])
    chunk_count = result.get("chunk_count", 0)
    filing_date = result.get("filing_date", "unknown")

    click.echo(f"Source: {source}")
    click.echo(f"Filing date: {filing_date}")
    click.echo(f"Sections found: {', '.join(sections) if sections else 'none'}")
    click.echo(f"Chunks: {chunk_count}")


@cli.command("list-filings")
@click.argument("ticker")
def list_filings(ticker: str) -> None:
    """Show cached filings for a ticker.

    TICKER is the stock ticker symbol (e.g., AAPL).
    """
    try:
        result = list_available_filings(ticker=ticker)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    filings = result.get("cached_filings", [])
    if not filings:
        click.echo(f"No cached filings for {ticker.upper()}.")
        return

    click.echo(f"Cached filings for {ticker.upper()}:")
    for key in filings:
        click.echo(f"  {key}")
