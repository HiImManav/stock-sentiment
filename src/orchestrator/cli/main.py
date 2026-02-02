"""CLI interface for the orchestration agent.

Provides interactive chat mode, one-shot queries, and comparison commands.
"""

from __future__ import annotations

import json
import sys
from typing import Literal

import click

from orchestrator.agent import OrchestrationAgent, OrchestrationResult


@click.group()
def cli() -> None:
    """Orchestration Agent - Unified news and SEC filings analysis."""


@cli.command()
def chat() -> None:
    """Start an interactive chat session with the orchestration agent."""
    click.echo("Orchestration Agent - Interactive Mode")
    click.echo("Coordinates news sentiment and SEC filings analysis.")
    click.echo("Type 'exit' or 'quit' to end the session.\n")

    agent = OrchestrationAgent()

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

        # Check for special commands
        if user_input.strip().lower() == "summary":
            _print_session_summary(agent)
            continue

        if user_input.strip().lower() == "history":
            _print_query_history(agent)
            continue

        if user_input.strip().lower() == "reset":
            agent.reset()
            click.echo("Session reset.\n")
            continue

        try:
            result = agent.query(user_input)
            _print_result(result)
        except Exception as e:
            click.echo(f"\nError: {e}\n", err=True)


@cli.command()
@click.argument("question")
@click.option(
    "--ticker",
    "-t",
    default=None,
    help="Stock ticker symbol (e.g., AAPL). Auto-detected if not provided.",
)
@click.option(
    "--source",
    "-s",
    type=click.Choice(["news", "sec", "both"], case_sensitive=False),
    default=None,
    help="Force specific source(s). Default: auto-route based on query.",
)
@click.option(
    "--json-output",
    "-j",
    is_flag=True,
    help="Output as JSON.",
)
def query(
    question: str,
    ticker: str | None,
    source: str | None,
    json_output: bool,
) -> None:
    """Ask a one-shot question about a company.

    QUESTION is the natural language question to ask.

    Examples:
        orchestrator query "What's the outlook for Apple?"
        orchestrator query "What are Tesla's risk factors?" -t TSLA -s sec
        orchestrator query "Compare news vs SEC for AAPL" -s both -j
    """
    agent = OrchestrationAgent()

    # Map source to force_route
    force_route: Literal["news_only", "sec_only", "both"] | None = None
    if source:
        source_lower = source.lower()
        if source_lower == "news":
            force_route = "news_only"
        elif source_lower == "sec":
            force_route = "sec_only"
        elif source_lower == "both":
            force_route = "both"

    try:
        result = agent.query(
            question,
            ticker=ticker.upper() if ticker else None,
            force_route=force_route,
        )

        if json_output:
            click.echo(json.dumps(result.to_dict(), indent=2, default=str))
        else:
            _print_result(result)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("ticker")
@click.option(
    "--json-output",
    "-j",
    is_flag=True,
    help="Output as JSON.",
)
def compare(ticker: str, json_output: bool) -> None:
    """Run an explicit comparison between news and SEC data for a ticker.

    TICKER is the stock ticker symbol (e.g., AAPL).

    This command forces both news and SEC agents to run and highlights
    any discrepancies between the sources.

    Examples:
        orchestrator compare AAPL
        orchestrator compare TSLA -j
    """
    agent = OrchestrationAgent()

    try:
        click.echo(f"Analyzing {ticker.upper()} from news and SEC sources...")
        result = agent.compare(ticker.upper())

        if json_output:
            click.echo(json.dumps(result.to_dict(), indent=2, default=str))
        else:
            _print_comparison_result(result, ticker.upper())

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def _print_result(result: OrchestrationResult) -> None:
    """Pretty-print an orchestration result."""
    click.echo("")

    # Show agents used
    agents_str = ", ".join(result.agents_used) if result.agents_used else "none"
    click.echo(f"[Agents: {agents_str} | Route: {result.route_type}]")

    # Show confidence
    confidence_color = "green" if result.confidence >= 0.7 else "yellow" if result.confidence >= 0.4 else "red"
    click.secho(
        f"[Confidence: {result.confidence:.0%}]",
        fg=confidence_color,
        nl=False,
    )

    # Show discrepancy indicator
    if result.had_discrepancies:
        click.secho(" [Discrepancies detected]", fg="yellow")
    else:
        click.echo("")

    click.echo("-" * 60)
    click.echo(result.response)
    click.echo("-" * 60)
    click.echo(f"[Execution time: {result.execution_time_ms:.0f}ms]\n")


def _print_comparison_result(result: OrchestrationResult, ticker: str) -> None:
    """Pretty-print a comparison result with more detail."""
    click.echo("")
    click.echo("=" * 60)
    click.echo(f"COMPARISON ANALYSIS: {ticker}")
    click.echo("=" * 60)

    # Show agents used
    agents_str = ", ".join(result.agents_used) if result.agents_used else "none"
    click.echo(f"\nSources: {agents_str}")

    # Show confidence
    confidence_color = "green" if result.confidence >= 0.7 else "yellow" if result.confidence >= 0.4 else "red"
    click.echo("Confidence: ", nl=False)
    click.secho(f"{result.confidence:.0%}", fg=confidence_color)

    # Show discrepancy status
    click.echo("Discrepancies: ", nl=False)
    if result.had_discrepancies:
        click.secho("Yes - sources disagree on some points", fg="yellow")
    else:
        click.secho("No - sources are aligned", fg="green")

    # Show comparison details if available
    if result.comparison:
        click.echo("\n" + "-" * 40)
        click.echo("COMPARISON DETAILS")
        click.echo("-" * 40)

        # Show alignment score
        alignment = result.comparison.overall_alignment
        alignment_color = "green" if alignment > 0.3 else "yellow" if alignment > -0.3 else "red"
        click.echo("Alignment Score: ", nl=False)
        click.secho(f"{alignment:+.2f}", fg=alignment_color)

        # Show summary
        if result.comparison.summary:
            click.echo(f"\n{result.comparison.summary}")

        # Show discrepancies
        if result.comparison.discrepancies:
            click.echo("\n" + "-" * 40)
            click.echo("DISCREPANCIES")
            click.echo("-" * 40)
            for disc in result.comparison.discrepancies[:5]:
                severity_color = {
                    "HIGH": "red",
                    "MEDIUM": "yellow",
                    "LOW": "cyan",
                }.get(disc.severity.name, "white")
                click.secho(f"  [{disc.severity.name}] ", fg=severity_color, nl=False)
                click.echo(disc.description)

        # Show agreements
        if result.comparison.agreements:
            click.echo("\n" + "-" * 40)
            click.echo("AGREEMENTS")
            click.echo("-" * 40)
            for agreement in result.comparison.agreements[:5]:
                click.echo(f"  • {agreement.description}")

    # Show synthesized response
    click.echo("\n" + "-" * 40)
    click.echo("SYNTHESIZED ANALYSIS")
    click.echo("-" * 40)
    click.echo(result.response)

    click.echo("\n" + "=" * 60)
    click.echo(f"[Execution time: {result.execution_time_ms:.0f}ms]")
    click.echo("")


def _print_session_summary(agent: OrchestrationAgent) -> None:
    """Print session summary."""
    summary = agent.get_session_summary()
    click.echo("\n" + "=" * 40)
    click.echo("SESSION SUMMARY")
    click.echo("=" * 40)
    click.echo(f"Session ID: {summary.get('session_id', 'N/A')}")
    click.echo(f"Total queries: {summary.get('query_count', 0)}")
    click.echo(f"Tickers analyzed: {', '.join(summary.get('tickers_analyzed', [])) or 'none'}")
    click.echo(f"Queries with discrepancies: {summary.get('discrepancy_count', 0)}")
    click.echo(f"Average confidence: {summary.get('average_confidence', 0):.0%}")
    click.echo("=" * 40 + "\n")


def _print_query_history(agent: OrchestrationAgent) -> None:
    """Print recent query history."""
    queries = agent.get_recent_queries(limit=10)
    click.echo("\n" + "=" * 40)
    click.echo("RECENT QUERIES")
    click.echo("=" * 40)
    if not queries:
        click.echo("No queries in this session.")
    else:
        for i, q in enumerate(queries, 1):
            ticker = q.get("ticker") or "N/A"
            query_text = q.get("user_query", "")[:50]
            if len(q.get("user_query", "")) > 50:
                query_text += "..."
            confidence = q.get("confidence", 0)
            discrepancy = "⚠" if q.get("had_discrepancies") else "✓"
            click.echo(f"  {i}. [{ticker}] {query_text}")
            click.echo(f"     Confidence: {confidence:.0%} {discrepancy}")
    click.echo("=" * 40 + "\n")


if __name__ == "__main__":
    cli()
