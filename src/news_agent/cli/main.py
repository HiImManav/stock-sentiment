"""CLI commands for the News Sentiment Agent."""

from __future__ import annotations

import json

import click

from news_agent.agent import NewsSentimentAgent
from news_agent.tools.analyze import analyze_sentiment
from news_agent.tools.fetch_news import fetch_news
from news_agent.tools.trends import get_trends


@click.group()
def cli() -> None:
    """News Sentiment Agent - Analyze news sentiment for companies."""
    pass


@cli.command()
def chat() -> None:
    """Start an interactive chat session with the agent."""
    agent = NewsSentimentAgent()
    click.echo("News Sentiment Agent - Interactive Mode")
    click.echo("Type 'quit' or 'exit' to end the session.\n")

    while True:
        try:
            user_input = click.prompt("You", prompt_suffix="> ")
        except (EOFError, KeyboardInterrupt):
            click.echo("\nGoodbye!")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            click.echo("Goodbye!")
            break

        if not user_input.strip():
            continue

        try:
            response = agent.query(user_input)
            click.echo(f"\nAgent: {response}\n")
        except Exception as e:
            click.echo(f"\nError: {e}\n", err=True)


@cli.command()
@click.argument("ticker")
@click.option("--days", "-d", default=30, help="Days to look back (default: 30)")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def analyze(ticker: str, days: int, json_output: bool) -> None:
    """Analyze news sentiment for a company.

    Example: news-agent analyze AAPL
    """
    click.echo(f"Fetching news for {ticker.upper()}...")

    # Fetch news
    fetch_result = fetch_news(ticker, days_back=days)

    if fetch_result["status"] == "error":
        click.echo(f"Error: {fetch_result.get('message', 'Unknown error')}", err=True)
        raise SystemExit(1)

    if fetch_result["status"] == "insufficient_data":
        click.echo(f"Warning: {fetch_result.get('message')}")
        if json_output:
            click.echo(json.dumps(fetch_result, indent=2))
        raise SystemExit(0)

    article_count = fetch_result.get("article_count", 0)
    click.echo(f"Found {article_count} articles. Analyzing sentiment...")

    # Analyze sentiment
    result = analyze_sentiment(ticker)

    if result["status"] != "ok":
        click.echo(f"Error: {result.get('message', 'Analysis failed')}", err=True)
        raise SystemExit(1)

    if json_output:
        click.echo(json.dumps(result["result"], indent=2))
    else:
        _print_analysis_result(result["result"])


@cli.command()
@click.argument("ticker")
@click.option("--days", "-d", default=30, help="Days to look back (default: 30)")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def trends(ticker: str, days: int, json_output: bool) -> None:
    """Show sentiment trends for a company.

    Example: news-agent trends TSLA --days 30
    """
    result = get_trends(ticker, days_back=days)

    if json_output:
        click.echo(json.dumps(result, indent=2))
    else:
        _print_trend_result(result)


@cli.command()
@click.argument("ticker")
@click.option("--days", "-d", default=30, help="Days to look back (default: 30)")
@click.option("--force", "-f", is_flag=True, help="Force refresh from API")
def fetch(ticker: str, days: int, force: bool) -> None:
    """Pre-fetch and cache news articles.

    Example: news-agent fetch AAPL --days 30
    """
    click.echo(f"Fetching news for {ticker.upper()}...")

    result = fetch_news(ticker, days_back=days, force_refresh=force)

    if result["status"] == "ok":
        click.echo(f"✓ Fetched {result['article_count']} articles")
        click.echo(f"  Company: {result.get('company_name', ticker)}")
        click.echo(f"  Period: {result.get('date_range', 'N/A')}")
        click.echo(f"  Cache hit: {result.get('cache_hit', False)}")
        if result.get("sources"):
            click.echo(f"  Sources: {', '.join(result['sources'][:5])}")
    elif result["status"] == "insufficient_data":
        click.echo(f"⚠ {result.get('message')}")
    else:
        click.echo(f"✗ Error: {result.get('message', 'Unknown error')}", err=True)
        raise SystemExit(1)


def _print_analysis_result(result: dict) -> None:
    """Pretty-print analysis results."""
    click.echo("\n" + "=" * 60)
    click.echo(f"SENTIMENT ANALYSIS: {result['company_name']} ({result['ticker']})")
    click.echo("=" * 60)

    # Overall sentiment
    sentiment = result["overall_sentiment"].upper()
    score = result["sentiment_score"]
    confidence = result["confidence"]

    sentiment_color = {
        "POSITIVE": "green",
        "NEGATIVE": "red",
        "NEUTRAL": "yellow",
        "MIXED": "cyan",
    }.get(sentiment, "white")

    click.echo(f"\nOverall Sentiment: ", nl=False)
    click.secho(f"{sentiment}", fg=sentiment_color, bold=True, nl=False)
    click.echo(f" (score: {score:.2f}, confidence: {confidence:.0%})")

    # Trend
    trend = result.get("trend_direction", "stable")
    trend_color = {"improving": "green", "worsening": "red", "stable": "yellow"}.get(
        trend, "white"
    )
    click.echo(f"Trend: ", nl=False)
    click.secho(trend.upper(), fg=trend_color)

    # Article stats
    click.echo(
        f"\nArticles: {result['articles_analyzed']} analyzed "
        f"({result.get('material_article_count', 0)} material)"
    )
    click.echo(f"Period: {result.get('time_period', 'N/A')}")

    # Topics
    if result.get("top_topics"):
        click.echo(f"Top Topics: {', '.join(result['top_topics'])}")

    # Narrative summary
    if result.get("narrative_summary"):
        click.echo("\n" + "-" * 40)
        click.echo("NARRATIVE SUMMARY")
        click.echo("-" * 40)
        click.echo(result["narrative_summary"])

    # Material events
    if result.get("material_events"):
        click.echo("\n" + "-" * 40)
        click.echo("MATERIAL EVENTS")
        click.echo("-" * 40)
        for event in result["material_events"][:5]:
            click.echo(f"• {event}")

    # Key claims
    if result.get("key_claims"):
        click.echo("\n" + "-" * 40)
        click.echo("KEY CLAIMS")
        click.echo("-" * 40)
        for claim in result["key_claims"][:5]:
            source = claim.get("source", "Unknown")
            date = claim.get("date", "")
            click.echo(f'• "{claim["claim"]}" - {source} ({date})')

    click.echo("\n" + "=" * 60)


def _print_trend_result(result: dict) -> None:
    """Pretty-print trend results."""
    if result.get("status") == "no_data":
        click.echo(f"No trend data available for {result.get('ticker', 'ticker')}")
        return

    click.echo("\n" + "=" * 60)
    click.echo(f"SENTIMENT TRENDS: {result['ticker']}")
    click.echo("=" * 60)

    trend = result.get("trend_direction", "stable")
    trend_color = {"improving": "green", "worsening": "red", "stable": "yellow"}.get(
        trend, "white"
    )

    click.echo(f"\nTrend Direction: ", nl=False)
    click.secho(trend.upper(), fg=trend_color, bold=True)

    click.echo(f"Trend Magnitude: {result.get('trend_magnitude', 0):.3f}")
    click.echo(f"Current Score: {result.get('current_score', 0):.3f}")
    click.echo(f"Score at Start: {result.get('score_period_ago', 0):.3f}")

    # Daily scores
    daily_scores = result.get("daily_scores", [])
    if daily_scores:
        click.echo("\n" + "-" * 40)
        click.echo("DAILY SCORES (Recent)")
        click.echo("-" * 40)
        for day in daily_scores[-7:]:
            score = day.get("score", 0)
            articles = day.get("articles", 0)
            click.echo(f"  {day['date']}: {score:+.3f} ({articles} articles)")

    # Inflection points
    inflections = result.get("inflection_points", [])
    if inflections:
        click.echo("\n" + "-" * 40)
        click.echo("INFLECTION POINTS")
        click.echo("-" * 40)
        for point in inflections[:5]:
            click.echo(f"  {point['date']}: {point.get('description', '')}")

    click.echo("\n" + "=" * 60)


if __name__ == "__main__":
    cli()
