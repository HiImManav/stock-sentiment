"""Tests for the SEC filing parser."""

from __future__ import annotations

import pytest

from sec_agent.parser.filing_parser import Section, parse_8k, parse_filing


# -- Helpers -----------------------------------------------------------------

def _make_10k_html(items: dict[str, str]) -> str:
    """Build a minimal 10-K HTML with given item sections."""
    parts = ["<html><body>", "<h1>Table of Contents</h1>"]
    # Add a ToC with links first (should be skipped by parser)
    for item_num, content in items.items():
        parts.append(f'<a href="#item{item_num}">Item {item_num}</a><br>')
    parts.append("<hr>")
    # Actual sections
    for item_num, content in items.items():
        parts.append(f'<h2 id="item{item_num}">Item {item_num}.</h2>')
        parts.append(f"<p>{content}</p>")
    parts.append("</body></html>")
    return "\n".join(parts)


def _make_8k_html(items: dict[str, str]) -> str:
    """Build a minimal 8-K HTML with given item sections."""
    parts = ["<html><body>"]
    for item_num, content in items.items():
        parts.append(f"<h2>Item {item_num}.</h2>")
        parts.append(f"<p>{content}</p>")
    parts.append("</body></html>")
    return "\n".join(parts)


# Enough text to pass the 100-char minimum
_LONG_TEXT = "This is a sufficiently long paragraph. " * 10


# -- Tests: parse_filing for 10-K -------------------------------------------


class TestParseFiling10K:
    def test_extracts_sections(self) -> None:
        html = _make_10k_html({"1A": _LONG_TEXT, "7": _LONG_TEXT})
        sections = parse_filing(html, "10-K")

        assert len(sections) == 2
        assert sections[0].item_number == "1A"
        assert sections[0].name == "Risk Factors"
        assert sections[1].item_number == "7"
        assert sections[1].name == "Management's Discussion and Analysis"

    def test_skips_short_sections(self) -> None:
        html = _make_10k_html({"1A": "short", "7": _LONG_TEXT})
        sections = parse_filing(html, "10-K")

        # Item 1A is too short, only Item 7 should remain
        item_nums = [s.item_number for s in sections]
        assert "7" in item_nums

    def test_unknown_item_gets_default_name(self) -> None:
        html = _make_10k_html({"99": _LONG_TEXT})
        sections = parse_filing(html, "10-K")

        assert len(sections) == 1
        assert sections[0].name == "Item 99"

    def test_empty_html(self) -> None:
        assert parse_filing("", "10-K") == []

    def test_no_items_found(self) -> None:
        html = "<html><body><p>No items here at all.</p></body></html>"
        assert parse_filing(html, "10-K") == []

    def test_deduplicates_toc_entries(self) -> None:
        """When an item appears in ToC and content, keep only the content occurrence."""
        html = (
            "<html><body>"
            f"<p>Item 1A. Risk Factors</p>"  # ToC-like (first occurrence)
            f"<p>{_LONG_TEXT}</p>"
            f"<h2>Item 1A. Risk Factors</h2>"  # Actual content (second occurrence)
            f"<p>{_LONG_TEXT}</p>"
            "</body></html>"
        )
        sections = parse_filing(html, "10-K")
        # Should have one section for 1A (the last occurrence)
        items_1a = [s for s in sections if s.item_number == "1A"]
        assert len(items_1a) == 1


class TestParseFiling10Q:
    def test_extracts_10q_sections(self) -> None:
        html = _make_10k_html({"1": _LONG_TEXT, "2": _LONG_TEXT})
        sections = parse_filing(html, "10-Q")

        assert len(sections) == 2
        assert sections[0].item_number == "1"
        assert sections[0].name == "Financial Statements"
        assert sections[1].item_number == "2"
        assert sections[1].name == "Management's Discussion and Analysis"


class TestParse8K:
    def test_extracts_8k_items(self) -> None:
        html = _make_8k_html({"1.01": _LONG_TEXT, "2.02": _LONG_TEXT})
        sections = parse_8k(html)

        assert len(sections) == 2
        assert sections[0].item_number == "1.01"
        assert sections[0].name == "Item 1.01"
        assert sections[1].item_number == "2.02"

    def test_empty_html(self) -> None:
        assert parse_8k("") == []


class TestSectionDataclass:
    def test_fields(self) -> None:
        s = Section(name="Risk Factors", item_number="1A", text="hello", start_pos=0, end_pos=5)
        assert s.name == "Risk Factors"
        assert s.item_number == "1A"
        assert s.text == "hello"
        assert s.start_pos == 0
        assert s.end_pos == 5
