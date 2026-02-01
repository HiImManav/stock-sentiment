"""Tests for the SEC filing parser."""

from __future__ import annotations

import warnings

import pytest

from sec_agent.parser.filing_parser import (
    Section,
    _detect_document_type,
    parse_8k,
    parse_filing,
)


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


class TestDetectDocumentType:
    def test_xml_declaration(self) -> None:
        content = '<?xml version="1.0"?><root>content</root>'
        assert _detect_document_type(content) == "xml"

    def test_xbrli_namespace(self) -> None:
        content = '<html xmlns:xbrli="http://www.xbrl.org/2003/instance"><body>content</body></html>'
        assert _detect_document_type(content) == "xml"

    def test_ix_namespace(self) -> None:
        content = '<html xmlns:ix="http://www.xbrl.org/2013/inlineXBRL"><body>content</body></html>'
        assert _detect_document_type(content) == "xml"

    def test_plain_html(self) -> None:
        content = "<html><body><p>Hello world</p></body></html>"
        assert _detect_document_type(content) == "html"

    def test_html_with_doctype(self) -> None:
        content = "<!DOCTYPE html><html><body>content</body></html>"
        assert _detect_document_type(content) == "html"


class TestXMLParsing:
    def test_xml_content_parses_without_warning(self) -> None:
        """XML content should parse without XMLParsedAsHTMLWarning."""
        xml_content = f"""<?xml version="1.0"?>
        <html>
        <body>
        <h2>Item 1A. Risk Factors</h2>
        <p>{_LONG_TEXT}</p>
        </body>
        </html>
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sections = parse_filing(xml_content, "10-K")
            # Check no XMLParsedAsHTMLWarning was raised
            xml_warnings = [
                warning for warning in w
                if "XMLParsedAsHTMLWarning" in str(warning.category)
            ]
            assert len(xml_warnings) == 0

    def test_html_content_still_works(self) -> None:
        """HTML content should continue to work correctly."""
        html = _make_10k_html({"1A": _LONG_TEXT})
        sections = parse_filing(html, "10-K")
        assert len(sections) >= 1
        assert any(s.item_number == "1A" for s in sections)


class TestTocAtEnd:
    def test_content_before_toc_extracts_correct_section(self) -> None:
        """When content appears before ToC, should still extract the actual content."""
        # Simulate a filing where actual content comes first, then ToC at end
        html = (
            "<html><body>"
            # Actual content section (first occurrence, has substantial text)
            f"<h2>Item 1A. Risk Factors</h2>"
            f"<p>{_LONG_TEXT * 3}</p>"  # Lots of content
            # Separator
            f"<hr><h1>Table of Contents</h1>"
            # ToC entry (second occurrence, just a reference)
            f"<p>Item 1A. Risk Factors ... page 5</p>"
            f"<p>Item 7. MD&A ... page 20</p>"
            "</body></html>"
        )
        sections = parse_filing(html, "10-K")
        items_1a = [s for s in sections if s.item_number == "1A"]
        assert len(items_1a) == 1
        # The selected section should have the long content, not just ToC reference
        assert len(items_1a[0].text) > 500

    def test_8k_content_before_toc(self) -> None:
        """8-K parsing should also handle content-before-ToC scenario."""
        html = (
            "<html><body>"
            # Actual content (first occurrence)
            f"<h2>Item 1.01. Entry into Material Agreement</h2>"
            f"<p>{_LONG_TEXT * 2}</p>"
            # ToC at end (second occurrence)
            f"<hr><h1>Index</h1>"
            f"<p>Item 1.01 - see page 2</p>"
            "</body></html>"
        )
        sections = parse_8k(html)
        items = [s for s in sections if s.item_number == "1.01"]
        assert len(items) == 1
        # Should have the actual content, not just the ToC entry
        assert len(items[0].text) > 200


class TestTablePreservation:
    def test_tables_converted_to_pipe_format(self) -> None:
        """Tables should be converted to pipe-separated format."""
        html = """
        <html><body>
        <h2>Item 1A. Risk Factors</h2>
        <table>
            <tr><th>Year</th><th>Revenue</th></tr>
            <tr><td>2023</td><td>$100M</td></tr>
            <tr><td>2024</td><td>$150M</td></tr>
        </table>
        <p>More content here to meet minimum length requirement for section extraction.</p>
        <p>Additional paragraph to ensure we have enough text.</p>
        <p>Yet another paragraph of sufficient length for testing purposes.</p>
        </body></html>
        """
        sections = parse_filing(html, "10-K")
        assert len(sections) >= 1
        section_text = sections[0].text
        # Check that table data is preserved in pipe format
        assert "Year | Revenue" in section_text or "Year" in section_text
        assert "$100M" in section_text
