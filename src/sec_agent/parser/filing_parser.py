"""Parse SEC filing HTML/SGML into structured sections."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)


@dataclass
class Section:
    """A single section extracted from a SEC filing."""

    name: str
    item_number: str
    text: str
    start_pos: int
    end_pos: int


# Canonical section names by filing type and item number.
_10K_SECTIONS: dict[str, str] = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity",
    "6": "Reserved",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements With Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
}

_10Q_SECTIONS: dict[str, str] = {
    "1": "Financial Statements",
    "1A": "Risk Factors",
    "2": "Management's Discussion and Analysis",
    "3": "Quantitative and Qualitative Disclosures About Market Risk",
    "4": "Controls and Procedures",
}

_SECTION_MAPS: dict[str, dict[str, str]] = {
    "10-K": _10K_SECTIONS,
    "10-Q": _10Q_SECTIONS,
}

# Regex to match item headers like "Item 1A." or "ITEM 7 -" or "Item 1A:"
_ITEM_HEADER_RE = re.compile(
    r"(?:Item|ITEM)\s+(\d+[A-Za-z]?)[\.\:\s\-\u2013\u2014]",
    re.IGNORECASE,
)

# Alternate header patterns for known section names
_SECTION_NAME_PATTERNS: list[tuple[str, str]] = [
    ("RISK FACTORS", "1A"),
    ("MANAGEMENT'S DISCUSSION AND ANALYSIS", "7"),
    ("MD&A", "7"),
    ("FINANCIAL STATEMENTS", "8"),
    ("BUSINESS", "1"),
]


def _detect_document_type(content: str) -> str:
    """Detect if content is XML or HTML.

    Args:
        content: Raw document content.

    Returns:
        'xml' if the content appears to be XML/XBRL, 'html' otherwise.
    """
    content_start = content.strip()[:500].lower()
    if content_start.startswith("<?xml"):
        return "xml"
    if "xmlns:xbrli" in content_start or "xmlns:ix" in content_start:
        return "xml"
    return "html"


def _extract_text(soup: BeautifulSoup) -> str:
    """Get clean text from parsed HTML, preserving paragraph breaks and tables.

    Tables are converted to pipe-separated format to preserve structure.
    """
    for tag in soup.find_all(["script", "style"]):
        tag.decompose()

    # Convert tables to pipe-separated text format
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [
                td.get_text(strip=True) for td in tr.find_all(["td", "th"])
            ]
            if cells:
                rows.append(" | ".join(cells))
        table.replace_with("\n".join(rows) + "\n")

    text = soup.get_text(separator="\n")
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_toc_link(element: Tag) -> bool:
    """Check if an element is likely a table-of-contents link."""
    parent = element.parent
    while parent:
        if parent.name == "a" and parent.get("href", "").startswith("#"):
            return True
        parent = parent.parent
    return False


def _content_length_after(
    pos: int, text: str, all_positions: list[tuple[int, str]]
) -> int:
    """Calculate content length from a position to the next item header.

    Args:
        pos: Starting position in text.
        text: Full document text.
        all_positions: All (position, item_number) tuples, sorted by position.

    Returns:
        Number of characters between this position and the next item header.
    """
    # Find the next position after this one
    next_pos = len(text)
    for p, _ in all_positions:
        if p > pos:
            next_pos = p
            break
    return next_pos - pos


def _find_item_positions(text: str) -> list[tuple[int, str]]:
    """Find all Item header positions and their item numbers in the text.

    Returns a list of (position, item_number) tuples sorted by position.
    For duplicate item numbers, keeps the occurrence with the most content
    following it (the actual section, not ToC references).
    """
    matches: list[tuple[int, str]] = []
    for m in _ITEM_HEADER_RE.finditer(text):
        item_num = m.group(1).upper()
        matches.append((m.start(), item_num))

    if not matches:
        return []

    # Sort by position for content length calculation
    matches.sort(key=lambda x: x[0])

    # Deduplicate: for each item number, if it appears multiple times,
    # keep the occurrence with the most text following it (likely the actual
    # section, not the ToC reference).
    by_item: dict[str, list[tuple[int, str]]] = {}
    for pos, item_num in matches:
        by_item.setdefault(item_num, []).append((pos, item_num))

    result: list[tuple[int, str]] = []
    for item_num, occurrences in by_item.items():
        if len(occurrences) == 1:
            result.append(occurrences[0])
        else:
            # Pick the occurrence with the most content following it
            best = max(
                occurrences,
                key=lambda x: _content_length_after(x[0], text, matches),
            )
            logger.debug(
                "Item %s has %d occurrences, selected position %d with most content",
                item_num,
                len(occurrences),
                best[0],
            )
            result.append(best)

    result.sort(key=lambda x: x[0])
    logger.debug("Found %d unique item positions", len(result))
    return result


def parse_filing(html: str, filing_type: str) -> list[Section]:
    """Parse a SEC filing HTML string into a list of Section objects.

    Args:
        html: Raw HTML content of the filing.
        filing_type: One of "10-K", "10-Q", "8-K".

    Returns:
        List of Section objects extracted from the filing.
    """
    doc_type = _detect_document_type(html)
    parser = "lxml-xml" if doc_type == "xml" else "lxml"
    logger.debug("Detected document type: %s, using parser: %s", doc_type, parser)

    soup = BeautifulSoup(html, parser)
    full_text = _extract_text(soup)

    if not full_text:
        logger.warning("Empty text extracted from %s filing", filing_type)
        return []

    section_map = _SECTION_MAPS.get(filing_type.upper(), {})
    item_positions = _find_item_positions(full_text)

    if not item_positions:
        # Fallback: try matching known section name patterns
        return _parse_by_section_names(full_text, filing_type)

    sections: list[Section] = []
    for i, (start_pos, item_num) in enumerate(item_positions):
        # End position is the start of the next section or end of text
        end_pos = item_positions[i + 1][0] if i + 1 < len(item_positions) else len(full_text)
        section_text = full_text[start_pos:end_pos].strip()

        # Skip very short sections (likely just a header with no content)
        if len(section_text) < 100:
            continue

        name = section_map.get(item_num, f"Item {item_num}")

        sections.append(
            Section(
                name=name,
                item_number=item_num,
                text=section_text,
                start_pos=start_pos,
                end_pos=end_pos,
            )
        )

    return sections


def _parse_by_section_names(full_text: str, filing_type: str) -> list[Section]:
    """Fallback parser that matches section names instead of item numbers."""
    section_map = _SECTION_MAPS.get(filing_type.upper(), {})
    found: list[tuple[int, str, str]] = []  # (pos, item_num, name)

    for name_pattern, default_item in _SECTION_NAME_PATTERNS:
        pattern = re.compile(re.escape(name_pattern), re.IGNORECASE)
        match = pattern.search(full_text)
        if match:
            name = section_map.get(default_item, name_pattern.title())
            found.append((match.start(), default_item, name))

    found.sort(key=lambda x: x[0])

    sections: list[Section] = []
    for i, (start_pos, item_num, name) in enumerate(found):
        end_pos = found[i + 1][0] if i + 1 < len(found) else len(full_text)
        section_text = full_text[start_pos:end_pos].strip()

        if len(section_text) < 100:
            continue

        sections.append(
            Section(
                name=name,
                item_number=item_num,
                text=section_text,
                start_pos=start_pos,
                end_pos=end_pos,
            )
        )

    return sections


def parse_8k(html: str) -> list[Section]:
    """Parse an 8-K filing which has variable item numbers (1.01-9.01).

    8-K items use a different numbering scheme (e.g., Item 1.01, Item 2.02).
    """
    doc_type = _detect_document_type(html)
    parser = "lxml-xml" if doc_type == "xml" else "lxml"
    logger.debug("Detected document type: %s, using parser: %s", doc_type, parser)

    soup = BeautifulSoup(html, parser)
    full_text = _extract_text(soup)

    if not full_text:
        logger.warning("Empty text extracted from 8-K filing")
        return []

    # 8-K items use decimal numbering: Item 1.01, Item 2.02, etc.
    pattern = re.compile(
        r"(?:Item|ITEM)\s+(\d+\.\d+)[\.\:\s\-\u2013\u2014]",
        re.IGNORECASE,
    )

    matches: list[tuple[int, str]] = []
    for m in pattern.finditer(full_text):
        matches.append((m.start(), m.group(1)))

    if not matches:
        logger.debug("No 8-K item headers found")
        return []

    # Sort by position for content length calculation
    matches.sort(key=lambda x: x[0])

    # Deduplicate: for each item, keep the occurrence with most content after it
    by_item: dict[str, list[tuple[int, str]]] = {}
    for pos, item_num in matches:
        by_item.setdefault(item_num, []).append((pos, item_num))

    deduped: list[tuple[int, str]] = []
    for item_num, occurrences in by_item.items():
        if len(occurrences) == 1:
            deduped.append(occurrences[0])
        else:
            best = max(
                occurrences,
                key=lambda x: _content_length_after(x[0], full_text, matches),
            )
            logger.debug(
                "8-K Item %s has %d occurrences, selected position %d",
                item_num,
                len(occurrences),
                best[0],
            )
            deduped.append(best)

    sorted_items = sorted(deduped, key=lambda x: x[0])
    logger.debug("Found %d unique 8-K items", len(sorted_items))

    sections: list[Section] = []
    for i, (start_pos, item_num) in enumerate(sorted_items):
        end_pos = sorted_items[i + 1][0] if i + 1 < len(sorted_items) else len(full_text)
        section_text = full_text[start_pos:end_pos].strip()

        if len(section_text) < 50:
            continue

        sections.append(
            Section(
                name=f"Item {item_num}",
                item_number=item_num,
                text=section_text,
                start_pos=start_pos,
                end_pos=end_pos,
            )
        )

    return sections
