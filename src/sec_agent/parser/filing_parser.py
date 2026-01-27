"""Parse SEC filing HTML/SGML into structured sections."""

from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, Tag


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


def _extract_text(soup: BeautifulSoup) -> str:
    """Get clean text from parsed HTML, preserving paragraph breaks."""
    for tag in soup.find_all(["script", "style"]):
        tag.decompose()
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


def _find_item_positions(text: str) -> list[tuple[int, str]]:
    """Find all Item header positions and their item numbers in the text.

    Returns a list of (position, item_number) tuples sorted by position.
    Filters out likely table-of-contents entries by requiring substantial
    text between consecutive items.
    """
    matches: list[tuple[int, str]] = []
    for m in _ITEM_HEADER_RE.finditer(text):
        item_num = m.group(1).upper()
        matches.append((m.start(), item_num))

    if not matches:
        return []

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
            # Pick the last occurrence — usually the actual content section
            # (ToC comes first in filings).
            result.append(occurrences[-1])

    result.sort(key=lambda x: x[0])
    return result


def parse_filing(html: str, filing_type: str) -> list[Section]:
    """Parse a SEC filing HTML string into a list of Section objects.

    Args:
        html: Raw HTML content of the filing.
        filing_type: One of "10-K", "10-Q", "8-K".

    Returns:
        List of Section objects extracted from the filing.
    """
    soup = BeautifulSoup(html, "lxml")
    full_text = _extract_text(soup)

    if not full_text:
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
    soup = BeautifulSoup(html, "lxml")
    full_text = _extract_text(soup)

    if not full_text:
        return []

    # 8-K items use decimal numbering: Item 1.01, Item 2.02, etc.
    pattern = re.compile(
        r"(?:Item|ITEM)\s+(\d+\.\d+)[\.\:\s\-\u2013\u2014]",
        re.IGNORECASE,
    )

    matches: list[tuple[int, str]] = []
    for m in pattern.finditer(full_text):
        matches.append((m.start(), m.group(1)))

    # Deduplicate: keep last occurrence of each item
    by_item: dict[str, tuple[int, str]] = {}
    for pos, item_num in matches:
        by_item[item_num] = (pos, item_num)

    sorted_items = sorted(by_item.values(), key=lambda x: x[0])

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
