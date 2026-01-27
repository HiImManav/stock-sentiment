"""Section-aware chunking engine for SEC filings."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import tiktoken

from sec_agent.parser.filing_parser import Section

# Sentence boundary regex: split after period/question/exclamation followed by space and capital.
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

# Encoding used for token counting.
_ENCODING = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    """A retrieval-friendly chunk of a SEC filing section."""

    text: str
    metadata: dict[str, str | int] = field(default_factory=dict)
    token_count: int = 0


def _count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def _split_paragraphs(text: str) -> list[str]:
    """Split text on double-newline paragraph boundaries."""
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    parts = _SENTENCE_BOUNDARY_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _is_table_block(text: str) -> bool:
    """Heuristic: a paragraph that contains multiple pipe chars or tab-aligned columns."""
    pipe_count = text.count("|")
    tab_count = text.count("\t")
    # If it has several pipes or tabs per line, likely a table
    lines = text.strip().splitlines()
    if len(lines) >= 2 and (pipe_count >= len(lines) or tab_count >= len(lines)):
        return True
    return False


def chunk_section(
    section: Section,
    *,
    max_tokens: int = 1500,
    overlap: int = 200,
    filing_type: str = "",
    ticker: str = "",
    accession_number: str = "",
) -> list[Chunk]:
    """Chunk a single section into retrieval-friendly pieces.

    Splits at paragraph boundaries, falls back to sentence boundaries,
    and never splits table rows across chunks.
    """
    paragraphs = _split_paragraphs(section.text)
    if not paragraphs:
        return []

    chunks: list[Chunk] = []
    current_parts: list[str] = []
    current_tokens = 0

    def _flush(parts: list[str]) -> None:
        if not parts:
            return
        text = "\n\n".join(parts)
        tokens = _count_tokens(text)
        chunks.append(
            Chunk(
                text=text,
                metadata={
                    "section_name": section.name,
                    "item_number": section.item_number,
                    "filing_type": filing_type,
                    "ticker": ticker,
                    "accession_number": accession_number,
                    "chunk_index": len(chunks),
                },
                token_count=tokens,
            )
        )

    for para in paragraphs:
        para_tokens = _count_tokens(para)

        # If a single paragraph exceeds max_tokens and is NOT a table, split by sentence.
        if para_tokens > max_tokens and not _is_table_block(para):
            # Flush what we have so far
            _flush(current_parts)
            current_parts = []
            current_tokens = 0

            sentences = _split_sentences(para)
            sent_parts: list[str] = []
            sent_tokens = 0
            for sent in sentences:
                st = _count_tokens(sent)
                if sent_tokens + st > max_tokens and sent_parts:
                    _flush(sent_parts)
                    # Overlap: keep last portion
                    overlap_parts: list[str] = []
                    overlap_tokens = 0
                    for s in reversed(sent_parts):
                        t = _count_tokens(s)
                        if overlap_tokens + t > overlap:
                            break
                        overlap_parts.insert(0, s)
                        overlap_tokens += t
                    sent_parts = overlap_parts
                    sent_tokens = overlap_tokens
                sent_parts.append(sent)
                sent_tokens += st
            if sent_parts:
                _flush(sent_parts)
            continue

        # Table blocks: always keep as a single chunk (don't split rows)
        if _is_table_block(para) and para_tokens <= max_tokens:
            if current_tokens + para_tokens > max_tokens:
                _flush(current_parts)
                current_parts = []
                current_tokens = 0
            current_parts.append(para)
            current_tokens += para_tokens
            continue

        # Normal paragraph accumulation
        if current_tokens + para_tokens > max_tokens and current_parts:
            _flush(current_parts)
            # Overlap: carry over trailing paragraphs
            overlap_parts: list[str] = []
            overlap_tokens = 0
            for p in reversed(current_parts):
                t = _count_tokens(p)
                if overlap_tokens + t > overlap:
                    break
                overlap_parts.insert(0, p)
                overlap_tokens += t
            current_parts = overlap_parts
            current_tokens = overlap_tokens

        current_parts.append(para)
        current_tokens += para_tokens

    _flush(current_parts)
    return chunks


def chunk_filing(
    sections: list[Section],
    *,
    max_tokens: int = 1500,
    overlap: int = 200,
    filing_type: str = "",
    ticker: str = "",
    accession_number: str = "",
) -> list[Chunk]:
    """Chunk all sections of a filing.

    Returns a flat list of Chunk objects with correct chunk_index values
    scoped per-section.
    """
    all_chunks: list[Chunk] = []
    for section in sections:
        section_chunks = chunk_section(
            section,
            max_tokens=max_tokens,
            overlap=overlap,
            filing_type=filing_type,
            ticker=ticker,
            accession_number=accession_number,
        )
        all_chunks.extend(section_chunks)
    return all_chunks
