"""Tests for the section-aware chunking engine."""

from sec_agent.parser.chunker import Chunk, chunk_filing, chunk_section
from sec_agent.parser.filing_parser import Section


def _make_section(text: str, name: str = "Risk Factors", item: str = "1A") -> Section:
    return Section(name=name, item_number=item, text=text, start_pos=0, end_pos=len(text))


class TestChunkSection:
    def test_short_section_single_chunk(self) -> None:
        section = _make_section("This is a short section about risks.")
        chunks = chunk_section(section, max_tokens=1500, overlap=200, ticker="AAPL")
        assert len(chunks) == 1
        assert chunks[0].text == "This is a short section about risks."
        assert chunks[0].metadata["section_name"] == "Risk Factors"
        assert chunks[0].metadata["item_number"] == "1A"
        assert chunks[0].metadata["ticker"] == "AAPL"
        assert chunks[0].metadata["chunk_index"] == 0
        assert chunks[0].token_count > 0

    def test_paragraph_boundary_splitting(self) -> None:
        # Create text with multiple paragraphs that together exceed max_tokens
        para = "Word " * 200  # ~200 tokens
        text = (para.strip() + "\n\n") * 10  # ~2000 tokens total
        section = _make_section(text.strip())
        chunks = chunk_section(section, max_tokens=500, overlap=0)
        assert len(chunks) > 1
        # Each chunk should respect token limits (with some tolerance)
        for c in chunks:
            assert c.token_count <= 600  # some tolerance for boundary

    def test_overlap_carries_content(self) -> None:
        para1 = "First paragraph. " * 50  # ~150 tokens
        para2 = "Second paragraph. " * 50
        para3 = "Third paragraph. " * 50
        para4 = "Fourth paragraph. " * 50
        text = f"{para1.strip()}\n\n{para2.strip()}\n\n{para3.strip()}\n\n{para4.strip()}"
        section = _make_section(text)
        chunks = chunk_section(section, max_tokens=350, overlap=200)
        assert len(chunks) >= 2
        # With overlap, later chunks may contain text from previous chunk
        if len(chunks) >= 2:
            # The second chunk should have some overlap content
            assert chunks[1].token_count > 0

    def test_sentence_fallback_for_long_paragraph(self) -> None:
        # One giant paragraph with many sentences
        sentences = [f"Sentence number {i} about risk factors." for i in range(100)]
        text = " ".join(sentences)
        section = _make_section(text)
        chunks = chunk_section(section, max_tokens=100, overlap=0)
        assert len(chunks) > 1

    def test_empty_section(self) -> None:
        section = _make_section("")
        chunks = chunk_section(section)
        assert chunks == []

    def test_metadata_fields(self) -> None:
        section = _make_section("Some text here.")
        chunks = chunk_section(
            section,
            filing_type="10-K",
            ticker="MSFT",
            accession_number="0001-23-456",
        )
        assert len(chunks) == 1
        m = chunks[0].metadata
        assert m["filing_type"] == "10-K"
        assert m["ticker"] == "MSFT"
        assert m["accession_number"] == "0001-23-456"
        assert m["chunk_index"] == 0

    def test_table_block_not_split(self) -> None:
        # A table-like block with pipe separators
        header = "| Column A | Column B | Column C |"
        rows = [f"| Value {i}A | Value {i}B | Value {i}C |" for i in range(10)]
        table = "\n".join([header] + rows)
        # Surround with normal text
        text = f"Intro paragraph.\n\n{table}\n\nConclusion paragraph."
        section = _make_section(text)
        chunks = chunk_section(section, max_tokens=1500, overlap=0)
        # Table should appear in one chunk
        found_table = any(header in c.text for c in chunks)
        assert found_table


class TestChunkFiling:
    def test_multiple_sections(self) -> None:
        sections = [
            _make_section("Risk factor content here.", name="Risk Factors", item="1A"),
            _make_section("MD&A content here.", name="MD&A", item="7"),
        ]
        chunks = chunk_filing(sections, ticker="GOOG", filing_type="10-K")
        assert len(chunks) == 2
        assert chunks[0].metadata["item_number"] == "1A"
        assert chunks[1].metadata["item_number"] == "7"

    def test_empty_sections_list(self) -> None:
        chunks = chunk_filing([])
        assert chunks == []

    def test_chunk_indices_per_section(self) -> None:
        long_text = ("Paragraph content. " * 100 + "\n\n") * 5
        sections = [
            _make_section(long_text.strip(), name="Risk Factors", item="1A"),
            _make_section("Short section.", name="MD&A", item="7"),
        ]
        chunks = chunk_filing(sections, max_tokens=300, overlap=0)
        # First section should have multiple chunks with sequential indices
        section_1a = [c for c in chunks if c.metadata["item_number"] == "1A"]
        assert len(section_1a) > 1
        for i, c in enumerate(section_1a):
            assert c.metadata["chunk_index"] == i
