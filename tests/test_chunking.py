import pytest

from app.chunking import chunk_text


def test_empty_and_whitespace_text_gives_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_short_text_is_one_chunk():
    assert chunk_text("hello world", chunk_size=512, overlap=64) == ["hello world"]


def test_chunks_respect_size_bound():
    text = "lorem ipsum dolor sit amet " * 200
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_consecutive_chunks_overlap():
    text = "".join(f"{i:03d} " for i in range(300))
    chunks = chunk_text(text, chunk_size=80, overlap=20)
    assert len(chunks) > 2
    for prev, nxt in zip(chunks, chunks[1:]):
        # The start of the next chunk repeats text from the end of the previous one.
        assert nxt[:5] in prev


def test_prefers_paragraph_break():
    text = "a" * 17 + "\n\n" + "b" * 30
    chunks = chunk_text(text, chunk_size=20, overlap=0)
    assert chunks[0] == "a" * 17


def test_prefers_sentence_end_over_plain_space():
    text = "A" * 12 + ". " + "B" * 30
    chunks = chunk_text(text, chunk_size=16, overlap=0)
    assert chunks[0] == "A" * 12 + "."


def test_prefers_whitespace_over_mid_word_cut():
    text = "word " * 40
    chunks = chunk_text(text, chunk_size=20, overlap=0)
    assert all(set(c.split()) == {"word"} for c in chunks)


def test_long_text_without_whitespace_still_progresses():
    text = "x" * 1000
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert all(len(c) <= 100 for c in chunks)
    assert chunks[0] == "x" * 100
    assert chunks[-1].endswith("x")
    # Each chunk adds 90 new chars after the first 100, so 11 chunks cover 1000 chars.
    assert len(chunks) == 11


def test_overlap_equal_to_size_raises():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=64, overlap=64)


def test_overlap_larger_than_size_raises():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=64, overlap=100)


def _sheet(name, rows):
    return f"# Sheet: {name}\nregion | quarter | units\n" + "\n".join(rows)


def test_table_chunks_repeat_header_and_keep_rows_whole():
    rows = [f"Region{i} | Q{i % 4 + 1} 2025 | {i * 10}" for i in range(60)]
    text = _sheet("Sales", rows) + "\n\n" + _sheet("Prices", ["R7 | - | 54000"])
    chunks = chunk_text(text, 200, 40)
    sales = [c for c in chunks if c.startswith("# Sheet: Sales")]
    assert len(sales) > 1
    for c in sales:
        lines = c.split("\n")
        assert lines[:2] == ["# Sheet: Sales", "region | quarter | units"]
        assert all(line in rows for line in lines[2:])  # whole rows only
        assert len(c) <= 200
    # every row appears exactly once across the Sales chunks
    seen = [line for c in sales for line in c.split("\n")[2:]]
    assert seen == rows
    assert chunks[-1] == "# Sheet: Prices\nregion | quarter | units\nR7 | - | 54000"


def test_table_row_longer_than_chunk_is_split_with_header():
    long_row = "x | " + "word " * 100 + "| 1"
    chunks = chunk_text(_sheet("S", [long_row, "a | b | 2"]), 120, 20)
    assert all(c.startswith("# Sheet: S\nregion | quarter | units\n") for c in chunks)
    assert chunks[-1].endswith("a | b | 2")


def test_prose_unchanged_without_sheet_marker():
    text = "Para one. " * 30 + "\n\n" + "Para two. " * 30
    assert all(not c.startswith("# Sheet") for c in chunk_text(text, 100, 10))
