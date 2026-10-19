"""Split text into overlapping, character-based chunks for embedding."""
from __future__ import annotations

import re

# A sentence end: . ! or ? (optionally followed by a closing quote or bracket), then whitespace.
_SENTENCE_END = re.compile(r"[.!?][\"')\]]?\s")

# Table blocks written by parsing.py: "# Sheet: name", a header row, data rows; blocks end at a blank line.
SHEET_PREFIX = "# Sheet: "


def _find_break(text: str, start: int, end: int, size: int) -> int:
    """Pick a cut point in the last ~20% of the window: paragraph, sentence, line, then whitespace."""
    lo = start + int(size * 0.8)
    i = text.rfind("\n\n", lo, end)
    if i != -1:
        return i
    matches = list(_SENTENCE_END.finditer(text, lo, end))
    if matches:
        return matches[-1].end()
    # After sentences (PDF text hard-wraps lines), so list items stay whole.
    i = text.rfind("\n", lo, end)
    if i != -1:
        return i
    i = max(text.rfind(c, lo, end) for c in " \t")
    return i if i != -1 else end


def _chunk_prose(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            end = _find_break(text, start, end, chunk_size)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        # Always move forward, even if the break landed right at the start.
        start = max(end - overlap, start + 1)
    return chunks


def _chunk_table(block: str, chunk_size: int, overlap: int) -> list[str]:
    """Pack whole rows into chunks, each starting with the sheet line and header row.

    Rows are never split or repeated, so no overlap is needed. A row too long for one chunk
    is split as prose, still prefixed with the header.
    """
    lines = block.strip("\n").split("\n")
    head = "\n".join(lines[:2])
    rows = lines[2:]
    if not rows:
        return [head]
    budget = max(chunk_size - len(head) - 1, chunk_size // 4)  # never let a huge header starve rows
    chunks, current, used = [], [], 0
    for row in rows:
        if len(row) > budget:
            if current:
                chunks.append(head + "\n" + "\n".join(current))
                current, used = [], 0
            step = max(budget, overlap + 1)
            chunks.extend(head + "\n" + p for p in _chunk_prose(row, step, min(overlap, step - 1)))
            continue
        if current and used + len(row) + 1 > budget:
            chunks.append(head + "\n" + "\n".join(current))
            current, used = [], 0
        current.append(row)
        used += len(row) + 1
    if current:
        chunks.append(head + "\n" + "\n".join(current))
    return chunks


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Split text into chunks of about chunk_size chars that overlap by about overlap chars.

    Prose chunks are at most chunk_size chars. Table blocks (from spreadsheets) are packed by whole
    rows with the sheet header repeated in every chunk, so a chunk can exceed chunk_size by at
    most the header length when the header itself is very long.
    """
    if chunk_size <= 0 or overlap < 0:
        raise ValueError("chunk_size must be positive and overlap must not be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if SHEET_PREFIX not in text:
        return _chunk_prose(text, chunk_size, overlap)

    chunks: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        if block.lstrip().startswith(SHEET_PREFIX):
            chunks.extend(_chunk_table(block.strip(), chunk_size, overlap))
        elif block.strip():
            chunks.extend(_chunk_prose(block, chunk_size, overlap))
    return chunks
