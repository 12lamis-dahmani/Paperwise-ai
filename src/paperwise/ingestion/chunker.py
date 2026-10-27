
import re

from paperwise.models import Chunk


def _units(text: str) -> list[str]:
    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]
    units: list[str] = []

    for paragraph in paragraphs:
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        units.extend(
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        )

    return units


def _character_chunks(text: str, size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(chunk_text)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def _semantic_chunks(text: str, size: int, overlap: int) -> list[str]:
    units = _units(text)

    if not units:
        return []

    if len(units) == 1:
        return _character_chunks(text, size, overlap)

    # Split very long sentences using the existing fallback.
    normalized_units = []
    for unit in units:
        if len(unit) > size:
            normalized_units.extend(
                _character_chunks(unit, size, overlap)
            )
        else:
            normalized_units.append(unit)

    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    for unit in normalized_units:
        added_length = len(unit) if not current else len(unit) + 1

        if current and current_length + added_length > size:
            chunks.append(" ".join(current).strip())

            carry: list[str] = []
            carry_length = 0

            for previous in reversed(current):
                length = len(previous) if not carry else len(previous) + 1

                if carry_length + length > overlap:
                    break

                carry.insert(0, previous)
                carry_length += length

            # Do not keep overlap if it makes the next chunk too large.
            if carry_length + len(unit) + (1 if carry else 0) > size:
                carry = []
                carry_length = 0

            current = carry
            current_length = carry_length

        current.append(unit)
        current_length += len(unit) if len(current) == 1 else len(unit) + 1

    if current:
        final_chunk = " ".join(current).strip()
        if not chunks or final_chunk != chunks[-1]:
            chunks.append(final_chunk)

    return chunks


def chunk_pages(
    pages: list[tuple[int, str]],
    size: int = 1200,
    overlap: int = 200,
) -> list[Chunk]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("overlap must be non-negative and smaller than size")

    chunks: list[Chunk] = []
    index = 0

    for page, text in pages:
        for chunk_text in _semantic_chunks(text.strip(), size, overlap):
            if chunk_text:
                chunks.append(
                    Chunk(text=chunk_text, page=page, index=index)
                )
                index += 1

    return chunks
