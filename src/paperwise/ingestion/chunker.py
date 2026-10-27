from paperwise.models import Chunk


def chunk_pages(
    pages: list[tuple[int, str]], size: int = 1200, overlap: int = 200
) -> list[Chunk]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("overlap must be non-negative and smaller than size")

    chunks: list[Chunk] = []
    index = 0
    for page, text in pages:
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(text=chunk_text, page=page, index=index))
                index += 1
            if end == len(text):
                break
            start = end - overlap
    return chunks