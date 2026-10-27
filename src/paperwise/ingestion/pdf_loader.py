import re

import pymupdf


def normalize_page_text(text: str) -> str:
    """Normalize PDF text while retaining paragraph boundaries."""
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    paragraphs = []
    for paragraph in re.split(r"\n\s*\n", text):
        normalized = re.sub(r"\s*\n\s*", " ", paragraph)
        normalized = re.sub(r"[ \t]+", " ", normalized).strip()
        if normalized:
            paragraphs.append(normalized)
    return "\n\n".join(paragraphs)


def load_pdf(pdf_bytes: bytes) -> list[tuple[int, str]]:
    """Return normalized, non-empty page text with paragraphs preserved."""
    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        pages = []
        for page_number, page in enumerate(document, start=1):
            text = normalize_page_text(page.get_text("text"))
            if text:
                pages.append((page_number, text))
        return pages
    finally:
        document.close()