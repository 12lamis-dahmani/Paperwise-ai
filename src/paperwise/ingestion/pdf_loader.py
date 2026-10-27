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


def extract_page_text(page: pymupdf.Page) -> str:
    """Read text blocks in column order from left to right, top to bottom."""
    blocks = page.get_text("blocks", sort=False)
    if not blocks:
        return ""

    page_width = page.rect.width
    midpoint = page_width / 2
    left_blocks = []
    right_blocks = []

    for block in blocks:
        x0, y0, x1, _y1, text = block[:5]
        if not text.strip() or len(block) >= 7 and block[6] != 0:
            continue
        target = left_blocks if (x0 + x1) / 2 < midpoint else right_blocks
        target.append((y0, x0, text))

    ordered_blocks = sorted(left_blocks) + sorted(right_blocks)
    return "\n\n".join(
        normalized
        for _, _, text in ordered_blocks
        if (normalized := normalize_page_text(text))
    )


def load_pdf(pdf_bytes: bytes) -> list[tuple[int, str]]:
    """Return normalized, non-empty page text with paragraphs preserved."""
    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        pages = []
        for page_number, page in enumerate(document, start=1):
            text = extract_page_text(page)
            if text:
                pages.append((page_number, text))
        return pages
    finally:
        document.close()