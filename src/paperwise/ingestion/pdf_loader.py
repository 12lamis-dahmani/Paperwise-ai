import fitz


def load_pdf(pdf_bytes: bytes) -> list[tuple[int, str]]:
    """Return normalized, non-empty page text as (one_based_page, text)."""
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return [
            (page_number, " ".join(page.get_text("text").split()))
            for page_number, page in enumerate(document, start=1)
            if " ".join(page.get_text("text").split())
        ]
    finally:
        document.close()