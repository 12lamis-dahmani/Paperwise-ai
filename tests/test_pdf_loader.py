import pymupdf

from paperwise.ingestion.pdf_loader import (
    extract_page_text,
    load_pdf,
    normalize_page_text,
)


def test_normalize_page_text_preserves_paragraphs_and_repairs_hyphenation() -> None:
    text = "First para-\ngraph.\ncontinues here.\n\nSecond paragraph."

    assert normalize_page_text(text) == (
        "First paragraph. continues here.\n\nSecond paragraph."
    )


def test_load_pdf_extracts_each_page_once_with_page_numbers() -> None:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Page one\n\nA paragraph.")
    pdf_bytes = document.tobytes()
    document.close()

    assert load_pdf(pdf_bytes) == [(1, "Page one\n\nA paragraph.")]


def test_extract_page_text_reads_two_columns_in_order() -> None:
    class FakePage:
        rect = type("Rect", (), {"width": 600})()

        def get_text(self, kind: str, sort: bool = False):
            assert kind == "blocks"
            assert sort is False
            return [
                (310, 20, 550, 40, "Right top\n", 0, 0),
                (40, 100, 280, 120, "Left bottom\n", 0, 0),
                (40, 20, 280, 40, "Left top\n", 0, 0),
                (310, 100, 550, 120, "Right bottom\n", 0, 0),
            ]

    assert extract_page_text(FakePage()) == (
        "Left top\n\nLeft bottom\n\nRight top\n\nRight bottom"
    )

def test_load_pdf_multiple_pages() -> None:
    document = pymupdf.open()

    for number in range(1, 4):
        page = document.new_page()
        page.insert_text((72, 72), f"Page {number}")

    pdf_bytes = document.tobytes()
    document.close()

    pages = load_pdf(pdf_bytes)

    assert [page_number for page_number, _ in pages] == [1, 2, 3]


def test_load_pdf_skips_empty_pages() -> None:
    document = pymupdf.open()

    document.new_page()

    page = document.new_page()
    page.insert_text((72, 72), "Scientific paper")

    pdf_bytes = document.tobytes()
    document.close()

    assert load_pdf(pdf_bytes) == [(2, "Scientific paper")]


def test_normalize_page_text_handles_whitespace() -> None:
    text = "  First   sentence.  \n  Second sentence.  "

    assert normalize_page_text(text) == (
        "First sentence. Second sentence."
    )

    