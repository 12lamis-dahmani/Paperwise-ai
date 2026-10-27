import pytest

from paperwise.ingestion.chunker import chunk_pages


def test_empty_pages() -> None:
    chunks = chunk_pages([])

    assert chunks == []


def test_empty_page_text() -> None:
    chunks = chunk_pages([(1, "")])

    assert chunks == []


def test_whitespace_only_page() -> None:
    chunks = chunk_pages([(1, "   ")])

    assert chunks == []


def test_text_smaller_than_chunk_size() -> None:
    chunks = chunk_pages([(1, "Hello World")], size=100, overlap=20)

    assert len(chunks) == 1
    assert chunks[0].text == "Hello World"
    assert chunks[0].page == 1
    assert chunks[0].index == 0


def test_text_equal_to_chunk_size() -> None:
    chunks = chunk_pages([(1, "abcdefghij")], size=10, overlap=2)

    assert len(chunks) == 1
    assert chunks[0].text == "abcdefghij"


def test_chunking_without_overlap() -> None:
    chunks = chunk_pages([(1, "abcdefghij")], size=4, overlap=0)

    assert [chunk.text for chunk in chunks] == ["abcd", "efgh", "ij"]


def test_multiple_pages_preserve_page_numbers() -> None:
    pages = [
        (1, "abcdefgh"),
        (2, "12345678"),
    ]

    chunks = chunk_pages(pages, size=5, overlap=2)

    assert [chunk.text for chunk in chunks] == [
        "abcde",
        "defgh",
        "12345",
        "45678",
    ]
    assert [chunk.page for chunk in chunks] == [1, 1, 2, 2]


def test_chunk_indices_are_sequential() -> None:
    pages = [
        (1, "abcdefgh"),
        (2, "12345678"),
    ]

    chunks = chunk_pages(pages, size=5, overlap=2)

    assert [chunk.index for chunk in chunks] == [0, 1, 2, 3]


def test_whitespace_is_trimmed() -> None:
    chunks = chunk_pages([(1, "  hello  ")], size=20, overlap=0)

    assert chunks[0].text == "hello"


def test_chunk_pages_preserves_page_and_overlap() -> None:
    chunks = chunk_pages([(2, "abcdefghij")], size=6, overlap=2)

    assert [chunk.text for chunk in chunks] == ["abcdef", "efghij"]
    assert all(chunk.page == 2 for chunk in chunks)


def test_chunk_pages_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        chunk_pages([(1, "text")], size=4, overlap=4)


@pytest.mark.parametrize(
    "size, overlap",
    [
        (0, 0),
        (-1, 0),
        (4, -1),
        (4, 4),
        (4, 5),
    ],
)
def test_invalid_chunk_parameters(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_pages([(1, "abcdefgh")], size=size, overlap=overlap)