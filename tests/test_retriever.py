import pytest

from paperwise.models import Chunk
from paperwise.retrieval.vector_store import (
    InMemoryVectorStore,
    cosine_similarity,
)


def test_cosine_similarity_identical_vectors() -> None:
    assert cosine_similarity([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)


def test_cosine_similarity_opposite_vectors() -> None:
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_vector_store_respects_top_k() -> None:
    store = InMemoryVectorStore()

    chunks = [
        Chunk("first", 1, 0),
        Chunk("second", 2, 1),
        Chunk("third", 3, 2),
    ]

    store.add(
        chunks,
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [0.0, 1.0],
        ],
    )

    results = store.search([1.0, 0.0], top_k=2)

    assert len(results) == 2
    assert [chunk.text for chunk, _ in results] == ["first", "second"]


def test_new_index_replaces_old_index() -> None:
    store = InMemoryVectorStore()

    store.add(
        [Chunk("old paper", 1, 0)],
        [[1.0, 0.0]],
    )

    store.add(
        [Chunk("new paper", 2, 0)],
        [[0.0, 1.0]],
    )

    results = store.search([0.0, 1.0], top_k=5)

    assert len(results) == 1
    assert results[0][0].text == "new paper"


def test_vector_store_rejects_mismatched_lengths() -> None:
    store = InMemoryVectorStore()

    with pytest.raises(ValueError):
        store.add(
            [Chunk("first", 1, 0), Chunk("second", 2, 1)],
            [[1.0, 0.0]],
        )