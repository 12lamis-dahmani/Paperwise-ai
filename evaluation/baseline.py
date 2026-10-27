
"""Evaluate retrieval Hit Rate@k using a PDF and JSONL test questions."""

import argparse
import json
from pathlib import Path

from paperwise.config import settings
from paperwise.ingestion import chunk_pages, load_pdf
from paperwise.retrieval import OllamaEmbeddings, Retriever


def load_questions(path: Path) -> list[dict]:
    """Load and validate answerable questions from a JSONL file."""
    questions = []

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue

        try:
            case = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON on line {line_number}: {error}"
            ) from error

        if not isinstance(case, dict):
            raise ValueError(
                f"Line {line_number}: expected a JSON object"
            )

        question = case.get("question")
        expected_pages = case.get("expected_pages")

        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                f"Line {line_number}: question must not be empty"
            )

        if not isinstance(expected_pages, list) or any(
            type(page) is not int or page < 1
            for page in expected_pages
        ):
            raise ValueError(
                f"Line {line_number}: expected_pages must contain "
                "positive integer page numbers"
            )

        answerable = case.get("answerable", True)

        if not isinstance(answerable, bool):
            raise ValueError(
                f"Line {line_number}: answerable must be true or false"
            )

        # Unanswerable questions belong to answer evaluation,
        # not retrieval Hit Rate@k.
        if not answerable:
            continue

        if not expected_pages:
            raise ValueError(
                f"Line {line_number}: answerable questions need "
                "at least one expected page"
            )

        questions.append(
            {
                "question": question.strip(),
                "expected_pages": expected_pages,
            }
        )

    return questions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="PDF to evaluate")
    parser.add_argument("questions", type=Path, help="JSONL test dataset")
    parser.add_argument(
        "--top-k",
        type=int,
        default=settings.top_k,
        help="Number of chunks retrieved per question",
    )

    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    if not args.pdf.is_file():
        parser.error(f"PDF not found: {args.pdf}")

    if not args.questions.is_file():
        parser.error(f"Questions file not found: {args.questions}")

    try:
        questions = load_questions(args.questions)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    if not questions:
        parser.error(
            "No answerable questions found in the JSONL file."
        )

    chunks = chunk_pages(
        load_pdf(args.pdf.read_bytes()),
        settings.chunk_size,
        settings.chunk_overlap,
    )

    if not chunks:
        parser.error("PDF contains no extractable text.")

    retriever = Retriever(
        OllamaEmbeddings(
            settings.ollama_url,
            settings.embedding_model,
        ),
        args.top_k,
    )

    print(f"Indexing {len(chunks)} chunks...")
    retriever.index(chunks)

    hits = 0

    for case in questions:
        question = case["question"]
        expected_pages = set(case["expected_pages"])

        results = retriever.retrieve(question)
        retrieved_pages = {chunk.page for chunk, _ in results}

        found = bool(retrieved_pages & expected_pages)
        hits += int(found)

        status = "PASS" if found else "MISS"

        print(
            f"{status}: {question} "
            f"expected={sorted(expected_pages)} "
            f"retrieved={sorted(retrieved_pages)}"
        )

    hit_rate = hits / len(questions)

    print()
    print(f"Questions evaluated: {len(questions)}")
    print(f"Chunks indexed: {len(chunks)}")
    print(f"Embedding model: {settings.embedding_model}")
    print(
        f"Hit Rate@{args.top_k}: "
        f"{hit_rate:.3f} ({hits}/{len(questions)})"
    )


if __name__ == "__main__":
    main()
