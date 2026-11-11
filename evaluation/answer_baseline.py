"""Evaluate generated answers for supported and unsupported questions."""

import argparse
import json
import re
from pathlib import Path

from paperwise.config import settings
from paperwise.generation.prompts import ABSTENTION_ANSWER
from paperwise.rag import RAGPipeline


def load_question_cases(path: Path) -> list[dict]:
    """Load answer-evaluation cases from JSONL."""
    cases = []

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
            raise ValueError(f"Line {line_number}: expected a JSON object")

        question = case.get("question")
        expected_pages = case.get("expected_pages")
        expected_keywords = case.get("expected_keywords", [])
        answerable = case.get("answerable", True)

        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Line {line_number}: question must not be empty")
        if not isinstance(expected_pages, list) or any(
            type(page) is not int or page < 1 for page in expected_pages
        ):
            raise ValueError(
                f"Line {line_number}: expected_pages must contain "
                "positive integer page numbers"
            )
        if not isinstance(answerable, bool):
            raise ValueError(f"Line {line_number}: answerable must be true or false")
        if not isinstance(expected_keywords, list) or any(
            not isinstance(keyword, str) or not keyword.strip()
            for keyword in expected_keywords
        ):
            raise ValueError(
                f"Line {line_number}: expected_keywords must contain non-empty strings"
            )
        if answerable and not expected_pages:
            raise ValueError(
                f"Line {line_number}: answerable questions need at least one expected page"
            )

        cases.append(
            {
                "question": question.strip(),
                "expected_pages": expected_pages,
                "expected_keywords": [keyword.strip() for keyword in expected_keywords],
                "answerable": answerable,
            }
        )

    return cases


def cited_pages(answer: str) -> set[int]:
    """Extract page numbers explicitly cited by the generated answer."""
    return {
        int(page)
        for page in re.findall(r"\bSeite\s+(\d+)\b", answer, re.IGNORECASE)
    }


def is_abstention(answer: str) -> bool:
    """Return whether the answer is exactly the configured abstention."""
    normalize = lambda text: " ".join(text.strip().strip('"').split()).casefold()
    return normalize(answer) == normalize(ABSTENTION_ANSWER)


def contains_expected_keywords(answer: str, keywords: list[str]) -> bool:
    """Check optional, case-insensitive answer facts from a test case."""
    normalized_answer = answer.casefold()
    return all(keyword.casefold() in normalized_answer for keyword in keywords)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="PDF to evaluate")
    parser.add_argument("questions", type=Path, help="JSONL evaluation questions")
    args = parser.parse_args()

    if not args.pdf.is_file():
        parser.error(f"PDF not found: {args.pdf}")
    if not args.questions.is_file():
        parser.error(f"Questions file not found: {args.questions}")

    try:
        cases = load_question_cases(args.questions)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if not cases:
        parser.error("No evaluation questions found in the JSONL file.")

    pipeline = RAGPipeline(
        settings.ollama_url,
        settings.chat_model,
        settings.embedding_model,
        settings.top_k,
    )
    chunk_count = pipeline.index_pdf(
        args.pdf.read_bytes(), settings.chunk_size, settings.chunk_overlap
    )
    supported_total = supported_passes = 0
    retrieval_total = retrieval_passes = 0
    unsupported_total = unsupported_passes = 0

    for case in cases:
        answer = pipeline.ask(case["question"])
        if case["answerable"]:
            supported_total += 1
            expected_pages = set(case["expected_pages"])
            retrieved_pages = {citation.page for citation in answer.citations}
            answer_pages = cited_pages(answer.answer)
            retrieval_hit = bool(retrieved_pages & expected_pages)
            retrieval_total += 1
            retrieval_passes += int(retrieval_hit)
            passed = (
                retrieval_hit
                and bool(answer_pages & expected_pages)
                and not is_abstention(answer.answer)
                and contains_expected_keywords(
                    answer.answer, case["expected_keywords"]
                )
            )
            supported_passes += int(passed)
            detail = (
                f"expected={sorted(expected_pages)} "
                f"retrieved={sorted(retrieved_pages)} "
                f"cited={sorted(answer_pages)}"
            )
        else:
            unsupported_total += 1
            passed = is_abstention(answer.answer)
            unsupported_passes += int(passed)
            detail = "abstained=" + str(passed)

        status = "PASS" if passed else "MISS"
        print(f"{status}: {case['question']} {detail}")
        if not passed:
            print(f"  answer: {answer.answer}")

    print()
    print(f"Chunks indexed: {chunk_count}")
    print(
        f"Retrieval hit rate: {retrieval_passes}/{retrieval_total} "
        f"({retrieval_passes / retrieval_total:.3f})"
        if retrieval_total
        else "Retrieval hit rate: none"
    )
    print(
        f"Supported answers: {supported_passes}/{supported_total} "
        f"({supported_passes / supported_total:.3f})"
        if supported_total
        else "Supported answers: none"
    )
    print(
        f"Unsupported answers correctly rejected: {unsupported_passes}/{unsupported_total} "
        f"({unsupported_passes / unsupported_total:.3f})"
        if unsupported_total
        else "Unsupported answers: none"
    )


if __name__ == "__main__":
    main()