"""Evaluate answer, retrieval, citation, fact, and abstention regressions."""

import argparse
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from paperwise.config import settings
from paperwise.generation.prompts import ABSTENTION_ANSWER
from paperwise.rag import RAGPipeline


def _validate_expected_pages(value: Any, line_number: int) -> list[int]:
    if not isinstance(value, list) or any(
        type(page) is not int or page < 1 for page in value
    ):
        raise ValueError(
            f"Line {line_number}: expected_pages must contain positive integer page numbers"
        )
    return value


def load_question_cases(path: Path) -> list[dict]:
    """Load validated JSONL cases; facts are groups of equivalent variants."""
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
        expected_pages = _validate_expected_pages(
            case.get("expected_pages"), line_number
        )
        expected_facts = case.get("expected_facts")
        answerable = case.get("answerable", True)

        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Line {line_number}: question must not be empty")
        if not isinstance(answerable, bool):
            raise ValueError(f"Line {line_number}: answerable must be true or false")
        if not isinstance(expected_facts, list):
            raise ValueError(
                f"Line {line_number}: expected_facts must be a list of fact groups"
            )
        if any(
            not isinstance(group, list)
            or not group
            or any(not isinstance(variant, str) or not variant.strip() for variant in group)
            for group in expected_facts
        ):
            raise ValueError(
                f"Line {line_number}: each expected_facts entry must be a non-empty list of strings"
            )
        if answerable and not expected_pages:
            raise ValueError(
                f"Line {line_number}: answerable questions need at least one expected page"
            )
        if answerable and not expected_facts:
            raise ValueError(
                f"Line {line_number}: answerable questions need at least one expected fact"
            )
        if not answerable and (expected_pages or expected_facts):
            raise ValueError(
                f"Line {line_number}: unanswerable questions must have empty expected_pages and expected_facts"
            )

        cases.append(
            {
                "question": question.strip(),
                "expected_pages": expected_pages,
                "expected_facts": [
                    [variant.strip() for variant in group]
                    for group in expected_facts
                ],
                "answerable": answerable,
            }
        )

    return cases


def cited_pages(answer: str) -> set[int]:
    """Extract explicit bracketed citations, including comma-separated pages."""
    citation_groups = re.findall(
        r"\[\s*((?:Seite\s+\d+)(?:\s*,\s*Seite\s+\d+)*)\s*\]",
        answer,
        re.IGNORECASE,
    )
    return {
        int(page)
        for group in citation_groups
        for page in re.findall(r"Seite\s+(\d+)", group, re.IGNORECASE)
    }


def is_abstention(answer: str) -> bool:
    """Return whether the answer is exactly the configured abstention."""
    normalize = lambda text: " ".join(text.strip().strip('"').split()).casefold()
    return normalize(answer) == normalize(ABSTENTION_ANSWER)


def expected_fact_matches(answer: str, facts: list[list[str]]) -> list[bool]:
    """Match each fact independently with Unicode normalization and word boundaries."""
    normalized_answer = _normalize_for_match(answer)
    return [
        any(
            re.search(
                rf"(?<!\w){re.escape(_normalize_for_match(variant))}(?!\w)",
                normalized_answer,
            )
            is not None
            for variant in variants
        )
        for variants in facts
    ]


def _normalize_for_match(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = normalized.translate(
        str.maketrans({character: "-" for character in "‐‑‒–—―−"})
    )
    normalized = re.sub(r"\s+%", "%", normalized)
    return " ".join(normalized.split())


def evaluate_case(
    case: dict,
    answer: str,
    retrieved_pages: set[int],
) -> dict[str, Any]:
    """Evaluate one answer without making a semantic-correctness claim."""
    cited = cited_pages(answer)
    result: dict[str, Any] = {
        "question": case["question"],
        "answerable": case["answerable"],
        "expected_pages": case["expected_pages"],
        "retrieved_pages": sorted(retrieved_pages),
        "explicitly_cited_pages": sorted(cited),
        "unretrieved_cited_pages": sorted(cited - retrieved_pages),
        "answer": answer,
        "abstained": is_abstention(answer),
    }

    if case["answerable"]:
        matches = expected_fact_matches(answer, case["expected_facts"])
        result.update(
            retrieval_hit=bool(retrieved_pages & set(case["expected_pages"])),
            citation_hit=bool(cited & set(case["expected_pages"])),
            expected_facts=case["expected_facts"],
            fact_matches=matches,
            fact_details=[
                {"variants": variants, "matched": matched}
                for variants, matched in zip(case["expected_facts"], matches)
            ],
            matched_facts=sum(matches),
            total_facts=len(matches),
        )
    else:
        result["abstention_correct"] = is_abstention(answer)
    return result


def _rate(hits: int, total: int) -> dict[str, int | float | None]:
    return {
        "hits": hits,
        "total": total,
        "rate": hits / total if total else None,
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, dict[str, int | float | None]]:
    """Calculate distinct rates; intentionally no combined answer accuracy."""
    supported = [result for result in results if result["answerable"]]
    unsupported = [result for result in results if not result["answerable"]]
    fact_hits = sum(result["matched_facts"] for result in supported)
    fact_total = sum(result["total_facts"] for result in supported)
    return {
        "retrieval_hit_at_k": _rate(
            sum(result["retrieval_hit"] for result in supported), len(supported)
        ),
        "citation_hit_rate": _rate(
            sum(result["citation_hit"] for result in supported), len(supported)
        ),
        "fact_match_rate": _rate(fact_hits, fact_total),
        "abstention_accuracy": _rate(
            sum(result["abstention_correct"] for result in unsupported),
            len(unsupported),
        ),
    }


def write_report(output_path: Path, report: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="PDF to evaluate")
    parser.add_argument("questions", type=Path, help="JSONL evaluation questions")
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON report path (default: timestamped file in evaluation/results)",
    )
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
    results = []

    for case in cases:
        answer = pipeline.ask(case["question"])
        retrieved_pages = {citation.page for citation in answer.citations}
        result = evaluate_case(case, answer.answer, retrieved_pages)
        results.append(result)
        if case["answerable"]:
            print(f"\nQuestion: {case['question']}")
            print(
                f"Retrieval: {'PASS' if result['retrieval_hit'] else 'MISS'} "
                f"retrieved={result['retrieved_pages']}"
            )
            print(
                f"Citation:  {'PASS' if result['citation_hit'] else 'MISS'} "
                f"explicitly_cited={result['explicitly_cited_pages']}"
            )
            print(
                f"Source check: unretrieved_cited_pages="
                f"{result['unretrieved_cited_pages']}"
            )
            facts_status = (
                "PASS"
                if result["matched_facts"] == result["total_facts"]
                else "MISS"
            )
            print(
                f"Facts:     {facts_status} "
                f"({result['matched_facts']}/{result['total_facts']})"
            )
            for fact_number, fact in enumerate(result["fact_details"], start=1):
                print(
                    f"  Fact {fact_number}: "
                    f"{'PASS' if fact['matched'] else 'MISS'}"
                )
            print(f"Abstention: {'YES' if result['abstained'] else 'NO'}")
            print(f"Answer: {answer.answer}")
        else:
            status = "PASS" if result["abstention_correct"] else "MISS"
            detail = f"abstained={result['abstention_correct']}"
            print(f"{status}: {case['question']} {detail}")
            if status == "MISS":
                print(f"  answer: {answer.answer}")

    metrics = summarize_results(results)
    timestamp = datetime.now(timezone.utc)
    output_path = args.output or Path("evaluation/results") / (
        f"answer_baseline_{timestamp.strftime('%Y%m%dT%H%M%S_%fZ')}.json"
    )
    report = {
        "schema_version": 1,
        "created_at": timestamp.isoformat(),
        "pdf": args.pdf.name,
        "questions_file": args.questions.name,
        "models": {
            "chat": settings.chat_model,
            "embedding": settings.embedding_model,
        },
        "top_k": settings.top_k,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "chunks_indexed": chunk_count,
        "metrics": metrics,
        "questions": results,
    }
    write_report(output_path, report)

    print("\nMetrics (kept separate; no combined answer accuracy):")
    for metric_name, values in metrics.items():
        rate = "n/a" if values["rate"] is None else f"{values['rate']:.3f}"
        print(f"{metric_name}: {values['hits']}/{values['total']} ({rate})")
    print(f"JSON report: {output_path}")


if __name__ == "__main__":
    main()