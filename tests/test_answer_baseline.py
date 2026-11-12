import json

import pytest

from paperwise.generation.prompts import ABSTENTION_ANSWER
from evaluation.answer_baseline import (
    cited_pages,
    evaluate_case,
    expected_fact_matches,
    is_abstention,
    load_question_cases,
    summarize_results,
    write_report,
)


def write_jsonl(tmp_path, cases):
    path = tmp_path / "questions.jsonl"
    path.write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in cases),
        encoding="utf-8",
    )
    return path


def answerable_case(**overrides):
    case = {
        "question": "Welche Genauigkeit?",
        "expected_pages": [4],
        "expected_facts": [["85.4%", "85,4%"], ["classification accuracy"]],
        "answerable": True,
    }
    case.update(overrides)
    return case


def test_load_rejects_missing_expected_facts(tmp_path):
    path = write_jsonl(
        tmp_path,
        [{"question": "Fact?", "expected_pages": [1], "answerable": True}],
    )

    with pytest.raises(ValueError, match="expected_facts must be a list"):
        load_question_cases(path)


def test_load_rejects_empty_expected_facts_for_answerable_case(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            {
                "question": "Fact?",
                "expected_pages": [1],
                "expected_facts": [],
                "answerable": True,
            }
        ],
    )

    with pytest.raises(ValueError, match="at least one expected fact"):
        load_question_cases(path)


def test_load_accepts_empty_facts_for_unanswerable_case(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            {
                "question": "Unknown?",
                "expected_pages": [],
                "expected_facts": [],
                "answerable": False,
            }
        ],
    )

    assert load_question_cases(path)[0]["expected_facts"] == []


def test_load_rejects_malformed_fact_group(tmp_path):
    path = write_jsonl(
        tmp_path,
        [answerable_case(expected_facts=[["85.4%"], []])],
    )

    with pytest.raises(ValueError, match="non-empty list of strings"):
        load_question_cases(path)


def test_question_dataset_has_facts_for_every_answerable_case():
    from pathlib import Path

    cases = load_question_cases(Path("evaluation/questions.jsonl"))

    assert cases
    assert all(
        case["expected_facts"]
        for case in cases
        if case["answerable"]
    )


def test_fact_groups_support_alternative_spellings():
    matches = expected_fact_matches(
        "Die Genauigkeit beträgt 85,4%.",
        [["85.4%", "85,4%"]],
    )

    assert matches == [True]


def test_facts_are_checked_individually():
    matches = expected_fact_matches(
        "Das Netzwerk erreicht 85.4% classification accuracy.",
        [["85.4%", "85,4%"], ["classification accuracy"], ["15-53-1"]],
    )

    assert matches == [True, True, False]


@pytest.mark.parametrize(
    ("answer", "fact"),
    [
        ("Die Architektur ist 15-53-1.", "15–53–1"),
        ("Die Genauigkeit beträgt 85.4 %.", "85.4%"),
        ("Die Genauigkeit beträgt 85,4%.", "85,4 %"),
    ],
)
def test_fact_matching_normalizes_dashes_and_percent_spacing(answer, fact):
    assert expected_fact_matches(answer, [[fact]]) == [True]


def test_fact_matching_uses_word_boundaries():
    assert expected_fact_matches("The value is 13745.", [["3745"]]) == [False]


def test_cited_pages_only_reads_explicit_bracketed_citations():
    assert cited_pages("The answer is [Seite 4], not plain Seite 5.") == {4}


def test_cited_pages_extracts_comma_separated_pages_in_one_bracket():
    assert cited_pages("Belege [Seite 1, Seite 2] stützen die Antwort.") == {1, 2}


def test_wrong_explicit_citation_does_not_count_as_citation_hit():
    result = evaluate_case(
        answerable_case(),
        "Das Netzwerk erreicht 85.4% classification accuracy [Seite 5].",
        {4},
    )

    assert result["retrieval_hit"] is True
    assert result["citation_hit"] is False
    assert result["matched_facts"] == 2
    assert result["unretrieved_cited_pages"] == [5]


def test_retrieved_citations_are_not_reported_as_unretrieved():
    result = evaluate_case(
        answerable_case(),
        "Das Netzwerk erreicht 85.4% classification accuracy [Seite 4].",
        {4},
    )

    assert result["unretrieved_cited_pages"] == []


def test_retrieval_citations_are_not_treated_as_llm_citations():
    result = evaluate_case(
        answerable_case(),
        "Das Netzwerk erreicht 85.4% classification accuracy.",
        {4},
    )

    assert result["retrieved_pages"] == [4]
    assert result["explicitly_cited_pages"] == []
    assert result["citation_hit"] is False


def test_abstention_is_detected_for_answerable_questions():
    result = evaluate_case(
        answerable_case(),
        ABSTENTION_ANSWER,
        {4},
    )

    assert result["abstained"] is True
    assert result["matched_facts"] == 0


def test_correct_abstention_counts_for_unsupported_question():
    case = {
        "question": "Welche GPU?",
        "expected_pages": [],
        "expected_facts": [],
        "answerable": False,
    }

    result = evaluate_case(case, ABSTENTION_ANSWER, set())
    metrics = summarize_results([result])

    assert result["abstention_correct"] is True
    assert metrics["abstention_accuracy"] == {
        "hits": 1,
        "total": 1,
        "rate": 1.0,
    }


def test_metrics_are_separate_and_have_no_combined_accuracy():
    answer_result = evaluate_case(
        answerable_case(),
        "85.4% classification accuracy [Seite 4].",
        {4},
    )
    abstention_result = evaluate_case(
        {
            "question": "Unknown?",
            "expected_pages": [],
            "expected_facts": [],
            "answerable": False,
        },
        ABSTENTION_ANSWER,
        set(),
    )

    metrics = summarize_results([answer_result, abstention_result])

    assert set(metrics) == {
        "retrieval_hit_at_k",
        "citation_hit_rate",
        "fact_match_rate",
        "abstention_accuracy",
    }
    assert metrics["fact_match_rate"] == {"hits": 2, "total": 2, "rate": 1.0}


def test_report_is_written_as_readable_json(tmp_path):
    report_path = tmp_path / "results" / "evaluation.json"
    report = {"metrics": {"fact_match_rate": {"hits": 1, "total": 1, "rate": 1.0}}}

    write_report(report_path, report)

    assert json.loads(report_path.read_text(encoding="utf-8")) == report


@pytest.mark.parametrize(
    "answer",
    [
        ABSTENTION_ANSWER,
        f'  "{ABSTENTION_ANSWER}"  ',
        ABSTENTION_ANSWER.upper(),
    ],
)
def test_is_abstention_accepts_normalized_exact_sentence(answer):
    assert is_abstention(answer)


def test_is_abstention_rejects_extra_claims():
    assert not is_abstention(ABSTENTION_ANSWER + " Das Paper verwendet PPO.")