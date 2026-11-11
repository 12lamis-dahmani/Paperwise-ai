
import json

import pytest

from paperwise.generation.prompts import ABSTENTION_ANSWER
from evaluation.answer_baseline import (
    cited_pages,
    contains_expected_keywords,
    is_abstention,
    load_question_cases,
)


# --------------------------------------------------
# 1. Tests for cited_pages()
# --------------------------------------------------

@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("Methode [Seite 1] und Ergebnis **Seite 4**.", {1, 4}),
        ("Die Ergebnisse stehen auf Seite 12.", {12}),
        ("Siehe seite 3 und SEITE 8.", {3, 8}),
        ("Seite 2, Seite 2 und Seite 5.", {2, 5}),
        ("Keine Seitenreferenz vorhanden.", set()),
        ("", set()),
        ("Seite abc", set()),
        ("Seite -3", set()),
        ("Seite 1 und Seite 10", {1, 10}),
    ],
)
def test_cited_pages(answer, expected):
    assert cited_pages(answer) == expected


# --------------------------------------------------
# 2. Tests for is_abstention()
# --------------------------------------------------

def test_is_abstention_exact():
    assert is_abstention(ABSTENTION_ANSWER)


def test_is_abstention_quoted():
    assert is_abstention(f'"{ABSTENTION_ANSWER}"')


def test_is_abstention_case_insensitive():
    assert is_abstention(ABSTENTION_ANSWER.upper())


def test_is_abstention_extra_whitespace():
    answer = f"  \n {ABSTENTION_ANSWER} \n "
    assert is_abstention(answer)


def test_is_abstention_rejects_normal_answer():
    assert not is_abstention("Das Modell verwendet PPO.")


def test_is_abstention_rejects_partial_answer():
    assert not is_abstention(ABSTENTION_ANSWER + " Weitere Informationen.")


def test_is_abstention_rejects_empty_answer():
    assert not is_abstention("")


# --------------------------------------------------
# 3. Tests for contains_expected_keywords()
# --------------------------------------------------

@pytest.mark.parametrize(
    ("answer", "keywords", "expected"),
    [
        ("Das Modell verwendet PPO.", ["ppo"], True),
        ("Training mit PPO und LSTM.", ["PPO", "LSTM"], True),
        ("Training mit PPO.", ["PPO", "LSTM"], False),
        ("Das Modell nutzt Reinforcement Learning.", ["learning"], True),
        ("Keine Information vorhanden.", ["PPO"], False),
        ("", ["PPO"], False),
        ("Beliebige Antwort", [], True),
        ("PPO PPO PPO", ["PPO", "PPO"], True),
    ],
)
def test_contains_expected_keywords(answer, keywords, expected):
    assert contains_expected_keywords(answer, keywords) is expected


# --------------------------------------------------
# 4. Tests for valid JSONL cases
# --------------------------------------------------

def write_jsonl(tmp_path, cases):
    path = tmp_path / "questions.jsonl"
    path.write_text(
        "\n".join(json.dumps(case) for case in cases),
        encoding="utf-8",
    )
    return path


def test_load_valid_answerable_question(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            {
                "question": "Welche Methode?",
                "expected_pages": [1],
                "expected_keywords": ["PPO"],
                "answerable": True,
            }
        ],
    )

    cases = load_question_cases(path)

    assert len(cases) == 1
    assert cases[0] == {
        "question": "Welche Methode?",
        "expected_pages": [1],
        "expected_keywords": ["PPO"],
        "answerable": True,
    }


def test_load_unanswerable_question(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            {
                "question": "Welche GPU?",
                "expected_pages": [],
                "answerable": False,
            }
        ],
    )

    cases = load_question_cases(path)

    assert len(cases) == 1
    assert cases[0]["answerable"] is False
    assert cases[0]["expected_pages"] == []
    assert cases[0]["expected_keywords"] == []


def test_load_multiple_questions(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            {
                "question": "Welche Methode?",
                "expected_pages": [1, 3],
                "expected_keywords": ["PPO"],
            },
            {
                "question": "Welche GPU?",
                "expected_pages": [],
                "answerable": False,
            },
        ],
    )

    cases = load_question_cases(path)

    assert len(cases) == 2
    assert cases[0]["answerable"] is True
    assert cases[1]["answerable"] is False


def test_load_applies_defaults(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            {
                "question": "Welche Methode?",
                "expected_pages": [2],
            }
        ],
    )

    cases = load_question_cases(path)

    assert cases[0]["answerable"] is True
    assert cases[0]["expected_keywords"] == []


def test_load_trims_whitespace(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            {
                "question": "  Welche Methode?  ",
                "expected_pages": [2],
                "expected_keywords": ["  PPO  "],
            }
        ],
    )

    cases = load_question_cases(path)

    assert cases[0]["question"] == "Welche Methode?"
    assert cases[0]["expected_keywords"] == ["PPO"]


def test_load_empty_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    assert load_question_cases(path) == []


def test_load_skips_empty_lines(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(
        '\n\n{"question": "Test?", "expected_pages": [1]}\n\n',
        encoding="utf-8",
    )

    cases = load_question_cases(path)

    assert len(cases) == 1


# --------------------------------------------------
# 5. Tests for invalid JSONL cases
# --------------------------------------------------

@pytest.mark.parametrize(
    ("case", "error_message"),
    [
        (
            {"question": "", "expected_pages": [1]},
            "question must not be empty",
        ),
        (
            {"question": "Test?", "expected_pages": []},
            "answerable questions need at least one expected page",
        ),
        (
            {"question": "Test?", "expected_pages": [-1]},
            "expected_pages must contain",
        ),
        (
            {"question": "Test?", "expected_pages": [0]},
            "expected_pages must contain",
        ),
        (
            {"question": "Test?", "expected_pages": ["1"]},
            "expected_pages must contain",
        ),
        (
            {"question": "Test?", "expected_pages": [True]},
            "expected_pages must contain",
        ),
        (
            {"question": "Test?", "expected_pages": None},
            "expected_pages must contain",
        ),
        (
            {"question": "Test?", "expected_pages": [1], "answerable": "yes"},
            "answerable must be true or false",
        ),
        (
            {"question": "Test?", "expected_pages": [1], "expected_keywords": "PPO"},
            "expected_keywords must contain",
        ),
        (
            {"question": "Test?", "expected_pages": [1], "expected_keywords": [""]},
            "expected_keywords must contain",
        ),
        (
            {"question": "Test?", "expected_pages": [1], "expected_keywords": [123]},
            "expected_keywords must contain",
        ),
    ],
)
def test_load_rejects_invalid_cases(tmp_path, case, error_message):
    path = write_jsonl(tmp_path, [case])

    with pytest.raises(ValueError, match=error_message):
        load_question_cases(path)


def test_load_rejects_invalid_json(tmp_path):
    path = tmp_path / "invalid.jsonl"
    path.write_text(
        '{"question": "Test?", "expected_pages": [1]\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid JSON on line 1"):
        load_question_cases(path)


def test_load_rejects_non_object_json(tmp_path):
    path = tmp_path / "invalid.jsonl"
    path.write_text('["not", "an", "object"]', encoding="utf-8")

    with pytest.raises(ValueError, match="expected a JSON object"):
        load_question_cases(path)


def test_load_reports_correct_line_number(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(
        '\n{"question": "Valid?", "expected_pages": [1]}\ninvalid-json',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid JSON on line 3"):
        load_question_cases(path)
