from paperwise.generation.prompts import build_question_prompt
from paperwise.models import Chunk


def test_prompt_contains_question_and_page_source() -> None:
    prompt = build_question_prompt(
        "Welche Methode?",
        [(Chunk("PPO", 4, 0), 0.9)],
    )

    assert "Welche Methode?" in prompt
    assert "Seite 4" in prompt
    assert "PPO" in prompt
    assert "Erfinde keine Fakten" in prompt


def test_prompt_contains_multiple_sources() -> None:
    retrieved = [
        (Chunk("First method", 2, 0), 0.9),
        (Chunk("Second method", 5, 1), 0.8),
    ]

    prompt = build_question_prompt("Welche Methoden?", retrieved)

    assert "First method" in prompt
    assert "Second method" in prompt
    assert "Seite 2" in prompt
    assert "Seite 5" in prompt


def test_prompt_handles_empty_context() -> None:
    prompt = build_question_prompt("Was ist das Ziel?", [])

    assert "Was ist das Ziel?" in prompt
    assert "Im Paper wurde dazu keine ausreichende Information gefunden." in prompt


def test_prompt_requires_source_based_answers() -> None:
    prompt = build_question_prompt(
        "Was sind die Ergebnisse?",
        [(Chunk("Accuracy: 85.4%", 4, 0), 0.9)],
    )

    assert "ausschließlich" in prompt
    assert "[Seite X]" in prompt
    assert "Antworte auf Deutsch" in prompt