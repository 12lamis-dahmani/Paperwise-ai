from paperwise.generation.prompts import build_question_prompt
from paperwise.models import Chunk


def test_prompt_contains_question_and_page_source() -> None:
    prompt = build_question_prompt("Welche Methode?", [(Chunk("PPO", 4, 0), 0.9)])

    assert "Welche Methode?" in prompt
    assert "Seite 4" in prompt
    assert "Erfinde keine Fakten" in prompt