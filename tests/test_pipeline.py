from paperwise.generation.prompts import (
    ABSTENTION_ANSWER,
    build_question_prompt,
    remove_mixed_abstention,
)
from paperwise.models import Chunk, Citation
from paperwise.rag.pipeline import RAGPipeline


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


def test_comparison_prompt_answers_supported_parts_individually() -> None:
    question = (
        "Welche Unterschiede bestehen zwischen dem Best-Scoring-Point-Netzwerk "
        "und dem Success-Rate-Netzwerk hinsichtlich ihrer Aufgaben, Eingaben und Ausgaben?"
    )
    retrieved = [
        (
            Chunk(
                "Das erste Netz sagt den besten Schusspunkt voraus.",
                3,
                0,
            ),
            0.9,
        ),
        (
            Chunk(
                "Das Best-Scoring-Point-Netz hat 15-53-1 und gibt die Y-Koordinate aus. "
                "Das Success-Rate-Netz hat 19-80-2 und zwei Ausgaben: Tor und kein Tor.",
                4,
                1,
            ),
            0.8,
        ),
    ]

    prompt = build_question_prompt(question, retrieved)
    normalized_prompt = " ".join(prompt.split())

    assert question in prompt
    assert "Das erste Netz sagt den besten Schusspunkt voraus." in prompt
    assert "15-53-1" in prompt and "19-80-2" in prompt
    assert "Seite 3" in prompt and "Seite 4" in prompt
    assert "prüfe jeden erfragten Aspekt einzeln" in normalized_prompt
    assert "lehne nicht die gesamte Frage ab" in normalized_prompt
    assert "Vergleiche übersichtlich nebeneinander" in normalized_prompt
    assert "Behaupte keine gemeinsamen oder gleichen Eigenschaften" in normalized_prompt


def test_standalone_abstention_is_unchanged() -> None:
    assert remove_mixed_abstention(ABSTENTION_ANSWER) == ABSTENTION_ANSWER


def test_mixed_abstention_is_removed_from_substantive_answer() -> None:
    answer = (
        "Polarkoordinaten drücken Beziehungen zwischen Objekten aus [Seite 3].\n\n"
        f"{ABSTENTION_ANSWER}"
    )

    assert remove_mixed_abstention(answer) == (
        "Polarkoordinaten drücken Beziehungen zwischen Objekten aus [Seite 3]."
    )


def test_pipeline_sanitizes_generated_answer_without_ollama() -> None:
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.retriever = type(
        "RetrieverStub",
        (),
        {"retrieve": lambda self, question: [(Chunk("Relevant text", 3, 0), 0.9)]},
    )()
    pipeline.generator = type(
        "GeneratorStub",
        (),
        {
            "answer": lambda self, prompt: (
                f"Relevant answer [Seite 3]. {ABSTENTION_ANSWER}"
            )
        },
    )()

    response = pipeline.ask("Warum Polarkoordinaten?")

    assert response.answer == "Relevant answer [Seite 3]."
    assert response.citations == [Citation(page=3, text="Relevant text", score=0.9)]


def test_network_comparison_answers_with_retrieved_page_citations() -> None:
    question = (
        "Welche Unterschiede bestehen zwischen dem Best-Scoring-Point-Netzwerk "
        "und dem Success-Rate-Netzwerk hinsichtlich ihrer Aufgaben, Eingaben und Ausgaben?"
    )
    chunks = [
        (Chunk("The first network predicts the point to shoot at.", 3, 0), 0.9),
        (
            Chunk(
                "The target is the y-coordinate. The success-rate output is goal/no-goal. "
                "The topologies are 15-53-1 and 19-80-2, respectively.",
                4,
                1,
            ),
            0.8,
        ),
    ]
    generated = (
        "| Aspekt | Best-Scoring-Point | Success-Rate |\n"
        "| Aufgabe | Schusspunkt [Seite 3] | Tor/kein Tor [Seite 4] |\n"
        "| Architektur | 15-53-1 [Seite 4] | 19-80-2 [Seite 4] |\n"
        "| Ausgabe | Y-Koordinate [Seite 4] | Erfolgsrate 0 bis 1 [Seite 4] |"
    )
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.retriever = type(
        "RetrieverStub", (), {"retrieve": lambda self, prompt: chunks}
    )()
    generator_stub = type(
        "GeneratorStub",
        (),
        {"answer": lambda self, prompt: generated},
    )()
    pipeline.generator = generator_stub

    response = pipeline.ask(question)

    assert "Best-Scoring-Point" in response.answer
    assert "Success-Rate" in response.answer
    assert "[Seite 3]" in response.answer
    assert "[Seite 4]" in response.answer
    assert ABSTENTION_ANSWER not in response.answer
    assert {citation.page for citation in response.citations} == {3, 4}