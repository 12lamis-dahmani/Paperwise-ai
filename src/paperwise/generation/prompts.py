from paperwise.models import Chunk


def build_question_prompt(question: str, retrieved: list[tuple[Chunk, float]]) -> str:
    context = "\n\n".join(
        f"[Quelle: Seite {chunk.page}]\n{chunk.text}" for chunk, _ in retrieved
    )

    return f"""Beantworte die Frage ausschließlich anhand des Paper-Kontexts.
Erfinde keine Fakten und verwende nur Informationen aus den angegebenen Quellen.

Wenn die Frage anhand des Kontexts nicht beantwortet werden kann, antworte ausschließlich:
"Im Paper wurde dazu keine ausreichende Information gefunden."

Verwende diesen Satz niemals zusätzlich zu einer inhaltlichen Antwort.
Wenn nur ein Teil der Frage beantwortbar ist, beantworte diesen Teil und kennzeichne fehlende Informationen.

Nenne wichtige Aussagen mit [Seite X]. Antworte auf Deutsch.

Frage: {question}

Paper-Kontext:
{context}
"""