from paperwise.models import Chunk


def build_question_prompt(question: str, retrieved: list[tuple[Chunk, float]]) -> str:
    context = "\n\n".join(
        f"[Quelle: Seite {chunk.page}]\n{chunk.text}" for chunk, _ in retrieved
    )
    return f"""Beantworte die Frage ausschließlich anhand des Paper-Kontexts.
Wenn die Information fehlt, antworte exakt: "Im Paper wurde dazu keine ausreichende Information gefunden."
Erfinde keine Fakten. Nenne wichtige Aussagen mit [Seite X]. Antworte auf Deutsch.

Frage: {question}

Paper-Kontext:
{context}
"""