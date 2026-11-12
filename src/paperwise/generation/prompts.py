import re

from paperwise.models import Chunk


ABSTENTION_ANSWER = "Im Paper wurde dazu keine ausreichende Information gefunden."


def remove_mixed_abstention(answer: str) -> str:
    """Keep a standalone abstention, but remove it from substantive answers."""
    if answer.strip() == ABSTENTION_ANSWER:
        return answer

    answer = re.sub(
        re.escape(ABSTENTION_ANSWER),
        "",
        answer,
        flags=re.IGNORECASE,
    )
    answer = re.sub(r"[ \t]+\n", "\n", answer)
    answer = re.sub(r"\n{3,}", "\n\n", answer)
    return answer.strip()


def build_question_prompt(question: str, retrieved: list[tuple[Chunk, float]]) -> str:
    context = "\n\n".join(
        f"[Quelle: Seite {chunk.page}]\n{chunk.text}" for chunk, _ in retrieved
    )

    return f"""Beantworte die Frage ausschließlich anhand des Paper-Kontexts.
Erfinde keine Fakten und verwende nur Informationen aus den angegebenen Quellen.

    Antworte ausschließlich mit der Ablehnung, wenn der Kontext keinen Teil der Frage beantwortet:
"{ABSTENTION_ANSWER}"

Verwende diesen Satz niemals zusätzlich zu einer inhaltlichen Antwort.
    Bei mehrteiligen Fragen und Vergleichen prüfe jeden erfragten Aspekt einzeln.
    Beantworte alle Aspekte, die durch den Kontext belegt sind. Wenn nur ein einzelner
    Aspekt fehlt, benenne nur diesen als nicht angegeben; lehne nicht die gesamte Frage ab.
    Stelle Vergleiche übersichtlich nebeneinander, zum Beispiel als Tabelle oder parallele Aufzählung.
    Ordne Eigenschaften klar dem jeweiligen Modell oder Gegenstand zu. Behaupte keine
    gemeinsamen oder gleichen Eigenschaften, wenn der Kontext das nicht ausdrücklich sagt.

Nenne wichtige Aussagen mit [Seite X]. Antworte auf Deutsch.

Frage: {question}

Paper-Kontext:
{context}
"""