import hashlib

import streamlit as st

from paperwise.config import settings
from paperwise.rag import RAGPipeline


st.set_page_config(page_title="Paperwise AI", page_icon="📄", layout="wide")
st.title("Paperwise AI")
st.caption("Lokaler Research Paper Assistant mit RAG und Ollama")

uploaded_file = st.file_uploader("Wissenschaftliches PDF hochladen", type=["pdf"])
if uploaded_file:
    pdf_bytes = uploaded_file.getvalue()
    document_id = hashlib.sha256(pdf_bytes).hexdigest()
    if st.session_state.get("document_id") != document_id:
        with st.spinner("Paper wird gelesen und indexiert..."):
            try:
                pipeline = RAGPipeline(
                    settings.ollama_url,
                    settings.chat_model,
                    settings.embedding_model,
                    settings.top_k,
                )
                count = pipeline.index_pdf(
                    pdf_bytes, settings.chunk_size, settings.chunk_overlap
                )
                st.session_state.update(
                    document_id=document_id,
                    pipeline=pipeline,
                    file_name=uploaded_file.name,
                    chunk_count=count,
                )
            except Exception as error:
                st.error(f"Indexierung fehlgeschlagen: {error}")

    if st.session_state.get("document_id") == document_id:
        st.success(
            f"{st.session_state.file_name} ist bereit: "
            f"{st.session_state.chunk_count} Abschnitte indexiert."
        )
        question = st.text_input(
            "Deine Frage", placeholder="Welche Methode verwenden die Autoren?"
        )
        if question:
            with st.spinner("Antwort wird erstellt..."):
                try:
                    result = st.session_state.pipeline.ask(question)
                    st.subheader("Antwort")
                    st.write(result.answer)
                    with st.expander("Verwendete Paper-Stellen"):
                        for citation in result.citations:
                            st.markdown(
                                f"**Seite {citation.page}** · "
                                f"Ähnlichkeit {citation.score:.3f}"
                            )
                            st.write(citation.text)
                except Exception as error:
                    st.error(f"Anfrage fehlgeschlagen: {error}")