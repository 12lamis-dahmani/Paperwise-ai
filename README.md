# Paperwise AI

Paperwise AI is a local research-paper assistant. Version 0.1 accepts one PDF, extracts its text, creates chunks and embeddings, retrieves relevant pages, and asks a local Ollama model to answer with sources.

## Requirements

- Python 3.11 or newer
- [Ollama](https://ollama.com/) running locally
- At least one chat model and one embedding model downloaded in Ollama

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

ollama pull qwen3:4b
ollama pull embeddinggemma
streamlit run frontend/app.py
```

Open the URL printed by Streamlit, upload a scientific PDF, and ask a question about it.

The default models can be changed in the sidebar. Ollama must be reachable at `http://localhost:11434`; this is the default Ollama address.

## Current scope: v0.1

- One PDF at a time
- Text extraction with page numbers
- In-memory chunking and cosine-similarity retrieval
- Local Ollama chat and embedding models
- Answers grounded in retrieved excerpts
- Page references in the answer and expandable evidence blocks

This version does not yet reliably understand figures, tables, scanned PDFs, or mathematical notation. It also does not persist documents between application restarts.

## Evaluation: v0.2

The evaluation now reports retrieval hit rate, explicit citation hit rate, expected-fact match rate, and abstention accuracy separately. It flags citations to pages that were not retrieved and saves each run as a JSON report with model and chunking settings. Fact matching is based on configured text variants, so it is a regression check rather than a semantic correctness judge. See [evaluation/README.md](evaluation/README.md) for usage and dataset format.

## Start the API

```powershell
uvicorn paperwise.api.main:app --reload
```

## Project structure

```text
src/paperwise/      Application package split by responsibility
frontend/app.py     Streamlit UI
tests/              Focused unit tests
```
