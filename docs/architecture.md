# Architecture

```text
PDF -> pdf_loader -> chunker -> Ollama embeddings -> in-memory vector store
                                                    |
Question -> query embedding -> retriever -> prompt -> Ollama chat model
```

The v0.1 pipeline is intentionally local and stateless. The vector store lives in process memory and is replaced when a new PDF is indexed. Page numbers remain attached to every chunk so the UI and API can expose evidence.