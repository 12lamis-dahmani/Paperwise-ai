# Evaluation

Create a JSONL file for your sample paper. Each line needs a question and the page or pages where the answer is located:

```json
{"question": "Which method do the authors use?", "expected_pages": [4]}
```

Run the retrieval baseline with Ollama available:

```powershell
python evaluation/baseline.py data/samples/paper.pdf evaluation/questions.jsonl
```

The script reports `Recall@5`: the percentage of questions for which at least one expected page appears in the retrieved top five chunks. Save the result before changing ingestion or chunking so later improvements can be compared.

Review:

- Is the answer supported by the cited pages?
- Does the assistant say when evidence is missing?
- Are the retrieved pages relevant?

Automated retrieval and answer metrics will be added in a later version.