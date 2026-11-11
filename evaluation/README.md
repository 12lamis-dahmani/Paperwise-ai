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

## Answer evaluation

Run the answer baseline with Ollama available:

```powershell
python evaluation/answer_baseline.py data/samples/paper.pdf evaluation/questions.jsonl
```

The script evaluates all questions in the JSONL file:

- `answerable: true`: the answer must cite at least one expected page.
- `answerable: false`: the answer must use the configured abstention sentence.
- `expected_keywords` (optional): every listed fact must occur in the answer.

The output reports retrieval hit rate, supported-answer accuracy, and correct
rejection of unsupported questions. This is still a lightweight regression
check, not a complete semantic judge.