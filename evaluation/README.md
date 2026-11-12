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

Each answerable JSONL case must contain `expected_facts`. It is a list of required fact groups; each group is a list of equivalent accepted spellings, and at least one variant in every group must occur in the answer:

```json
{"question":"Which accuracy did the network achieve?","expected_pages":[4],"expected_facts":[["85.4%","85,4%"],["classification accuracy"]],"answerable":true}
{"question":"Which GPU did the authors use?","expected_pages":[],"expected_facts":[],"answerable":false}
```

The script evaluates all questions in the JSONL file and reports these metrics independently:

- **Retrieval Hit@K:** fraction of answerable questions where a retrieved chunk (`answer.citations` from the pipeline) comes from an expected page.
- **Citation Hit Rate:** fraction of answerable questions where the generated answer explicitly contains `[Seite X]` for an expected page. Retrieved pages are not treated as LLM citations.
- **Fact Match Rate:** matched required fact groups divided by all required fact groups across answerable questions. Any listed spelling variant satisfies its group.
- **Abstention Accuracy:** fraction of unanswerable questions answered with the configured abstention sentence.

No combined answer-accuracy score is reported. Each run writes a timestamped JSON report to `evaluation/results/`; pass `--output path/to/report.json` to choose another path. The report records model and chunk settings alongside per-question retrieved pages, explicit citations, and each individual fact match, so runs can be compared later.

These are lightweight regression metrics, not a semantic judge. A fact match only confirms that an expected string appears in the answer; it does not prove that the answer is fully correct, that the fact is used in the right context, or that the cited passage actually supports the claim. Review generated answers and cited passages manually, especially for numerical or multi-part questions.