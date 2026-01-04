# Ragas: Retrieval Augmented Generation Assessment

## Overview
Ragas is a framework for evaluating Retrieval Augmented Generation (RAG) pipelines. It provides metrics to grade the quality of your RAG application without needing human labels for every query.

## Core Metrics

### 1. Faithfulness
- **Question**: "Is the answer derived purely from the retrieved context?"
- **Goal**: Detect hallucinations. If the AI makes up facts not in the docs, Faithfulness is low.
- **Calculation**: It compares the `generated_answer` against the `retrieved_context`.

### 2. Answer Relevance
- **Question**: "Does the answer actually address the user's query?"
- **Goal**: Prevent vague or evasive answers.
- **Calculation**: It compares the `generated_answer` against the original `user_question`.

### 3. Context Precision
- **Question**: "Did the retrieval system find the relevant chunks?"
- **Goal**: Evaluate the search engine (Vector Store).
- **Calculation**: Checks if the "Gold Standard" answer is present in the top-k retrieved docs.

## Integration with LangSmith
Ragas integrates natively with LangSmith. When you run an evaluation:
1. Ragas computes the scores locally.
2. It pushes the detailed trace and scores to LangSmith.
3. You can view a dashboard showing "Average Faithfulness" over time.

## How to use
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

results = evaluate(
    dataset=my_dataset,
    metrics=[faithfulness, answer_relevancy]
)
print(results)
```
