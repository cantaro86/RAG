# RAGAS evaluation

`questions.txt` contains one independent Italian question per nonblank line. Duplicate questions are rejected.

This directory is a repository-only Python package for evaluation tooling and data. Setuptools discovers packages only under `src/`, so `evaluation` is not included in the distributable `agentic_rag` wheel.

The workflow has three explicit commands:

```bash
# Safe on a login node: validates files and does not load models.
uv run python -m evaluation validate

# GPU node: runs the production graph once per question and records every checkpoint.
uv run python -m evaluation collect --limit 1

# GPU node: loads the independent judge and scores an existing run.
uv run --extra evaluation python -m evaluation score evaluation/results/<run-id> --limit 1
```

Install the optional scoring dependencies with:

```bash
uv sync --locked --extra dev --extra evaluation
```

Run the repository-only evaluation tests explicitly with:

```bash
uv run --extra evaluation pytest -m evaluation tests/test_evaluation.py
```

These tests use mocked models and are intentionally excluded from the CI `cpu` selection.

The `collect` and `score` commands automatically select MPS when available, otherwise CUDA, and stop before model construction when neither accelerator is available. Collection and scoring are separate processes so the Mistral generator is released before the Qwen judge is loaded.

RAGAS usage telemetry and LangSmith tracing are disabled by default. Run artifacts use owner-only permissions. Model downloads still follow the `online` flags in `evaluation/config.yaml` and the application-wide Hugging Face offline policy.

Submit the complete GPU workflow through Slurm with:

```bash
sbatch ragas_evaluation.sbatch
```

Set `LIMIT` for a smoke run or `RUN_ID` for a custom result directory:

```bash
LIMIT=1 RUN_ID=smoke sbatch ragas_evaluation.sbatch
```

## Artifacts

Each collection creates an immutable directory under `evaluation/results/`:

```text
<run-id>/
├── run.json
├── samples.jsonl
├── scores.jsonl
├── summary.json
└── traces/
    └── <question-id>.json.gz
```

The trace contains every chronological LangGraph checkpoint. `Document` content and metadata are stored once in a top-level dictionary and graph states refer to them by stable SHA-256 identifiers. The trace also includes the executed node path, every distinct question version, all retrieval attempts, routing status, and partial checkpoints from failed invocations.

With the current independent-question route, the graph records the raw input, the sanitized question, and at most one `transform_query` rewrite before the second failed retrieval ends the run. The trace format supports additional rewrite iterations if that routing limit changes later.

`samples.jsonl` contains the raw question, final response, accepted retrieval contexts, and trace path. The RAGAS judge reads this file and does not receive internal graph routing data. Existing scores are never replaced implicitly; pass `--overwrite` to `score` when a deliberate rescore is required. A limited score run is marked `scored_partial`.

Generated results are ignored by Git. Preserve selected baseline artifacts elsewhere if they need to be versioned.

## Metrics

The question-only dataset supports `faithfulness`, `context_utilization`, and `answer_relevancy`. `answer_relevancy` uses an independent multilingual sentence-transformer instead of the BGE-M3 retrieval model.

Reference-based answer correctness and context recall require independently curated answers or reference context annotations. The presence of an answer somewhere in `Markdown_IT` is not a ground-truth annotation.

## Dependency note

RAGAS 0.4.3 imports removed VertexAI modules with LangChain 1.x. The optional dependency is temporarily pinned to the immutable commit from upstream pull request 2956, which only makes those unrelated VertexAI imports optional. Replace the commit dependency with a released RAGAS version after the upstream fix is published.
