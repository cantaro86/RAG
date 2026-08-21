# RAGAS evaluation

`questions.txt` contains one independent Italian question per nonblank line. Duplicate questions are rejected.

This directory is a repository-only Python package for evaluation tooling and data. Setuptools discovers packages only under `src/`, so `evaluation` is not included in the distributable `agentic_rag` wheel.

The workflow has four explicit commands:

```bash
# Safe on a login node: validates files and does not load models.
uv run python -m evaluation validate

# GPU node: runs the production graph once per question and records every checkpoint.
uv run python -m evaluation collect --limit 1

# GPU node: loads the independent judge and scores an existing run.
uv run --extra evaluation python -m evaluation score evaluation/results/<run-id> --limit 1

# Login or compute node: creates a formatted workbook without loading models.
uv run python -m evaluation export-excel evaluation/results/<run-id>
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

The `collect` and `score` commands automatically select MPS when available, otherwise CUDA, and stop before model construction when neither accelerator is available. Collection and scoring are separate processes so the agentic_rag LLM is released before the judge LLM is loaded.

RAGAS usage telemetry and LangSmith tracing are disabled by default. Run artifacts use owner-only permissions. Model downloads still follow the `online` flags in `evaluation/config.yaml` and the application-wide Hugging Face offline policy.

Submit the complete GPU workflow through Slurm with:

```bash
sbatch ragas_evaluation.sbatch
```

## Slurm run controls

`LIMIT` is an optional positive integer that restricts collection and scoring to the first N nonblank questions in `questions.txt`. If it is unset, the batch job processes every question. The validation phase always checks the complete questions file, even when a limit is set.

`RUN_ID` names the result directory under `evaluation/results/`. The batch script defaults it to `slurm-<job-id>`, which associates the artifacts with the Slurm job that created them. A custom ID may contain letters, numbers, dots, underscores, and hyphens, and must start with a letter or number. It is only a label: it does not change the configuration or resume an earlier run. Collection refuses to use an existing result directory, so every run ID must be unique.

A **smoke run** is a deliberately small end-to-end run used to verify that validation, GPU allocation, model loading, graph execution, artifact writing, and RAGAS scoring all work before spending time on the complete dataset. It is not large enough to provide representative aggregate metrics. For example, this command runs the first question through the entire workflow and stores it in `evaluation/results/smoke/`:

```bash
LIMIT=1 RUN_ID=smoke sbatch ragas_evaluation.sbatch
```

Use a new ID when repeating a smoke run, for example:

```bash
LIMIT=3 RUN_ID="smoke-$(date -u +%Y%m%dT%H%M%SZ)" sbatch ragas_evaluation.sbatch
```

Omit `LIMIT` for the complete dataset:

```bash
sbatch ragas_evaluation.sbatch
```

The batch script uses unbuffered Python output. Phase and model-loading messages are written to
`ragas-<job-id>.out`; collection and per-metric `tqdm` progress, including elapsed time and ETA, are written to
`ragas-<job-id>.err`. Progress identifies questions by stable ID without printing their text.

## Artifacts

Each collection creates a new directory under `evaluation/results/`. Collection never overwrites an existing run directory; scoring subsequently adds its files to the collected run:

```text
<run-id>/
├── run.json
├── samples.jsonl
├── summary.json
├── scores.jsonl
├── evaluation_results.xlsx
└── traces/
    └── <question-id>.json.gz
```

`scores.jsonl` appears only after scoring starts, and the Slurm workflow creates `evaluation_results.xlsx` after scoring completes. `summary.json` is written after collection finishes and is extended after scoring finishes. Consequently, these files may be absent or incomplete while a job is running or after an abrupt interruption.

### `run.json`

This is the run manifest and the first file to inspect. It identifies the run, source questions, application configuration, generator model, Git revision, timestamps, and number of selected questions. The judge model is added when scoring starts. Its `status` records the latest phase transition:

| Status | Meaning |
| --- | --- |
| `collecting` | Collection started but has not completed. |
| `collected` | All selected questions were attempted and collection artifacts were finalized. |
| `scoring` | The judge started processing the collected samples. |
| `scored` | Scoring finished for every available sample. |
| `scored_partial` | A manual `score --limit N` scored fewer samples than the collected run contains. |
| `collection_failed` | Collection raised an error; partial traces and samples may still be available. |
| `scoring_failed` | Scoring raised an error; partial score records may still be available. |

If Slurm terminates the process without allowing error handling to run, the status can remain `collecting` or `scoring`. In that case, compare the artifact counts and the Slurm logs rather than treating the manifest status as proof that no work was completed.

For a quick overview after scoring, inspect the formatted manifests and count the incremental records. The examples below use `jq` for JSON formatting:

```bash
RUN_DIR=evaluation/results/<run-id>
jq . "$RUN_DIR/run.json"
jq . "$RUN_DIR/summary.json"
wc -l "$RUN_DIR/samples.jsonl" "$RUN_DIR/scores.jsonl"
```

### `samples.jsonl`

This is the input dataset for RAGAS. JSONL means that each line is one complete JSON object, allowing records to be appended and preserved incrementally. A record contains:

| Field | Meaning |
| --- | --- |
| `id` | Stable question ID derived from its source line and a text hash, such as `q0001-abcd1234`. |
| `user_input` | Original question from `questions.txt`. |
| `response` | Final answer returned by the production graph. |
| `retrieved_contexts` | Text of the final accepted documents supplied for scoring. |
| `trace_path` | Relative path to the corresponding compressed trace. |

Only questions that produce a valid final response and document list become samples. A failed question still has a trace but has no line in `samples.jsonl`. The judge reads this file and does not receive internal graph routing data.

### `scores.jsonl`

This file also has one JSON object per line, keyed by the same question `id` used in `samples.jsonl`. Each record contains `metrics`, `errors`, and `skipped` mappings. A successful metric has a numeric value in `metrics`. A failed metric has `null` there and a structured exception in `errors`. Context-dependent metrics are instead recorded in `skipped` when no retrieval context is available. Join samples, scores, and traces by question ID rather than relying on line counts, especially after an interrupted or limited run.

Existing scores are never replaced implicitly. Pass `--overwrite` to `score` only when a deliberate rescore is required.

### `summary.json`

The `collection` section reports selected, completed, and failed question counts; scoreable sample count; successful retrievals; reformulated questions; and total retrieval attempts. After scoring, the `scoring` section adds the number of processed and available samples, model and device information, and a per-metric mean with `scored`, `errors`, and `skipped` counts. Each mean uses only successfully scored values for that metric, so read it together with those counts.

### `traces/<question-id>.json.gz`

There is one gzip-compressed JSON trace for every attempted question, including failed invocations. The trace records every available LangGraph checkpoint in chronological order. `Document` content and metadata are stored once in the top-level `documents` dictionary; checkpoint states and retrieval diagnostics refer to them by stable SHA-256 document IDs. Partial checkpoints are retained when graph invocation fails.

With the current independent-question route, the graph records the raw input, the sanitized question, and at most one `transform_query` rewrite before the second failed retrieval ends the run. The trace format supports additional rewrite iterations if that routing limit changes later.

## Reading a trace

Start with the derived `diagnostics` rather than the full checkpoint states. The following command shows the outcome, any top-level error, executed path, question rewrites, retrieval attempts, and failed nodes:

```bash
TRACE=evaluation/results/<run-id>/traces/<question-id>.json.gz
gzip -cd "$TRACE" | jq '{
  question_id,
  raw_question,
  status,
  error,
  checkpoint_error,
  node_path: .diagnostics.node_path,
  question_versions: .diagnostics.question_versions,
  retrieval_attempts: .diagnostics.retrieval_attempts,
  failed_nodes: .diagnostics.failed_nodes
}'
```

Read the diagnostic fields as follows:

| Field | How to interpret it |
| --- | --- |
| `node_path` | Graph nodes executed in chronological order. Repeated retrieval nodes indicate retries. |
| `question_versions` | Raw, sanitized, and rewritten question text, with the checkpoint sequence that introduced each change. |
| `retrieval_attempts` | Query, attempt number, rewrite count, source filter, `has_docs` result, and document IDs for each `retrieve_and_filter` execution. |
| `reformulation_count` | Number of distinct question changes attributed to sanitization or rewrite nodes. |
| `maximum_rewrite_count` | Highest retry counter observed in the graph state. |
| `retrieval_succeeded` | Whether any retrieval attempt produced accepted documents. |
| `final_has_docs` | Value of `has_docs` in the final available checkpoint. |
| `guardrail_status` | Final recorded guardrail decision, when present. |
| `terminal_node` | Last node represented in the checkpoint history. |
| `next_nodes` | Nodes still scheduled after the final available checkpoint; useful when diagnosing an incomplete run. |
| `failed_nodes` | Node names whose serialized LangGraph tasks contain errors. |

`status: completed` means graph invocation and trace serialization succeeded; it does not guarantee a successful retrieval or a high-quality answer. For `status: failed`, inspect `error`, `checkpoint_error`, and `cleanup_error`, then use `failed_nodes` and the last checkpoints to locate the failure.

For checkpoint-level detail, inspect `steps`:

```bash
gzip -cd "$TRACE" | jq '.steps[] | {
  sequence,
  checkpoint_step,
  created_at,
  nodes,
  next_nodes,
  question: .state.question,
  has_docs: .state.has_docs,
  rewrite_count: .state.rewrite_count,
  generation: .state.generation,
  tasks
}'
```

`sequence` is the trace's zero-based chronological order. `checkpoint_step` is LangGraph's own step number. `nodes` identifies the node or nodes whose execution produced the checkpoint, `next_nodes` shows what was scheduled next, `state` is the complete serialized graph state at that point, and `tasks` captures task names, results, and errors. The initial input checkpoint is represented by `__input__`.

A document reference in a state looks like `{"$document":"doc-<sha256>"}`. Resolve retrieval document IDs against the top-level dictionary with:

```bash
gzip -cd "$TRACE" | jq '
  . as $trace
  | $trace.diagnostics.retrieval_attempts[]
  | . as $attempt
  | $attempt.documents[]
  | . as $document_id
  | {
      attempt: $attempt.attempt,
      document_id: $document_id,
      document: $trace.documents[$document_id]
    }'
```

The resolved `document` contains `page_content`, `metadata`, and an optional original document `id`. This indirection avoids copying the same retrieved content into every checkpoint.

## Excel report

Create a presentation-ready workbook after collection or scoring with:

```bash
uv run python -m evaluation export-excel evaluation/results/<run-id>
```

The default output is `<run-dir>/evaluation_results.xlsx`. Use `--output` to choose another `.xlsx` path:

```bash
uv run python -m evaluation export-excel evaluation/results/<run-id> \
  --output evaluation/results/<run-id>/my_report.xlsx
```

The command refuses to replace an existing workbook unless `--overwrite` is passed. It does not load the application or judge models, so it is safe to run on a login node. It can also export a partially completed run: attempted questions remain visible, while unavailable responses and metrics are left blank and identified by the status and error columns.

The workbook contains three sheets:

| Sheet | Contents |
| --- | --- |
| `Results` | One row per attempted question, including failed questions that never became RAGAS samples. |
| `Run Summary` | Flattened `run.json` and `summary.json` metadata, counts, models, and aggregate metric results. |
| `Legend` | Description of every result column and the mapping from internal guardrail values to readable classifications. |

The `Results` sheet joins traces, samples, and scores by stable question ID. Its fields cover the original, sanitized, transformed, and final questions; readable guardrail classification; final response; retrieval counts, sources, reranker scores, and contexts; all three RAGAS metrics; and reasons for skipped metrics. `transformed_question` contains the last `transform_query` result when that node ran.

Long text is wrapped, the header row is frozen, filters are enabled, classifications and statuses are color-coded, and metric columns use a red-to-green scale. Because Excel limits a cell to 32,767 characters, exceptionally large combined contexts end with an explicit truncation notice.

Generated results are ignored by Git. Preserve selected baseline artifacts elsewhere if they need to be versioned.

## Metrics

The question-only dataset supports `faithfulness`, `context_utilization`, and `answer_relevancy`. `answer_relevancy` uses an independent multilingual sentence-transformer instead of the BGE-M3 retrieval model.

Reference-based answer correctness and context recall require independently curated answers or reference context annotations. The presence of an answer somewhere in `Markdown_IT` is not a ground-truth annotation.

## Dependency note

RAGAS 0.4.3 imports removed VertexAI modules with LangChain 1.x. The optional dependency is temporarily pinned to the immutable commit from upstream pull request 2956, which only makes those unrelated VertexAI imports optional. Replace the commit dependency with a released RAGAS version after the upstream fix is published.
