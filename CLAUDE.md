# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`AGENTS.md` is the source of truth for setup, commands (tests, single tests, coverage, pre-commit, packaging, GPU runs), config rules, and indexing gotchas. It is imported here rather than repeated:

@AGENTS.md

## Additional Commands

- Evaluation lives in the repo-only `evaluation/` package, which is not shipped in the wheel. `uv run python -m evaluation validate` and `export-excel <run-dir>` load no models; `collect` and `score` require CUDA or MPS. Artifact schema and Slurm usage are in `evaluation/README.md`.
- `uv run python scripts/print_graph.py` re-renders `graph.png`, which is embedded in the README, from the graph topology using mocked dependencies. It loads no models but calls the Mermaid web API.

## Architecture

A LangGraph agent answers Italian questions about CT colonography (Colon-TC) from a FAISS index built over `Markdown_IT/`.

### Import-Time Configuration

`agentic_rag._load_env` runs at import time. It resolves and validates the config, loads `.env`, sets `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `HF_HOME`, and `HF_HUB_CACHE`, rewrites `cfg.hf_home` to the effective path, and picks `DEVICE` in the order `mps`, `cuda`, `cpu`. Its module-level `cfg` is a process singleton that `loggers.Logger` and `build_faiss._SPLITTER` also read at import time.

- Importing any `agentic_rag` module, including from tests, requires a valid config reachable from the working directory or through `AGENTIC_RAG_CONFIG`.
- `cli.py`, `llm_build.py`, and `ui_gradio.py` import `_load_env` before the heavy libraries and suppress Ruff import sorting (`I001`). Keep that ordering.

### Request Path

- `cli.interactive_loop` and `ui_gradio._response_callback` both reject non-Italian input with `DetectLanguage` before invoking the graph. They then stream the compiled graph and display the last node's `generation`.
- `agent_factory.build_rag_agent` is the single builder for both front ends. It loads the vector store, the retriever, and exactly one LLM, then derives every chain by calling `.bind()` with different generation settings. The classifiers use greedy decoding with `max_new_tokens=5`, while the rewriters sample at low temperature.
- `llm_build.SimpleLLM` is a minimal bind/invoke adapter, not a LangChain chat model. It applies the tokenizer chat template and dispatches either to a transformers pipeline, optionally with bitsandbytes 4-bit quantization on CUDA, or, when `quantization: true` on MPS, to `mlx-lm` with a prequantized MLX model.

### Graph (`graph.py`)

Nodes are plain functions bound with `functools.partial` to the dependencies held in `RAGContext`. Each node returns a full copy of `GraphState` (`state.py`).

```text
sanitize_question -> social_intent
social_intent:        SALUTO -> handle_hello -> END | GRAZIE -> handle_thanks -> END | DOMANDA -> init_first_question
init_first_question:  first turn -> domain_guardrail | follow-up -> topic_detector
topic_detector:       same topic -> pre_retrieval_rewriter -> domain_guardrail | new topic -> domain_guardrail
domain_guardrail:     OFF_TOPIC -> handle_off_topic -> END | new topic -> clear_history -> retrieve_and_filter | ON_TOPIC -> retrieve_and_filter
retrieve_and_filter:  docs -> generate_with_docs -> clean_answer -> END | 1st miss -> transform_query -> retrieve_and_filter | 2nd miss -> no_generation -> END
```

- Classifier outputs are normalized, and unexpected values fall back to `DOMANDA`, `ON_TOPIC`, or the same topic. Tests assert these fallbacks.
- The greeting, thanks, off-topic, and no-document replies are hardcoded Italian strings in the handler nodes. All prompts are in `agent_prompts.py`.
- Only `clean_answer` appends to `history`, so canned replies never enter it. It stores the sanitized question and the final answer, capped at `max_history_turns` pairs. The LLM cleaner runs only when `clean_answer: true` and `_META_PATTERNS` detects meta-commentary.
- `transform_query` adds synonyms from `dizionario.xlsx` through `dizionario.SynonymStore`, which reloads whenever the workbook's mtime changes.
- The CLI always uses `thread_id="default"`. Gradio creates a UUID thread per session and replaces it whenever the visible chat diverges from the history it last returned, deleting the old checkpoint and reseeding `history` from the UI.
- `evaluation/tracing.py` and `evaluation/export_excel.py` hard-code node names: `clean_answer`, `no_generation`, the `handle_*` nodes, `retrieve_and_filter`, `transform_query`, `sanitize_question`, and `pre_retrieval_rewriter`. Update them when renaming nodes or adding terminal nodes.

### Retrieval (`retriever.py`, `utils.py`)

- `utils.extract_source_filter` turns phrases such as "informazioni per pazienti" into a `{"source": "Informazioni_per_pazienti.md"}` metadata filter. `sanitize_question` captures the filter, and it persists across query rewrites and retries.
- With reranking enabled, `FilterableRerankerRetriever` runs a FAISS similarity or MMR search. A cross-encoder then writes `rerank_score` into copies of the retrieved documents. The FAISS docstore is shared across requests, so never mutate retrieved documents in place.
- If a source filter is active and the query names "tabella N" or "table N", the retriever skips FAISS and looks up matching `type: table` chunks directly.
- Before the RAG prompt, `render_context` puts the patient document and recommendation sections first and labels each excerpt with its corpus.

### Indexing (`build_faiss.py`)

Chunking follows the Markdown structure instead of splitting plain text:

- Reference and bibliography sections are dropped.
- Each recommendation section becomes a single `recomm` chunk, whatever its size.
- Tables are kept whole.
- Lists stay attached to their introductory line.
- Numbered lists split across paragraphs are re-merged by `merge_continuation_lists`.

Every chunk carries `source`, `section`, `language`, and `type` metadata, which the retriever and `render_context` depend on.

### Tests

- CPU tests never load models. For graph tests, the `tests/conftest.py::mock_ctx` fixture supplies `MagicMock` chains and a mock retriever.
- The `gpu_guardrail_llm` fixture loads the configured LLM once per session for the `gpu` prompt-behavior tests. Those tests are skipped when neither CUDA nor MPS is available.
- Every test module sets `pytestmark` to `cpu`, `gpu`, or `evaluation`. A new module needs one of these markers to be selected, and pre-commit requires the `test_*.py` filename pattern.
- `conftest.py` disables logging globally. Use the `enabled_test_logging` fixture to assert on log output.
