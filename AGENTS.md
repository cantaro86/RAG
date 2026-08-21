# AGENTS.md

## Source Of Truth

- Use `pyproject.toml` and `uv.lock` for dependencies and Python support. They and CI require Python 3.14. `requirements.txt` is a generated `uv export --locked --extra dev --no-hashes` for Linux x86_64 and macOS 14+ ARM64; regenerate it after lockfile changes. Hashes are omitted so pip can install the editable project entry. `environment.yml` is a cross-platform Conda-forge bootstrap for Python and `uv`, followed by `uv sync --locked --extra dev`.
- `src/agentic_rag/_version.py` is generated and ignored by Git; do not edit it. Builds use setuptools-scm, so versioning requires Git tags/history (CI checks out with `fetch-depth: 0`).

## Commands

- Match CI setup with `uv sync --locked --extra dev`.
- PDF/DOCX conversion is optional; install it with `uv sync --locked --extra dev --extra markdown` before using `scripts/build_markdown.py`.
- Run from the repository root with `uv run agentic_rag`. Do not run `python src/agentic_rag/cli.py`; the uninstalled fallback is `PYTHONPATH=src python -m agentic_rag`.
- Run the CI test selection with `uv run pytest -m cpu`. Evaluation tooling tests run separately with `uv run --extra evaluation pytest -m evaluation tests/test_evaluation.py`. Focus package tests with `uv run pytest -m cpu tests/test_guardrail.py` or append a node such as `::test_graph_routing_on_topic`.
- Run all configured checks with `uvx pre-commit run --all-files`. This can rewrite files because Ruff lint runs with `--fix` and Ruff format also runs.
- Verify packaging in order with `uv build`, then `uvx twine check --strict dist/*`.

## Runtime Wiring

- `python -m agentic_rag` and the console script enter `cli.main()`, which optionally rebuilds/loads FAISS, creates the retriever and shared LLM in `agent_factory.py`, then runs either terminal chat or Gradio through `graph.py`.
- Config discovery order is `AGENTIC_RAG_CONFIG`, then `./config.yaml`, then the source-tree root. `_load_env.py` strictly validates YAML before loading the adjacent `.env`; `.env` values do not interpolate YAML.
- All config fields are required and unknown keys fail. When adding a field, update `Config`, `config.yaml`, and `tests/helpers.py::make_valid_config` together. The obsolete `workers` key is rejected.
- `online: false`, `HF_HUB_OFFLINE=1`, or `TRANSFORMERS_OFFLINE=1` prevents remote model access. The connectivity probe is informational; `HF_TOKEN` is needed only when the selected Hub model requires authentication.
- `hf_home` is the Hugging Face root; Hub models use `HF_HUB_CACHE` or `<hf_home>/hub`. FastText is loaded lazily from `<hf_home>/fasttext` and falls back to `langdetect` when unavailable.
- Relative data, index, dictionary, and log paths are resolved from the process working directory, not from the config location; run the application and tests from the repository root.
- Startup validates top-level Markdown, the dictionary workbook, and both `index.faiss` and `index.pkl` before loading expensive models.
- Runtime input is Italian-only despite multilingual helpers and metadata. Conversation checkpoints use `InMemorySaver`, so chat history is process-local and not durable.

## Indexing And Tests

- Normal startup consumes existing Markdown; `scripts/build_markdown.py` conversion is a separate utility. Reindexing scans only top-level `md_dir/*.md`, and startup still requires `md_dir` to exist and be nonempty when loading an existing index.
- `chunk_size`, `chunk_overlap`, and `min_chunk_length` drive new indexes. Existing FAISS data is unchanged until `reindex: true`; rebuild after changing chunk settings.
- `search_type`, `fetch_k`, and `lambda_mult` are forwarded to retrieval. `threshold` applies only to reranker scores; with `rerank: false`, FAISS top-k documents are accepted directly.
- The exact basename `Informazioni_per_pazienti.md` controls patient-corpus filtering and ordering. `tests/test_build_faiss.py` also exercises this tracked file, so parser changes can alter corpus-sensitive assertions.
- FAISS loading enables pickle deserialization; use only index directories created from trusted repository data.
- `rag_gradio.sbatch` loads Python 3.14 and `uv`, selects Gradio with `AGENTIC_RAG_MODE=gradio`, and leaves `config.yaml` unchanged. The Cloudflare tunnel is mandatory, while `GRADIO_SERVER_NAME` and `GRADIO_SERVER_PORT` override YAML for deployment.
