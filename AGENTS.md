# AGENTS.md

## Project Overview

This is a medical RAG (Retrieval-Augmented Generation) system using LangChain and LangGraph, built for answering questions about Colon-TC examinations.

## Key Setup Commands

- Install with: `pip install -e .` or `pip install -e ".[dev]"`
- Run CLI: `agentic_rag`
- Run Gradio UI: `sbatch rag_gradio.sbatch`
- Build FAISS index: Set `reindex: true` in config.yaml

## Configuration

The system uses:
- `config.yaml` for core settings
- `.env` for API tokens (HF_TOKEN required)
- Uses BAAI embeddings, Mistral-Small-24B LLM, and BAAI reranker by default
- Supports both CPU and GPU execution

## Execution Methods

1. **Interactive CLI**: Set `chat: true` in config.yaml
2. **Gradio Web UI**: Set `gradio: true` in config.yaml and run with `sbatch rag_gradio.sbatch`
3. **Direct script**: `PYTHONPATH=src python -m agentic_rag`

## Environment & Dependencies

- Python 3.12+ required
- HuggingFace token required for model downloads
- FAISS index built automatically from Markdown files in `./Markdown_IT`
- GPU support requires `faiss-gpu` package or SLURM allocation with GPUs

## Testing

Run: `pytest tests/ -v`

## Deployment

The system is designed to work in SLURM environments:
- Use `salloc` for allocation before running
- Requires appropriate module loading (conda, python 3.12)
- For production runs, use SLURM batch scripts

## Special Considerations

- The system uses FAISS vector search with optional reranking
- Supports multi-language queries with language detection
- Implements guardrails and topic continuity analysis
- Gradio UI requires network port setup and SSH tunneling
- Index re-building with `reindex: true` may take significant time
- Large models require GPU allocation from SLURM
