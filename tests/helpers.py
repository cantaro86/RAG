# TEST Helper functions

from pathlib import Path

import yaml


def write_yaml(path: Path, data: dict) -> None:
    """Write a dictionary to a YAML file."""
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def make_valid_config() -> dict:
    """Return a minimal valid configuration dictionary for tests."""
    return {
        "log_level": "DEBUG",
        "log_to_file": True,
        "log_file": "agent_rag.log",
        "debugger": False,
        "online": True,
        "md_dir": "./Markdown_IT",
        "dizionario_path": "./dizionario.xlsx",
        "index_dir": "./faiss_index",
        "reindex": False,
        "workers": 8,
        "use_gpu_index": False,
        "quantization": False,
        "embed_model": "BAAI/bge-m3",
        "llm_model": "mistralai/Mistral-Small-24B-Instruct-2501",
        "rerank_model": "BAAI/bge-reranker-v2-m3",
        "rerank": True,
        "k": 32,
        "k_reranked": 6,
        "search_type": "similarity",
        "fetch_k": 120,
        "lambda_mult": 0.3,
        "threshold": 0.1,
        "clean_answer": False,
        "chunk_size": 1200,
        "chunk_overlap": 300,
        "min_chunk_length": 100,
        "max_new_tokens": 1024,
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 40,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "hf_home": "/fast_disk/models/huggingface",
        "chat": True,
        "gradio": False,
        "gradio_host": "0.0.0.0",
        "gradio_port": 7860,
        "gradio_share": False,
        "max_history_turns": 6,
    }
