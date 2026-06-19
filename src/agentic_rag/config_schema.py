from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class Config(BaseModel):
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    log_to_file: bool
    log_file: str

    debugger: bool
    online: bool

    md_dir: str
    dizionario_path: str
    index_dir: str

    reindex: bool
    workers: int = Field(ge=1)

    use_gpu_index: bool
    quantization: bool

    embed_model: str
    llm_model: str
    rerank_model: str

    rerank: bool
    k: int = Field(gt=0)
    k_reranked: int = Field(gt=0)
    search_type: Literal["similarity", "mmr"]
    fetch_k: int = Field(gt=0)
    lambda_mult: float = Field(ge=0.0, le=1.0)
    threshold: float

    clean_answer: bool

    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    min_chunk_length: int = Field(gt=0)

    max_new_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0.0, le=2.0)
    top_p: float = Field(gt=0.0, le=1.0)
    top_k: int = Field(gt=0)
    repetition_penalty: float = Field(ge=1.0)
    no_repeat_ngram_size: int = Field(ge=0)

    hf_home: str

    chat: bool
    gradio: bool
    gradio_host: str
    gradio_port: int = Field(gt=0, le=65535)
    gradio_share: bool
    max_history_turns: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_cross_fields(self):
        if self.chat and self.gradio:
            raise ValueError("'chat' and 'gradio' cannot both be true")
        if self.k_reranked > self.k:
            raise ValueError("'k_reranked' must be <= 'k'")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("'chunk_overlap' must be < 'chunk_size'")
        return self


def load_config(path: Path) -> Config:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Configuration file is empty: {path}")

    return Config.model_validate(data)
