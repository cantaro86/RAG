import os

import torch
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from ._load_env import DEVICE, console


# ------------------------
# LLM pipeline
# ------------------------
def build_llm_pipe(model_name: str, max_new_tokens: int, temperature: float) -> HuggingFacePipeline:
    console.print(f"Loading LLM: [bold]{model_name}[/bold] on device [bold]{DEVICE}[/bold]")

    tok = AutoTokenizer.from_pretrained(model_name, token=os.environ.get("HF_TOKEN"))

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=os.environ.get("HF_TOKEN"),
        device_map="auto",
        torch_dtype=torch.float16 if DEVICE in ("cuda", "mps") else torch.float32,
    )

    if DEVICE == "mps":
        model.to("mps")

    gen = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tok,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        pad_token_id=tok.eos_token_id,
    )
    return HuggingFacePipeline(pipeline=gen)
