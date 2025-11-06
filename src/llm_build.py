import os

import torch
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from ._load_env import DEVICE, console
from .loggers import Logger

logger = Logger.get_logger(__name__)


# ------------------------
# LLM pipeline
# ------------------------
def build_llm_pipe(model_name: str, max_new_tokens: int, temperature: float) -> HuggingFacePipeline:
    """
    Build a HuggingFace LLM pipeline with proper conversation handling.
    """
    console.print(f"Loading LLM: [bold]{model_name}[/bold] on device [bold]{DEVICE}[/bold]")
    logger.info(f"Loading LLM: {model_name} on device {DEVICE}")

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

    def invoke(messages):
        # Convert all messages to a consistent format
        formatted_messages = []

        for msg in messages:
            if hasattr(msg, "type"):  # LangChain message object
                if msg.type == "human":
                    formatted_messages.append({"role": "user", "content": msg.content})
                elif msg.type == "ai":
                    formatted_messages.append({"role": "assistant", "content": msg.content})
                elif msg.type == "system":
                    formatted_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, dict):  # Dictionary format
                formatted_messages.append(msg)
            else:
                # Fallback - try to extract content
                try:
                    content = getattr(msg, "content", str(msg))
                    role = getattr(msg, "role", "user")
                    formatted_messages.append({"role": role, "content": content})
                except Exception:
                    formatted_messages.append({"role": "user", "content": str(msg)})

        # DEBUG: Print what we're sending to the model
        logger.debug("=== FORMATTED MESSAGES FOR MODEL ===")
        for i, msg in enumerate(formatted_messages):
            logger.debug(f"{i}: {msg['role']}: {msg['content']}")
        logger.debug("=====================================")

        # Apply Mistral chat template
        prompt = tok.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=True)

        # Run generation
        result = gen(prompt, return_full_text=False)[0]["generated_text"]

        return {"role": "assistant", "content": result.strip()}

    class SimpleLLM:
        def __init__(self, invoke_func):
            self.invoke = invoke_func

        def bind(self, **kwargs):
            return self

    return SimpleLLM(invoke)
