import os

import torch
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from ._load_env import DEVICE, ONLINE, cfg, console
from .loggers import Logger

logger = Logger.get_logger(__name__)


# Import based on device availability
if DEVICE == "cuda":
    from transformers import BitsAndBytesConfig
elif DEVICE == "mps":
    try:
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        MLX_AVAILABLE = True
    except ImportError:
        MLX_AVAILABLE = False


# ------------------------
# LLM pipeline
# ------------------------
def build_llm_pipe(
    model_name: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    quantization: bool = False,
) -> HuggingFacePipeline:
    """
    Build a HuggingFace LLM pipeline with proper conversation handling.
    Supports quantization via:
    - BitsAndBytes (CUDA)
    - MLX (Apple Silicon/MPS)
    - Standard loading (no quantization)
    """
    offline = not (ONLINE and getattr(cfg, "online", True))

    console.print(f"Loading LLM: [bold]{model_name}[/bold] on device [bold]{DEVICE}[/bold]")
    logger.info(f"Loading LLM: {model_name} on device {DEVICE}")
    console.print("Mode: " + ("online" if not offline else "offline"))

    tok = AutoTokenizer.from_pretrained(
        model_name,
        token=os.environ.get("HF_TOKEN"),
        local_files_only=offline,
    )

    # === OPTION 1: CUDA with BitsAndBytes quantization ===
    if quantization and DEVICE == "cuda":
        logger.info("Using BitsAndBytes 4-bit quantization (CUDA)")
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=os.environ.get("HF_TOKEN"),
            device_map="auto",
            quantization_config=quantization_config,
            local_files_only=offline,
        )
        use_mlx = False

    # === OPTION 2: MPS with MLX quantization ===
    elif quantization and DEVICE == "mps":
        if MLX_AVAILABLE:
            logger.info("Using MLX quantization (Apple Silicon)")
            # Load model with MLX (supports quantization natively)
            model, tokenizer = load(model_name)
            tok = tokenizer  # Use MLX's tokenizer
            use_mlx = True
        else:
            logger.warning("⚠️ MLX not available. Install with: pip install mlx mlx-lm")
            logger.info("Falling back to torch.float16 on MPS")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=os.environ.get("HF_TOKEN"),
                device_map="auto",
                dtype=torch.float16,
                low_cpu_mem_usage=True,
                local_files_only=offline,
            )
            model.to("mps")
            use_mlx = False

    # === OPTION 3: Standard loading (no quantization) ===
    else:
        logger.info("Loading model without quantization")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=os.environ.get("HF_TOKEN"),
            device_map="auto",
            dtype=torch.float16 if DEVICE in ("cuda", "mps") else torch.float32,  # before it was torch_dtype
            local_files_only=offline,
        )

        if DEVICE == "mps":
            model.to("mps")
        use_mlx = False

    # === Build pipeline based on backend ===
    if use_mlx:
        # MLX-based generation wrapper
        def mlx_generate(prompt, **kwargs):
            # Create sampler with temperature and top_p
            sampler = make_sampler(temp=temperature, top_p=top_p)

            # Create logits processors for repetition penalty
            logits_processors = make_logits_processors(repetition_penalty=repetition_penalty)

            response = generate(
                model,
                tok,  # Using tok as you assigned tokenizer to tok
                prompt=prompt,
                max_tokens=max_new_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
            )
            return [{"generated_text": response}]

        gen = mlx_generate

    else:
        # Standard HuggingFace pipeline
        gen = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tok,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=temperature > 0,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
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

        # Apply Mistral chat template
        prompt = tok.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=True)
        # DEBUG: Print what we're sending to the model
        logger.debug(f"🔍 PROMPT FROM apply_chat_template:\n{prompt}\n")

        # Run generation
        result = gen(prompt, return_full_text=False)[0]["generated_text"]

        return {"role": "assistant", "content": result.strip()}

    class SimpleLLM:
        def __init__(self, invoke_func):
            self.invoke = invoke_func

        def bind(self, **kwargs):
            return self

    return SimpleLLM(invoke)


def load_translator(repo_id: str):
    """
    Load translation model.
    For NLLB models, returns (model, tokenizer) tuple.
    For legacy OPUS-MT models, returns pipeline.
    """
    if ONLINE and getattr(cfg, "online", True):
        logger.info(f"[HF] Online → loading {repo_id}")
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(repo_id)
    else:
        logger.info(f"[HF] Offline → loading {repo_id} from local cache")
        tokenizer = AutoTokenizer.from_pretrained(repo_id, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(repo_id, local_files_only=True)

    # Move to device
    if DEVICE in ["cuda", "mps"]:
        model = model.to(DEVICE)
        logger.info(f"Translation model moved to {DEVICE}")

    return model, tokenizer
