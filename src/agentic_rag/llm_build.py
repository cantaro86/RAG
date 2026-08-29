# ruff: noqa: I001
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import agentic_rag._load_env as _  # noqa: F401
from agentic_rag._load_env import DEVICE, EFFECTIVE_HF_HUB_CACHE, console, hf_online_enabled

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from agentic_rag.loggers import Logger

logger = Logger.get_logger(__name__)

BitsAndBytesConfig = None
if DEVICE == "cuda":
    from transformers import BitsAndBytesConfig

MLX_AVAILABLE = False
MLX_IMPORT_ERROR: ImportError | None = None
generate = None
load = None
make_logits_processors = None
make_sampler = None
if DEVICE == "mps":
    try:
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        MLX_AVAILABLE = True
    except ImportError as exc:
        MLX_IMPORT_ERROR = exc


class BindableLLM(Protocol):
    """Minimal model interface consumed by the agent factory."""

    def bind(self, **kwargs: Any) -> BindableLLM: ...

    def invoke(self, input_data: Any) -> dict[str, str]: ...


class SimpleLLM:
    def __init__(
        self,
        invoke_func: Callable[..., dict[str, str]],
        bound_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.invoke_func = invoke_func
        self.bound_kwargs = bound_kwargs or {}

    def bind(self, **kwargs: Any) -> SimpleLLM:
        return SimpleLLM(self.invoke_func, {**self.bound_kwargs, **kwargs})

    def invoke(self, input_data: Any) -> dict[str, str]:
        return self.invoke_func(input_data, **self.bound_kwargs)


def _is_mistral_model(model_name: str, model_config: dict[str, Any] | None = None) -> bool:
    if "mistral" in model_name.lower():
        return True
    if not model_config:
        return False
    if "mistral" in str(model_config.get("model_type", "")).lower():
        return True
    architectures = model_config.get("architectures", [])
    return isinstance(architectures, list) and any("mistral" in str(name).lower() for name in architectures)


def _load_tokenizer(model_name: str, offline: bool, cache_folder: str):
    tokenizer_kwargs = {
        "token": os.environ.get("HF_TOKEN"),
        "local_files_only": offline,
        "cache_dir": cache_folder,
    }
    if _is_mistral_model(model_name):
        tokenizer_kwargs["fix_mistral_regex"] = True

    try:
        return AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    except TypeError as exc:
        if "fix_mistral_regex" not in tokenizer_kwargs or "fix_mistral_regex" not in str(exc):
            raise
        tokenizer_kwargs.pop("fix_mistral_regex")
        logger.warning("Tokenizer does not support fix_mistral_regex; retrying without it")
        return AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)


def _mlx_model_source(model_name: str, online: bool, cache_folder: str) -> str:
    local_path = Path(model_name).expanduser()
    if local_path.exists() or online:
        return str(local_path) if local_path.exists() else model_name

    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=model_name,
        token=os.environ.get("HF_TOKEN"),
        cache_dir=cache_folder,
        local_files_only=True,
    )


def _load_mlx_model_config(
    model_name: str,
    online: bool,
    cache_folder: str = EFFECTIVE_HF_HUB_CACHE,
) -> dict[str, Any]:
    local_path = Path(model_name).expanduser()
    if local_path.exists():
        if not local_path.is_dir():
            raise ValueError(f"MLX model path must be a directory: {local_path}")
        config_path = local_path / "config.json"
    else:
        from huggingface_hub import hf_hub_download

        config_path = Path(
            hf_hub_download(
                repo_id=model_name,
                filename="config.json",
                token=os.environ.get("HF_TOKEN"),
                cache_dir=cache_folder,
                local_files_only=not online,
            )
        )

    try:
        with config_path.open(encoding="utf-8") as config_file:
            model_config = json.load(config_file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"MLX model config not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"MLX model config is not valid JSON: {config_path}") from exc

    if not isinstance(model_config, dict):
        raise ValueError(f"MLX model config must contain a JSON object: {config_path}")
    return model_config


def _validate_mlx_quantization_config(model_config: dict[str, Any]) -> None:
    if model_config.get("quantization") is not None:
        quantization = model_config["quantization"]
        if not isinstance(quantization, dict):
            raise ValueError("MLX quantization metadata must be an object")

        group_size = quantization.get("group_size")
        bits = quantization.get("bits")
        mode = quantization.get("mode", "affine")
        supported_values = {
            "affine": ({32, 64, 128}, {2, 3, 4, 5, 6, 8}),
            "mxfp4": ({32}, {4}),
            "mxfp8": ({32}, {8}),
            "nvfp4": ({16}, {4}),
        }
        allowed = supported_values.get(mode) if isinstance(mode, str) else None
        if (
            allowed is None
            or not isinstance(group_size, int)
            or isinstance(group_size, bool)
            or not isinstance(bits, int)
            or isinstance(bits, bool)
            or group_size not in allowed[0]
            or bits not in allowed[1]
        ):
            raise ValueError(
                "Unsupported MLX quantization metadata; expected a supported mode/group_size/bits combination"
            )
        return

    legacy_config = model_config.get("quantization_config")
    if isinstance(legacy_config, dict) and legacy_config.get("quant_method") in {
        "bitnet",
        "mxfp4",
        "compressed-tensors",
    }:
        return

    raise ValueError("MPS quantization requires an already-quantized MLX model with supported quantization metadata")


def _require_mlx_no_repeat_support(no_repeat_ngram_size: int) -> None:
    if no_repeat_ngram_size != 0:
        raise NotImplementedError("no_repeat_ngram_size is not supported by the pinned mlx-lm backend; set it to 0")


def build_llm_pipe(
    model_name: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    quantization: bool = False,
    *,
    online: bool | None = None,
    debugger: bool = False,
    cache_folder: str = EFFECTIVE_HF_HUB_CACHE,
) -> BindableLLM:
    """Build the bind/invoke model adapter used by the RAG chains.

    ``online`` is the caller's configured online preference. It is always
    constrained by the process-wide effective Hugging Face mode.
    """
    effective_online = hf_online_enabled(True if online is None else online)
    debugger_enabled = bool(debugger)
    offline = not effective_online

    console.print(f"Loading LLM: [bold]{model_name}[/bold] on device [bold]{DEVICE}[/bold]")
    logger.info("Loading LLM: %s on device %s", model_name, DEVICE)
    console.print("Mode: " + ("online" if effective_online else "offline"))

    use_mlx = bool(quantization and DEVICE == "mps")
    if use_mlx:
        _require_mlx_no_repeat_support(no_repeat_ngram_size)
        if not MLX_AVAILABLE or load is None:
            raise RuntimeError("MPS quantization requires mlx-lm and a prequantized MLX model") from MLX_IMPORT_ERROR

        model_config = _load_mlx_model_config(model_name, effective_online, cache_folder)
        _validate_mlx_quantization_config(model_config)
        model_source = _mlx_model_source(model_name, effective_online, cache_folder)
        mlx_load_kwargs = {}
        if _is_mistral_model(model_name, model_config):
            mlx_load_kwargs["tokenizer_config"] = {"fix_mistral_regex": True}
        model, tok = load(model_source, **mlx_load_kwargs)
        logger.info("Loaded prequantized MLX model for Apple Silicon")
    else:
        tok = _load_tokenizer(model_name, offline, cache_folder)

        if quantization and DEVICE == "cuda":
            if BitsAndBytesConfig is None:
                raise RuntimeError("CUDA quantization requires BitsAndBytes support")
            logger.info("Using BitsAndBytes 4-bit quantization (CUDA)")
            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=os.environ.get("HF_TOKEN"),
                device_map="auto",
                quantization_config=quantization_config,
                local_files_only=offline,
                cache_dir=cache_folder,
            )
        else:
            logger.info("Loading model without quantization")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=os.environ.get("HF_TOKEN"),
                device_map="auto",
                dtype=torch.float16 if DEVICE in ("cuda", "mps") else torch.float32,
                local_files_only=offline,
                cache_dir=cache_folder,
            )

            if DEVICE == "mps":
                model.to("mps")

    if use_mlx:

        def mlx_generate(prompt: str, **kwargs: Any) -> str:
            current_temperature = kwargs.get("temperature", temperature)
            current_top_p = kwargs.get("top_p", top_p)
            current_top_k = kwargs.get("top_k", top_k)
            current_repetition_penalty = kwargs.get("repetition_penalty", repetition_penalty)
            current_max_tokens = kwargs.get("max_new_tokens", max_new_tokens)
            current_no_repeat = kwargs.get("no_repeat_ngram_size", no_repeat_ngram_size)
            do_sample = bool(kwargs.get("do_sample", current_temperature > 0))

            _require_mlx_no_repeat_support(current_no_repeat)
            sampler = make_sampler(
                temp=current_temperature if do_sample else 0.0,
                top_p=current_top_p,
                top_k=current_top_k,
            )
            logits_processors = make_logits_processors(repetition_penalty=current_repetition_penalty)

            return generate(
                model,
                tok,
                prompt=prompt,
                max_tokens=current_max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
            )

        generator = mlx_generate
    else:
        generator = pipeline(
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

    def invoke(messages, **runtime_kwargs: Any) -> dict[str, str]:
        formatted_messages = []
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "do_sample": temperature > 0,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
        }
        generation_kwargs.update(runtime_kwargs)

        for message in messages:
            if hasattr(message, "type"):
                if message.type == "human":
                    formatted_messages.append({"role": "user", "content": message.content})
                elif message.type == "ai":
                    formatted_messages.append({"role": "assistant", "content": message.content})
                elif message.type == "system":
                    formatted_messages.append({"role": "system", "content": message.content})
            elif isinstance(message, dict):
                formatted_messages.append(message)
            else:
                content = getattr(message, "content", str(message))
                role = getattr(message, "role", "user")
                formatted_messages.append({"role": role, "content": content})

        prompt = tok.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=True)
        if debugger_enabled:
            logger.debug("Prompt from apply_chat_template:\n%s\n", prompt)

        if use_mlx:
            result_text = generator(prompt, **generation_kwargs)
        else:
            result = generator(prompt, return_full_text=False, **generation_kwargs)
            result_text = result[0]["generated_text"]

        return {"role": "assistant", "content": result_text.strip()}

    return SimpleLLM(invoke)


def load_translator(
    repo_id: str,
    *,
    online: bool | None = None,
    cache_folder: str = EFFECTIVE_HF_HUB_CACHE,
):
    """Load an NLLB or OPUS-MT translation model and tokenizer."""
    effective_online = hf_online_enabled(True if online is None else online)
    load_kwargs = {"cache_dir": cache_folder, "local_files_only": not effective_online}
    logger.info("[HF] %s -> loading %s", "Online" if effective_online else "Offline", repo_id)

    tokenizer = AutoTokenizer.from_pretrained(repo_id, **load_kwargs)
    model = AutoModelForSeq2SeqLM.from_pretrained(repo_id, **load_kwargs)

    if DEVICE in ["cuda", "mps"]:
        model = model.to(DEVICE)
        logger.info("Translation model moved to %s", DEVICE)

    return model, tokenizer
