import json
from unittest.mock import MagicMock

import huggingface_hub
import pytest

import agentic_rag.llm_build as llm_build

pytestmark = pytest.mark.cpu


def _build_kwargs(**overrides):
    kwargs = {
        "model_name": "example/model",
        "max_new_tokens": 64,
        "temperature": 0.4,
        "top_p": 0.95,
        "top_k": 40,
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 0,
        "quantization": False,
        "online": False,
        "debugger": False,
    }
    kwargs.update(overrides)
    return kwargs


def _standard_backend(monkeypatch):
    tokenizer = MagicMock(eos_token_id=2)
    tokenizer.apply_chat_template.return_value = "formatted prompt"
    tokenizer_loader = MagicMock(return_value=tokenizer)
    model = MagicMock()
    model_loader = MagicMock(return_value=model)
    generator = MagicMock(return_value=[{"generated_text": "  risposta  "}])
    pipeline_factory = MagicMock(return_value=generator)
    monkeypatch.setattr(llm_build, "DEVICE", "cpu")
    monkeypatch.setattr(llm_build.AutoTokenizer, "from_pretrained", tokenizer_loader)
    monkeypatch.setattr(llm_build.AutoModelForCausalLM, "from_pretrained", model_loader)
    monkeypatch.setattr(llm_build, "pipeline", pipeline_factory)
    return tokenizer, tokenizer_loader, model_loader, generator, pipeline_factory


def test_standard_backend_honors_explicit_offline_debugger_and_bind(monkeypatch):
    tokenizer, tokenizer_loader, model_loader, generator, _pipeline_factory = _standard_backend(monkeypatch)
    debug = MagicMock()
    monkeypatch.setattr(llm_build.logger, "debug", debug)

    llm = llm_build.build_llm_pipe(**_build_kwargs(debugger=True))
    bound = llm.bind(do_sample=False, top_k=7)
    result = bound.invoke([{"role": "user", "content": "domanda"}])

    assert isinstance(llm, llm_build.SimpleLLM)
    assert result == {"role": "assistant", "content": "risposta"}
    tokenizer_loader.assert_called_once_with(
        "example/model",
        token=llm_build.os.environ.get("HF_TOKEN"),
        local_files_only=True,
        cache_dir=llm_build.EFFECTIVE_HF_HUB_CACHE,
    )
    model_loader.assert_called_once()
    assert model_loader.call_args.kwargs["local_files_only"] is True
    tokenizer.apply_chat_template.assert_called_once_with(
        [{"role": "user", "content": "domanda"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert generator.call_args.kwargs["do_sample"] is False
    assert generator.call_args.kwargs["top_k"] == 7
    debug.assert_called_once_with("Prompt from apply_chat_template:\n%s\n", "formatted prompt")


def test_non_mistral_tokenizer_type_error_is_not_masked(monkeypatch):
    monkeypatch.setattr(llm_build, "DEVICE", "cpu")
    original_error = TypeError("tokenizer implementation failed")
    tokenizer_loader = MagicMock(side_effect=original_error)
    monkeypatch.setattr(llm_build.AutoTokenizer, "from_pretrained", tokenizer_loader)

    with pytest.raises(TypeError) as raised:
        llm_build.build_llm_pipe(**_build_kwargs())

    assert raised.value is original_error
    tokenizer_loader.assert_called_once()


def test_mistral_regex_fallback_only_retries_unsupported_keyword(monkeypatch):
    tokenizer = MagicMock(eos_token_id=2)
    tokenizer.apply_chat_template.return_value = "prompt"
    tokenizer_loader = MagicMock(side_effect=[TypeError("unexpected keyword argument 'fix_mistral_regex'"), tokenizer])
    monkeypatch.setattr(llm_build, "DEVICE", "cpu")
    monkeypatch.setattr(llm_build.AutoTokenizer, "from_pretrained", tokenizer_loader)
    monkeypatch.setattr(llm_build.AutoModelForCausalLM, "from_pretrained", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(llm_build, "pipeline", MagicMock(return_value=MagicMock()))

    llm_build.build_llm_pipe(**_build_kwargs(model_name="mistralai/test"))

    assert tokenizer_loader.call_count == 2
    assert tokenizer_loader.call_args_list[0].kwargs["fix_mistral_regex"] is True
    assert "fix_mistral_regex" not in tokenizer_loader.call_args_list[1].kwargs


def test_mlx_uses_single_prequantized_load_and_honors_sampling_controls(monkeypatch):
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "mlx prompt"
    model = object()
    model_config = {"quantization": {"bits": 4, "group_size": 64}}
    mlx_load = MagicMock(return_value=(model, tokenizer))
    sampler = object()
    logits_processors = [object()]
    make_sampler = MagicMock(return_value=sampler)
    make_logits_processors = MagicMock(return_value=logits_processors)
    generate = MagicMock(return_value="  output  ")
    tokenizer_loader = MagicMock()
    monkeypatch.setattr(llm_build, "DEVICE", "mps")
    monkeypatch.setattr(llm_build, "hf_online_enabled", lambda online: online)
    monkeypatch.setattr(llm_build, "MLX_AVAILABLE", True)
    monkeypatch.setattr(llm_build, "_load_mlx_model_config", MagicMock(return_value=model_config))
    monkeypatch.setattr(llm_build, "load", mlx_load)
    monkeypatch.setattr(llm_build, "make_sampler", make_sampler)
    monkeypatch.setattr(llm_build, "make_logits_processors", make_logits_processors)
    monkeypatch.setattr(llm_build, "generate", generate)
    monkeypatch.setattr(llm_build.AutoTokenizer, "from_pretrained", tokenizer_loader)
    pipeline_factory = MagicMock()
    monkeypatch.setattr(llm_build, "pipeline", pipeline_factory)

    llm = llm_build.build_llm_pipe(**_build_kwargs(quantization=True, online=True, temperature=0.8, top_k=23))
    result = llm.bind(do_sample=False, top_k=7).invoke([{"role": "user", "content": "test"}])

    assert result == {"role": "assistant", "content": "output"}
    mlx_load.assert_called_once_with("example/model")
    tokenizer_loader.assert_not_called()
    pipeline_factory.assert_not_called()
    make_sampler.assert_called_once_with(temp=0.0, top_p=0.95, top_k=7)
    make_logits_processors.assert_called_once_with(repetition_penalty=1.1)
    generate.assert_called_once_with(
        model,
        tokenizer,
        prompt="mlx prompt",
        max_tokens=64,
        sampler=sampler,
        logits_processors=logits_processors,
    )


def test_mlx_sampling_passes_top_k_when_enabled(monkeypatch):
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "prompt"
    monkeypatch.setattr(llm_build, "DEVICE", "mps")
    monkeypatch.setattr(llm_build, "hf_online_enabled", lambda online: online)
    monkeypatch.setattr(llm_build, "MLX_AVAILABLE", True)
    monkeypatch.setattr(
        llm_build,
        "_load_mlx_model_config",
        MagicMock(return_value={"quantization_config": {"quant_method": "mxfp4"}}),
    )
    monkeypatch.setattr(
        llm_build,
        "load",
        MagicMock(return_value=(object(), tokenizer)),
    )
    make_sampler = MagicMock(return_value=object())
    monkeypatch.setattr(llm_build, "make_sampler", make_sampler)
    monkeypatch.setattr(llm_build, "make_logits_processors", MagicMock(return_value=[]))
    monkeypatch.setattr(llm_build, "generate", MagicMock(return_value="output"))

    llm = llm_build.build_llm_pipe(**_build_kwargs(quantization=True, online=True))
    llm.bind(do_sample=True, temperature=0.6, top_k=11).invoke([{"role": "user", "content": "test"}])

    make_sampler.assert_called_once_with(temp=0.6, top_p=0.95, top_k=11)


def test_mlx_unavailable_preserves_import_error(monkeypatch):
    import_error = ImportError("missing MLX dependency")
    monkeypatch.setattr(llm_build, "DEVICE", "mps")
    monkeypatch.setattr(llm_build, "MLX_AVAILABLE", False)
    monkeypatch.setattr(llm_build, "MLX_IMPORT_ERROR", import_error)
    monkeypatch.setattr(llm_build, "load", None)

    with pytest.raises(RuntimeError, match="requires mlx-lm") as raised:
        llm_build.build_llm_pipe(**_build_kwargs(quantization=True))

    assert raised.value.__cause__ is import_error


def test_mlx_quantization_rejects_ordinary_model_config(monkeypatch):
    tokenizer_loader = MagicMock()
    mlx_load = MagicMock()
    monkeypatch.setattr(llm_build, "DEVICE", "mps")
    monkeypatch.setattr(llm_build, "hf_online_enabled", lambda online: online)
    monkeypatch.setattr(llm_build, "MLX_AVAILABLE", True)
    monkeypatch.setattr(llm_build, "_load_mlx_model_config", MagicMock(return_value={}))
    monkeypatch.setattr(llm_build, "load", mlx_load)
    monkeypatch.setattr(llm_build.AutoTokenizer, "from_pretrained", tokenizer_loader)

    with pytest.raises(ValueError, match="already-quantized MLX model"):
        llm_build.build_llm_pipe(**_build_kwargs(quantization=True, online=True))
    mlx_load.assert_not_called()
    tokenizer_loader.assert_not_called()


def test_mlx_fails_clearly_for_nonzero_no_repeat_ngram(monkeypatch):
    mlx_load = MagicMock()
    monkeypatch.setattr(llm_build, "DEVICE", "mps")
    monkeypatch.setattr(llm_build, "hf_online_enabled", lambda online: online)
    monkeypatch.setattr(llm_build, "MLX_AVAILABLE", True)
    monkeypatch.setattr(llm_build, "load", mlx_load)

    with pytest.raises(NotImplementedError, match="not supported by the pinned mlx-lm backend"):
        llm_build.build_llm_pipe(**_build_kwargs(quantization=True, online=True, no_repeat_ngram_size=2))
    mlx_load.assert_not_called()


@pytest.mark.parametrize(
    "model_config",
    [
        {},
        {"quantization": True},
        {"quantization": {"bits": 4}},
        {"quantization": {"bits": 4, "group_size": 64, "mode": "unsupported"}},
        {"quantization_config": {"quant_method": "gptq"}},
    ],
)
def test_mlx_rejects_unsupported_quantization_metadata(model_config):
    with pytest.raises(ValueError, match="quantization"):
        llm_build._validate_mlx_quantization_config(model_config)


def test_mlx_reads_remote_metadata_with_explicit_offline_cache(monkeypatch, tmp_path):
    config_path = tmp_path / "config.json"
    model_config = {"quantization": {"bits": 4, "group_size": 64}}
    config_path.write_text(json.dumps(model_config), encoding="utf-8")
    download = MagicMock(return_value=str(config_path))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(llm_build, "EFFECTIVE_HF_HUB_CACHE", str(tmp_path / "hf-cache"))

    assert (
        llm_build._load_mlx_model_config(
            "example/model",
            online=False,
            cache_folder=str(tmp_path / "hf-cache"),
        )
        == model_config
    )
    download.assert_called_once_with(
        repo_id="example/model",
        filename="config.json",
        token=llm_build.os.environ.get("HF_TOKEN"),
        cache_dir=str(tmp_path / "hf-cache"),
        local_files_only=True,
    )


def test_mlx_reads_local_metadata_without_hub_access(monkeypatch, tmp_path):
    model_dir = tmp_path / "mlx-model"
    model_dir.mkdir()
    model_config = {"quantization_config": {"quant_method": "compressed-tensors"}}
    (model_dir / "config.json").write_text(json.dumps(model_config), encoding="utf-8")
    download = MagicMock()
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)

    assert llm_build._load_mlx_model_config(str(model_dir), online=True) == model_config
    download.assert_not_called()


def test_mlx_passes_mistral_tokenizer_fix_to_pinned_load_api(monkeypatch):
    tokenizer = MagicMock()
    mlx_load = MagicMock(return_value=(object(), tokenizer))
    monkeypatch.setattr(llm_build, "DEVICE", "mps")
    monkeypatch.setattr(llm_build, "hf_online_enabled", lambda online: online)
    monkeypatch.setattr(llm_build, "MLX_AVAILABLE", True)
    monkeypatch.setattr(
        llm_build,
        "_load_mlx_model_config",
        MagicMock(
            return_value={
                "model_type": "mistral",
                "quantization": {"bits": 4, "group_size": 64},
            }
        ),
    )
    monkeypatch.setattr(llm_build, "load", mlx_load)
    monkeypatch.setattr(llm_build, "make_sampler", MagicMock())
    monkeypatch.setattr(llm_build, "make_logits_processors", MagicMock())
    monkeypatch.setattr(llm_build, "generate", MagicMock())

    llm_build.build_llm_pipe(
        **_build_kwargs(model_name="organization/prequantized-mlx", quantization=True, online=True)
    )

    mlx_load.assert_called_once_with(
        "organization/prequantized-mlx",
        tokenizer_config={"fix_mistral_regex": True},
    )


def test_translator_explicit_offline_mode_uses_local_cache(monkeypatch):
    tokenizer = object()
    model = MagicMock()
    tokenizer_loader = MagicMock(return_value=tokenizer)
    model_loader = MagicMock(return_value=model)
    monkeypatch.setattr(llm_build, "DEVICE", "cpu")
    monkeypatch.setattr(llm_build, "hf_online_enabled", lambda online: online)
    monkeypatch.setattr(llm_build.AutoTokenizer, "from_pretrained", tokenizer_loader)
    monkeypatch.setattr(llm_build.AutoModelForSeq2SeqLM, "from_pretrained", model_loader)

    assert llm_build.load_translator("translation/model", online=False) == (model, tokenizer)
    tokenizer_loader.assert_called_once_with(
        "translation/model",
        cache_dir=llm_build.EFFECTIVE_HF_HUB_CACHE,
        local_files_only=True,
    )
    model_loader.assert_called_once_with(
        "translation/model",
        cache_dir=llm_build.EFFECTIVE_HF_HUB_CACHE,
        local_files_only=True,
    )
