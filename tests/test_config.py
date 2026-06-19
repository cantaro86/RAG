# tests/test_config.py
"""Tests for the project-level config.yaml configuration file.

These tests validate that config.yaml exists, is well-formed YAML,
contains all required keys with the correct types, and satisfies
the application's logical constraints.
"""

import pytest

# ---------------------------------------------------------------------------
# File existence
# ---------------------------------------------------------------------------


def test_config_file_exists(config_path):
    """Verify that config.yaml is present at the expected project root location."""
    assert config_path.exists(), f"config.yaml not found at: {config_path}"


def test_config_file_is_a_regular_file(config_path):
    """Verify that config.yaml is a regular file and not a directory or symlink."""
    assert config_path.is_file(), f"config.yaml path is not a regular file: {config_path}"


# ---------------------------------------------------------------------------
# YAML structure
# ---------------------------------------------------------------------------


def test_config_yaml_parses_without_error(config_data):
    """Verify that config.yaml can be parsed by PyYAML without raising an exception."""
    assert config_data is not None, "config.yaml is empty"


def test_config_yaml_is_a_mapping(config_data):
    """Verify that the top-level structure of config.yaml is a dictionary."""
    assert isinstance(config_data, dict), (
        f"Expected config.yaml top-level to be a dict, got {type(config_data).__name__}"
    )


# ---------------------------------------------------------------------------
# Required keys presence
# ---------------------------------------------------------------------------

REQUIRED_KEYS = [
    # Logging
    "log_level",
    "log_to_file",
    "log_file",
    # Debug
    "debugger",
    # Connectivity
    "online",
    # Paths
    "md_dir",
    "dizionario_path",
    "index_dir",
    # Indexing
    "reindex",
    "workers",
    # GPU / quantization
    "use_gpu_index",
    "quantization",
    # Models
    "embed_model",
    "llm_model",
    "rerank_model",
    # Retrieval
    "rerank",
    "k",
    "k_reranked",
    "search_type",
    "fetch_k",
    "lambda_mult",
    "threshold",
    # Output
    "clean_answer",
    # Chunking
    "chunk_size",
    "chunk_overlap",
    "min_chunk_length",
    # Generation
    "max_new_tokens",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "no_repeat_ngram_size",
    # HuggingFace
    "hf_home",
    # UI
    "chat",
    "gradio",
    "gradio_host",
    "gradio_port",
    "gradio_share",
    "max_history_turns",
]


@pytest.mark.parametrize("key", REQUIRED_KEYS)
def test_config_has_required_key(config_data, key):
    """Verify that each required key is present in config.yaml."""
    assert key in config_data, f"Missing required key in config.yaml: '{key}'"


# ---------------------------------------------------------------------------
# Type validation
# ---------------------------------------------------------------------------


def test_config_types_logging(config_data):
    """Verify logging-related fields have the expected types."""
    assert isinstance(config_data["log_level"], str), "log_level must be a string"
    assert isinstance(config_data["log_to_file"], bool), "log_to_file must be a bool"
    assert isinstance(config_data["log_file"], str), "log_file must be a string"


def test_config_types_flags(config_data):
    """Verify boolean flag fields have the expected types."""
    bool_keys = [
        "debugger",
        "online",
        "reindex",
        "use_gpu_index",
        "quantization",
        "rerank",
        "clean_answer",
        "gradio_share",
    ]
    for key in bool_keys:
        assert isinstance(config_data[key], bool), f"'{key}' must be a bool, got {type(config_data[key]).__name__}"


def test_config_types_paths(config_data):
    """Verify filesystem path fields are strings."""
    path_keys = ["md_dir", "dizionario_path", "index_dir", "hf_home"]
    for key in path_keys:
        if config_data[key] is not None:
            assert isinstance(config_data[key], str), (
                f"'{key}' must be a string or null, got {type(config_data[key]).__name__}"
            )


def test_config_types_models(config_data):
    """Verify HuggingFace model identifiers are non-empty strings."""
    for key in ("embed_model", "llm_model", "rerank_model"):
        assert isinstance(config_data[key], str), f"'{key}' must be a string"
        assert len(config_data[key]) > 0, f"'{key}' must not be an empty string"


def test_config_types_integers(config_data):
    """Verify integer fields have the correct type."""
    int_keys = [
        "workers",
        "k",
        "k_reranked",
        "fetch_k",
        "chunk_size",
        "chunk_overlap",
        "min_chunk_length",
        "max_new_tokens",
        "top_k",
        "no_repeat_ngram_size",
        "gradio_port",
        "max_history_turns",
    ]
    for key in int_keys:
        assert isinstance(config_data[key], int), f"'{key}' must be an int, got {type(config_data[key]).__name__}"


def test_config_types_floats(config_data):
    """Verify float fields have a numeric type (int or float are both acceptable)."""
    float_keys = ["lambda_mult", "threshold", "temperature", "top_p", "repetition_penalty"]
    for key in float_keys:
        assert isinstance(config_data[key], int | float), (
            f"'{key}' must be numeric, got {type(config_data[key]).__name__}"
        )


# ---------------------------------------------------------------------------
# Value range validation
# ---------------------------------------------------------------------------


def test_config_log_level_is_valid(config_data):
    """Verify log_level is one of the accepted Python logging level names."""
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    assert config_data["log_level"] in valid_levels, (
        f"log_level '{config_data['log_level']}' is not a valid logging level. Expected one of: {valid_levels}"
    )


def test_config_search_type_is_valid(config_data):
    """Verify search_type is one of the supported retrieval strategies."""
    valid_types = {"similarity", "mmr"}
    assert config_data["search_type"] in valid_types, (
        f"search_type '{config_data['search_type']}' is invalid. Expected one of: {valid_types}"
    )


def test_config_k_values_are_positive(config_data):
    """Verify retrieval k values are positive integers."""
    assert config_data["k"] > 0, "k must be a positive integer"
    assert config_data["k_reranked"] > 0, "k_reranked must be a positive integer"
    assert config_data["fetch_k"] > 0, "fetch_k must be a positive integer"


def test_config_k_reranked_less_than_k(config_data):
    """Verify k_reranked <= k, since you cannot return more results than retrieved."""
    assert config_data["k_reranked"] <= config_data["k"], (
        f"k_reranked ({config_data['k_reranked']}) must be <= k ({config_data['k']})"
    )


def test_config_lambda_mult_in_range(config_data):
    """Verify lambda_mult is in [0.0, 1.0] as required by the MMR algorithm."""
    assert 0.0 <= config_data["lambda_mult"] <= 1.0, (
        f"lambda_mult must be between 0.0 and 1.0, got {config_data['lambda_mult']}"
    )


def test_config_temperature_in_range(config_data):
    """Verify temperature is in a reasonable range for LLM generation."""
    assert 0.0 <= config_data["temperature"] <= 2.0, (
        f"temperature must be between 0.0 and 2.0, got {config_data['temperature']}"
    )


def test_config_top_p_in_range(config_data):
    """Verify top_p is in (0.0, 1.0] as required by nucleus sampling."""
    assert 0.0 < config_data["top_p"] <= 1.0, f"top_p must be in (0.0, 1.0], got {config_data['top_p']}"


def test_config_workers_positive(config_data):
    """Verify the number of indexing workers is at least 1."""
    assert config_data["workers"] >= 1, f"workers must be >= 1, got {config_data['workers']}"


def test_config_chunk_overlap_less_than_chunk_size(config_data):
    """Verify chunk_overlap < chunk_size to avoid degenerate chunking."""
    assert config_data["chunk_overlap"] < config_data["chunk_size"], (
        f"chunk_overlap ({config_data['chunk_overlap']}) must be < chunk_size ({config_data['chunk_size']})"
    )


def test_config_repetition_penalty_valid(config_data):
    """Verify repetition_penalty is >= 1.0 (values below 1.0 encourage repetition)."""
    assert config_data["repetition_penalty"] >= 1.0, (
        f"repetition_penalty should be >= 1.0, got {config_data['repetition_penalty']}"
    )


# ---------------------------------------------------------------------------
# Logical / mutual exclusion constraints
# ---------------------------------------------------------------------------


def test_config_chat_and_gradio_are_mutually_exclusive(config_data):
    """Verify chat and gradio are not both true, as the app only supports one UI at a time."""
    assert not (config_data["chat"] and config_data["gradio"]), (
        "Invalid config.yaml: 'chat' and 'gradio' cannot both be true"
    )


def test_config_max_history_turns_positive(config_data):
    """Verify max_history_turns is at least 1 to maintain at least one conversational turn."""
    assert config_data["max_history_turns"] >= 1, (
        f"max_history_turns must be >= 1, got {config_data['max_history_turns']}"
    )
