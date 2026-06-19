# tests/test_config.py
"""Tests for the project-level config.yaml configuration file.

These tests validate that config.yaml exists, is well-formed YAML,
contains all required keys with the correct types, and satisfies
the application's logical constraints.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_rag.config_schema import Config, load_config

from .helpers import make_valid_config, write_yaml


def test_config_file_exists(config_path: Path) -> None:
    """Verify that config.yaml is present at the expected project root location."""
    assert config_path.exists(), f"config.yaml not found at: {config_path}"


def test_config_file_is_a_regular_file(config_path: Path) -> None:
    """Verify that config.yaml is a regular file."""
    assert config_path.is_file(), f"config.yaml path is not a regular file: {config_path}"


def test_config_loads_as_valid_schema(config_path: Path) -> None:
    """Verify that the real config.yaml loads successfully into the Config schema."""
    cfg = load_config(config_path)
    assert isinstance(cfg, Config)


@pytest.mark.parametrize("missing_key", ["llm_model", "dizionario_path"])
def test_load_config_raises_validation_error_when_required_key_is_missing(
    tmp_path: Path,
    missing_key: str,
) -> None:
    """Verify that load_config raises ValidationError when a required key is missing."""
    path = tmp_path / "config.yaml"
    data = make_valid_config()
    data.pop(missing_key)
    write_yaml(path, data)

    with pytest.raises(ValidationError) as exc_info:
        load_config(path)

    errors = exc_info.value.errors()
    assert any(err["loc"] == (missing_key,) and err["type"] == "missing" for err in errors), (
        f"Expected missing-field error for '{missing_key}', got: {errors}"
    )


def test_load_config_raises_for_missing_file(tmp_path) -> None:
    """Verify that loading a missing config file raises FileNotFoundError."""
    missing = tmp_path / "config.yaml"

    with pytest.raises(FileNotFoundError, match="not found|No such file"):
        load_config(missing)


def test_load_config_returns_config_instance_for_valid_yaml(tmp_path) -> None:
    """Verify that a valid YAML file is parsed and returned as a Config instance."""
    path = tmp_path / "config.yaml"
    write_yaml(path, make_valid_config())

    cfg = load_config(path)

    assert isinstance(cfg, Config)


def test_load_config_raises_validation_error_when_chat_and_gradio_are_both_true(
    tmp_path,
) -> None:
    """Verify that the schema rejects enabling chat and gradio at the same time."""
    path = tmp_path / "config.yaml"
    data = make_valid_config()
    data["chat"] = True
    data["gradio"] = True
    write_yaml(path, data)

    with pytest.raises(ValidationError, match="chat|gradio"):
        load_config(path)


def test_load_config_raises_validation_error_when_k_reranked_exceeds_k(
    tmp_path,
) -> None:
    """Verify that the schema rejects k_reranked values larger than k."""
    path = tmp_path / "config.yaml"
    data = make_valid_config()
    data["k"] = 4
    data["k_reranked"] = 6
    write_yaml(path, data)

    with pytest.raises(ValidationError, match="k_reranked|k"):
        load_config(path)


def test_load_config_raises_validation_error_when_chunk_overlap_is_too_large(
    tmp_path,
) -> None:
    """Verify that the schema rejects chunk_overlap greater than or equal to chunk_size."""
    path = tmp_path / "config.yaml"
    data = make_valid_config()
    data["chunk_size"] = 300
    data["chunk_overlap"] = 300
    write_yaml(path, data)

    with pytest.raises(ValidationError, match="chunk_overlap|chunk_size"):
        load_config(path)


def test_load_config_raises_validation_error_for_invalid_log_level(tmp_path) -> None:
    """Verify that the schema rejects unsupported log levels."""
    path = tmp_path / "config.yaml"
    data = make_valid_config()
    data["log_level"] = "TRACE"
    write_yaml(path, data)

    with pytest.raises(ValidationError, match="log_level"):
        load_config(path)
