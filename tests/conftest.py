# tests/conftest.py
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agentic_rag.config_schema import Config, load_config


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def config_path(project_root: Path) -> Path:
    """Return the absolute path to config.yaml."""
    return project_root / "config.yaml"


@pytest.fixture(scope="session")
def config_data(config_path: Path) -> Config:
    return load_config(config_path)


@pytest.fixture
def mock_vectorstore():
    return MagicMock()


@pytest.fixture
def dummy_config():
    cfg = MagicMock()
    cfg.hf_home = "/tmp/hf_home"
    cfg.chat = False
    cfg.gradio = False
    return cfg
