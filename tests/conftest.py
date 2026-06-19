# tests/conftest.py
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def config_path(project_root: Path) -> Path:
    """Return the absolute path to config.yaml."""
    return project_root / "config.yaml"


@pytest.fixture(scope="session")
def config_data(config_path: Path) -> dict:
    """Load config.yaml once and return its parsed contents.

    Skip dependent tests if the configuration file is missing.
    """
    if not config_path.exists():
        pytest.skip(f"config.yaml not found: {config_path}")

    if not config_path.is_file():
        pytest.skip(f"config.yaml is not a regular file: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        pytest.fail(f"config.yaml is empty: {config_path}")

    if not isinstance(data, dict):
        pytest.fail(f"config.yaml must contain a top-level mapping: {config_path}")

    return data


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
