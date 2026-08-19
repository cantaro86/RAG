import runpy
from unittest.mock import MagicMock

import pytest

import agentic_rag.cli as cli

pytestmark = pytest.mark.cpu


def test_package_main_delegates_to_cli_main(monkeypatch):
    """Verify ``python -m agentic_rag`` delegates to the shared CLI entry point."""
    main = MagicMock()
    monkeypatch.setattr(cli, "main", main)

    runpy.run_module("agentic_rag.__main__", run_name="agentic_rag.__not_main__")
    main.assert_not_called()

    runpy.run_module("agentic_rag.__main__", run_name="__main__")

    main.assert_called_once_with()
