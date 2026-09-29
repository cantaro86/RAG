import logging
import os
import socket
from pathlib import Path

import torch
from dotenv import load_dotenv
from rich.console import Console

from agentic_rag.config_schema import load_config

logging.getLogger("transformers.pipelines.base").setLevel(logging.ERROR)


def is_online_fast() -> bool:
    connection = None
    try:
        connection = socket.create_connection(("huggingface.co", 443), timeout=0.25)
        return True
    except OSError:
        return False
    finally:
        if connection is not None:
            connection.close()


def _env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _offline_requested(configured_online: bool) -> bool:
    return not configured_online or _env_flag_enabled("HF_HUB_OFFLINE") or _env_flag_enabled("TRANSFORMERS_OFFLINE")


def hf_online_enabled(configured_online: bool) -> bool:
    """Return whether remote model access is allowed by config and environment."""
    return not _offline_requested(configured_online)


def _network_available(configured_online: bool) -> bool:
    return False if _offline_requested(configured_online) else is_online_fast()


def _resolve_config_path() -> Path:
    override = os.environ.get("AGENTIC_RAG_CONFIG")
    if override:
        path = Path(override).expanduser()
        return (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()

    working_directory_config = Path.cwd() / "config.yaml"
    if working_directory_config.is_file():
        return working_directory_config.resolve()

    return Path(__file__).resolve().parents[2] / "config.yaml"


CONFIG_PATH = _resolve_config_path()
PROJECT_ROOT = CONFIG_PATH.parent
DOTENV_PATH = PROJECT_ROOT / ".env"

cfg = load_config(CONFIG_PATH)
load_dotenv(dotenv_path=DOTENV_PATH)

NETWORK_PROBE_SKIPPED: bool = _offline_requested(cfg.online)
NETWORK_AVAILABLE: bool = _network_available(cfg.online)
HF_ONLINE: bool = hf_online_enabled(cfg.online)
ONLINE: bool = HF_ONLINE  # Backward-compatible alias for the effective mode.

_offline_value = "0" if HF_ONLINE else "1"
os.environ["HF_HUB_OFFLINE"] = _offline_value
os.environ["TRANSFORMERS_OFFLINE"] = _offline_value

#####################################################################################


console = Console(record=True, width=120, force_terminal=True)
USE_MPS = torch.backends.mps.is_available()
DEVICE = "mps" if USE_MPS else ("cuda" if torch.cuda.is_available() else "cpu")


if not os.environ.get("HF_TOKEN"):
    console.print(
        "[bold red]Warning:[/bold red] HuggingFace token not found in environment variables. "
        "Set HF_TOKEN in your .env file to enable model downloads and updates."
    )


# Set HuggingFace cache dir
_configured_hf_home = getattr(cfg, "hf_home", None) or os.environ.get("HF_HOME")
_hf_home_path = (
    Path(_configured_hf_home).expanduser() if _configured_hf_home else Path.home() / ".cache" / "huggingface"
)
try:
    _hf_home_path.mkdir(parents=True, exist_ok=True)
except OSError:
    _fallback_hf_home = Path.home() / ".cache" / "huggingface"
    console.print(
        f"[yellow]Warning:[/yellow] Cannot use hf_home='{_hf_home_path}', falling back to '{_fallback_hf_home}'"
    )
    _fallback_hf_home.mkdir(parents=True, exist_ok=True)
    _hf_home_path = _fallback_hf_home

EFFECTIVE_HF_HOME: str = str(_hf_home_path.resolve())
os.environ["HF_HOME"] = EFFECTIVE_HF_HOME
cfg.hf_home = EFFECTIVE_HF_HOME

_hub_cache_path = Path(os.environ.get("HF_HUB_CACHE", _hf_home_path / "hub")).expanduser()
try:
    _hub_cache_path.mkdir(parents=True, exist_ok=True)
except OSError:
    _hub_cache_path = Path.home() / ".cache" / "huggingface" / "hub"
    _hub_cache_path.mkdir(parents=True, exist_ok=True)

EFFECTIVE_HF_HUB_CACHE: str = str(_hub_cache_path.resolve())
os.environ["HF_HUB_CACHE"] = EFFECTIVE_HF_HUB_CACHE


def hf_hub_cache_for(hf_home: str) -> str:
    """Return the Hub cache corresponding to a Config.hf_home value."""
    home = Path(hf_home).expanduser().resolve()
    if home == Path(EFFECTIVE_HF_HOME):
        return EFFECTIVE_HF_HUB_CACHE
    return str(home / "hub")


if getattr(cfg, "debugger", False):
    os.environ["DEBUG_MODE"] = "1"
    import logging

    logging.basicConfig(level=logging.DEBUG)


def attach_debugger_if_requested():
    if os.getenv("DEBUG_MODE"):
        import debugpy

        debugpy.listen(("0.0.0.0", 5643))
        print("✓ Models loaded. Debugger listening on 0.0.0.0:5643")
        print("Attach now, then press Enter to continue...")
        input()
