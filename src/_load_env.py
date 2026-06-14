import logging
import os
import socket
from pathlib import Path

import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from rich.console import Console

logging.getLogger("transformers.pipelines.base").setLevel(logging.ERROR)
# logging.getLogger("transformers").setLevel(logging.ERROR)

# Patch numpy.array for fasttext NumPy 2.x compatibility
_original_array = np.array


def patched_array(*args, **kwargs):
    if "copy" in kwargs and kwargs["copy"] is False:
        kwargs.pop("copy")
        return np.asarray(*args, **kwargs)
    return _original_array(*args, **kwargs)


np.array = patched_array


def is_online_fast() -> bool:
    try:
        socket.create_connection(("huggingface.co", 443), timeout=0.25)
        return True
    except OSError:
        return False


ONLINE: bool = is_online_fast()

if not ONLINE:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"


#####################################################################################


console = Console(record=True, width=120, force_terminal=True)
USE_MPS = torch.backends.mps.is_available()
DEVICE = "mps" if USE_MPS else ("cuda" if torch.cuda.is_available() else "cpu")


class Config:
    def __init__(self, path="config.yaml"):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        for k, v in data.items():
            setattr(self, k, v)

        self._validate_ui_mode()

    def _validate_ui_mode(self):
        chat = getattr(self, "chat", False)
        gradio = getattr(self, "gradio", False)

        if chat and gradio:
            raise ValueError(
                "Configuration error: 'chat' and 'gradio' are mutually exclusive. Set only one to true in config.yaml"
            )


config_path = Path(__file__).resolve().parents[1] / "config.yaml"
cfg = Config(str(config_path))

load_dotenv()

if not os.environ.get("HF_TOKEN"):
    console.print(
        "[bold red]Warning:[/bold red] HuggingFace token not found in environment variables. "
        "Set HF_TOKEN in your .env file to enable model downloads and updates."
    )
    if ONLINE:
        raise ValueError("HuggingFace token is required for online mode. Please set HF_TOKEN in your .env file.")


# Set HuggingFace cache dir
if getattr(cfg, "hf_home", None):
    os.makedirs(cfg.hf_home, exist_ok=True)
    os.environ["HF_HOME"] = cfg.hf_home

if getattr(cfg, "debugger", False):
    os.environ["DEBUG_MODE"] = "1"


def attach_debugger_if_requested():
    if os.getenv("DEBUG_MODE"):
        import debugpy

        debugpy.listen(("0.0.0.0", 5643))
        print("✓ Models loaded. Debugger listening on 0.0.0.0:5643")
        print("Attach now, then press Enter to continue...")
        input()
