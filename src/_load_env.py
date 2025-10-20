import os
from pathlib import Path

import torch
import yaml
from rich.console import Console

console = Console()
USE_MPS = torch.backends.mps.is_available()
DEVICE = "mps" if USE_MPS else ("cuda" if torch.cuda.is_available() else "cpu")


class Config:
    def __init__(self, path="config.yaml"):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        for k, v in data.items():
            setattr(self, k, v)


config_path = Path(__file__).resolve().parents[1] / "config.yaml"
cfg = Config(str(config_path))

# Set HF token if provided
if getattr(cfg, "hf_token", None):
    os.environ["HF_TOKEN"] = cfg.hf_token

# Set HuggingFace cache dir
if getattr(cfg, "hf_home", None):
    os.makedirs(cfg.hf_home, exist_ok=True)
    os.environ["HF_HOME"] = cfg.hf_home
