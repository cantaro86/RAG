import os

import yaml


class Config:
    def __init__(self, path="config.yaml"):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        for k, v in data.items():
            setattr(self, k, v)


cfg = Config("config.yaml")

# Set HF token if provided
if getattr(cfg, "hf_token", None):
    os.environ["HF_TOKEN"] = cfg.hf_token

# Set HuggingFace cache dir
if getattr(cfg, "hf_home", None):
    os.makedirs(cfg.hf_home, exist_ok=True)
    os.environ["HF_HOME"] = cfg.hf_home
