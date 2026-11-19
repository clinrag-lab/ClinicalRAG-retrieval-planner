"""Load project settings from a YAML config file."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("config.yaml")


@dataclass
class Config:
    data_dir: Path
    notes_path: Path
    embed_model: str = "BAAI/bge-base-en-v1.5"
    llm_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    top_k: int = 5
    probe_k: int = 20
    alpha: float = 1.0  # token weight in plan cost
    output_dir: Path = Path("outputs/")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        missing = {"data_dir", "notes_path"} - set(raw)
        if missing:
            raise ValueError(f"Missing required config keys: {sorted(missing)}")

        values = dict(raw)
        for key in ("data_dir", "notes_path", "output_dir"):
            if key in values:
                values[key] = Path(values[key])
        for key in ("top_k", "probe_k"):
            if key in values:
                values[key] = int(values[key])
        if "alpha" in values:
            values["alpha"] = float(values["alpha"])
        return cls(**values)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Read a YAML file and return a validated Config."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. Copy config.example.yaml to config.yaml and edit it."
        )
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping.")
    return Config.from_dict(raw)
