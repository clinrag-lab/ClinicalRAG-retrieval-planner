"""Load project settings from a YAML config file."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("config.yaml")


@dataclass(frozen=True)
class PlannerConfig:
    """Settings used by the planner's cost model and executor.

    Latency constants and per-row token averages are starting guesses; calibrate
    them on real runs.
    """

    top_k: int = 5  # evidence items passed to the LLM per note search
    probe_k: int = 20  # scores fetched by the cheap note probe
    alpha: float = 1.0  # token weight in plan cost
    max_candidates: int = 1000  # cap on search-first candidates
    avg_tokens_per_row: float = 20.0  # tokens per relational result row
    avg_tokens_per_chunk: float = 256.0  # tokens per note chunk
    relational_ms: float = 20.0  # latency of one relational query
    graph_ms: float = 1.0  # latency of one graph expansion
    note_scan_ms_per_1k_notes: float = 0.5  # scoring notes in the vector index
    note_fetch_ms_per_1k_candidates: float = 2.0  # pulling candidates out and post-filtering them
    id_filter_ms_per_1k_ids: float = 1.0  # building and applying an admission id filter
    min_similarity: float = 0.3  # evidence check: best note score below this fails
    min_surviving_notes: int = 1  # evidence check: fewer notes after a filter fails
    max_fallbacks: int = 2  # plans run after the first when the evidence check fails

    def __post_init__(self) -> None:
        for f in fields(self):
            if getattr(self, f.name) < 0:
                raise ValueError(f"{f.name} must be non-negative")
        if self.top_k < 1 or self.probe_k < 1 or self.max_candidates < 1:
            raise ValueError("top_k, probe_k and max_candidates must be at least 1")


_PLANNER_KEYS = {f.name: f.type for f in fields(PlannerConfig)}
_CASTS = {"int": int, "float": float}


@dataclass(frozen=True)
class Config:
    data_dir: Path
    notes_path: Path
    embed_model: str = "BAAI/bge-base-en-v1.5"
    llm_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    output_dir: Path = Path("outputs/")
    planner: PlannerConfig = field(default_factory=PlannerConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        """Build a Config from flat YAML keys; planner keys go into Config.planner."""
        top_level = {f.name for f in fields(cls)} - {"planner"}
        unknown = set(raw) - top_level - set(_PLANNER_KEYS)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        missing = {"data_dir", "notes_path"} - set(raw)
        if missing:
            raise ValueError(f"Missing required config keys: {sorted(missing)}")

        values: dict[str, Any] = {k: v for k, v in raw.items() if k in top_level}
        for key in ("data_dir", "notes_path", "output_dir"):
            if key in values:
                values[key] = Path(values[key])

        planner_values = {
            k: _CASTS[str(_PLANNER_KEYS[k])](v) for k, v in raw.items() if k in _PLANNER_KEYS
        }
        values["planner"] = PlannerConfig(**planner_values)
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


def load_planner_config(path: str | Path) -> PlannerConfig:
    """PlannerConfig from a YAML file of planner keys (e.g. outputs/calibrated_config.yaml).

    Top-level Config keys (data_dir, notes_path, ...) are ignored, so a full
    config.yaml works too; unknown keys are an error. Missing keys keep their defaults.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping.")
    top_level = {f.name for f in fields(Config)} - {"planner"}
    unknown = set(raw) - top_level - set(_PLANNER_KEYS)
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    return PlannerConfig(**{k: _CASTS[str(_PLANNER_KEYS[k])](v) for k, v in raw.items() if k in _PLANNER_KEYS})
