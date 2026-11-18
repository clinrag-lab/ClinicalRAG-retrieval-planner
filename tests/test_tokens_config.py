from pathlib import Path

import pytest

from clinrag.config import Config, PlannerConfig, load_config
from clinrag.sources.tokens import count_tokens

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    "text,expected",
    [("", 0), ("one", 2), ("one two three", 4), (" ".join(["w"] * 10), 13), ("a  b\n c", 4)],
)
def test_count_tokens(text, expected):
    assert count_tokens(text) == expected


def test_example_config_loads():
    config = load_config(REPO_ROOT / "config.example.yaml")
    assert config.planner == PlannerConfig()
    assert config.embed_model == "BAAI/bge-base-en-v1.5"


def test_planner_keys_are_cast():
    config = Config.from_dict({"data_dir": "d", "notes_path": "n", "top_k": "7", "alpha": 2, "max_candidates": 50})
    assert config.planner.top_k == 7
    assert config.planner.alpha == 2.0 and isinstance(config.planner.alpha, float)
    assert config.planner.max_candidates == 50


def test_unknown_key_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        Config.from_dict({"data_dir": "d", "notes_path": "n", "topk": 5})
