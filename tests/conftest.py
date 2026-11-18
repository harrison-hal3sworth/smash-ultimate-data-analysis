# Contains Claude generated code

"""Shared test setup: make src/ importable and give tests access to the step-05 module."""
from pathlib import Path
import importlib.util
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


def load_build():
    """src/05_build_models.py can't be imported by name (it starts with a digit)."""
    spec = importlib.util.spec_from_file_location("build_models", SRC / "05_build_models.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_model_dir(root: Path) -> Path:
    """A tiny, fully known set of step-05 outputs for testing predict.py."""
    d = root / "data" / "models"
    d.mkdir(parents=True)
    (root / "src").mkdir(exist_ok=True)
    pd.DataFrame([
        # player_id, tag, other_tags, state, last_event, last_date, rank, rating, rd, sets, sets_last_12m
        (1, "Alpha", "", "CA", "Big Major", "2026-10-01", 1, 1900.0, 60.0, 500, 80),
        (2, "Beta", "OldBeta", "NY", "Local", "2026-09-01", 2, 1700.0, 70.0, 300, 40),
        (3, "Alpha", "", "TX", "Weekly", "2019-03-01", None, 1550.0, 250.0, 6, 0),
        (4, "Gamma", "", "FL", "Weekly", "2025-01-01", None, None, None, 2, 0),
        (5, "Alphabet", "", "WA", "Weekly", "2026-08-01", 3, 1650.0, 90.0, 120, 30),
    ], columns=["player_id", "tag", "other_tags", "state", "last_event", "last_date", "rank",
                "rating", "rd", "sets", "sets_last_12m"]).to_csv(d / "players.csv", index=False)
    pd.DataFrame([
        (1, "Steve", "last 12 months", 70, 0.7), (1, "Cloud", "last 12 months", 20, 0.2),
        (1, "Fox", "last 12 months", 10, 0.1), (2, "Kirby", "last 12 months", 40, 1.0),
        (5, "Fox", "all time", 9, 0.9), (5, "Cloud", "all time", 1, 0.1),
    ], columns=["player_id", "char", "window", "games", "share"]).to_csv(d / "usage.csv", index=False)
    model = {
        "beta": 1.0, "scale": 173.7178,
        "characters": ["Cloud", "Fox", "Kirby", "Steve"],
        "strength": {"Cloud": 0.0, "Fox": 0.05, "Kirby": -0.2, "Steve": 0.2},
        "interaction": {"Kirby|Steve": 0.1, "Cloud|Fox": -0.05},
        "games": {"Kirby|Steve": 500, "Cloud|Fox": 40, "Cloud|Kirby": 0, "Cloud|Steve": 300,
                  "Fox|Kirby": 200, "Fox|Steve": 250},
    }
    (d / "matchup_model.json").write_text(json.dumps(model))
    return root
