"""
02_filter_events.py

Read monthly .json files and flatten into one row per
Smash Ultimate event.
"""

from pathlib import Path
import json

ROOT    = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "tournaments"
OUT_DIR = ROOT / "data" / "interim"
MIN_ENTRANTS = 32

def load_tournaments(dir):
    pass

def flatten_events(tournaments):
    pass

def drop_reason(row):
    pass

def apply_filters(df):
    pass

def summarize(df):
    pass

if __name__ == "__main__":
    pass