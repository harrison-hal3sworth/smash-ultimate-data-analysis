# Claude generated code for reading monthly .json files and filtering events

"""
02_filter_events.py

Read monthly .json files and flatten into one row per
Smash Ultimate event.
"""

from datetime import datetime, timezone
from pathlib import Path
import json

import pandas as pd

ROOT    = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "tournaments"
OUT_DIR = ROOT / "data" / "interim"
MIN_ENTRANTS = 32

SIDE_EVENT_WORDS = ("doubles", "crew", "amateur", "redemption", "ladder",
                    "squad strike", "2v2", "teams", "low tier", "arcadian")


def load_tournaments(dir: Path) -> list[dict]:
    """Read every YYYY-MM.json in dir; skip (and report) files that fail to parse."""
    tournaments = []
    for f in sorted(dir.glob("*.json")):
        try:
            rows = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"  ! skipping {f.name}: {e}")
            continue
        for t in rows:
            t["_source_file"] = f.name
        tournaments.extend(rows)
    return tournaments


def flatten_events(tournaments: list[dict]) -> pd.DataFrame:
    """One row per (tournament, Ultimate event)."""
    rows = []
    no_events = 0
    for t in tournaments:
        events = t.get("events") or []
        if not events:
            no_events += 1
        for ev in events:
            rows.append({
                "tournament_id":   t.get("id"),
                "tournament_name": t.get("name"),
                "slug":            t.get("slug"),
                "start_at":        t.get("startAt"),
                "city":            t.get("city"),
                "state":           t.get("addrState"),
                "event_id":        ev.get("id"),
                "event_name":      ev.get("name"),
                "is_online":       ev.get("isOnline"),
                "num_entrants":    ev.get("numEntrants"),
                "roster_max":      (ev.get("teamRosterSize") or {}).get("maxPlayers"),
                "source_file":     t.get("_source_file"),
            })
    print(f"  tournaments with no Ultimate events: {no_events:,}")

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ("tournament_id", "event_id", "start_at", "num_entrants", "roster_max"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df["date"] = pd.to_datetime(df["start_at"].astype("float"), unit="s", utc=True).dt.date
    return df


def drop_reason(row, now_ts: int | None = None) -> str | None:
    """First failing rule, in fixed order. None = kept."""
    now_ts = now_ts or int(datetime.now(timezone.utc).timestamp())

    if row["is_online"] is True:
        return "online"
    if not (pd.isna(row["roster_max"]) or row["roster_max"] == 1):
        return "not_singles"
    if pd.isna(row["num_entrants"]):
        return "no_entrant_count"
    if row["num_entrants"] < MIN_ENTRANTS:
        return "too_small"
    if pd.isna(row["start_at"]) or row["start_at"] > now_ts:
        return "not_yet_played"
    name = str(row["event_name"] or "").lower()
    if any(w in name for w in SIDE_EVENT_WORDS):
        return "side_event_name"
    return None


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset="event_id").copy()
    if len(df) < before:
        print(f"  removed {before - len(df):,} duplicate event rows")

    now_ts = int(datetime.now(timezone.utc).timestamp())
    df["drop_reason"] = df.apply(drop_reason, axis=1, now_ts=now_ts)
    df["kept"] = df["drop_reason"].isna()
    return df


def summarize(df: pd.DataFrame) -> None:
    kept = df[df["kept"]]
    print(f"\nEvents total: {len(df):,}   kept: {len(kept):,} ({len(kept) / max(len(df), 1):.1%})")

    print("\nDrop reasons:")
    print(df["drop_reason"].fillna("KEPT").value_counts().to_string())

    if kept.empty:
        return

    print("\nKept events per year:")
    years = pd.to_datetime(kept["date"]).dt.year
    print(kept.groupby(years).agg(events=("event_id", "size"),
                                  entrants=("num_entrants", "sum")).to_string())

    print("\nKept events by size:")
    bins = pd.cut(kept["num_entrants"], [MIN_ENTRANTS - 1, 63, 127, 255, float("inf")],
                  labels=["32-63", "64-127", "128-255", "256+"])
    print(kept.groupby(bins, observed=False).agg(events=("event_id", "size"),
                                                  entrants=("num_entrants", "sum")).to_string())

    print("\nLargest kept events (sanity check):")
    cols = ["date", "tournament_name", "event_name", "state", "num_entrants"]
    print(kept.nlargest(10, "num_entrants")[cols].to_string(index=False))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading {RAW_DIR}")
    tournaments = load_tournaments(RAW_DIR)
    print(f"  tournaments loaded: {len(tournaments):,}")

    df = flatten_events(tournaments)
    if df.empty:
        print("No events found. Has step 01 run?")
        return

    df = apply_filters(df)

    df.to_csv(OUT_DIR / "events_all.csv", index=False)
    df[df["kept"]].drop(columns=["kept", "drop_reason"]).to_csv(OUT_DIR / "events.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'events_all.csv'}")
    print(f"Wrote {OUT_DIR / 'events.csv'}")

    summarize(df)


if __name__ == "__main__":
    main()