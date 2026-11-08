"""
08_pull_seeding.py  —  organizer seeds + final placements for the auto-seeder test events.

Picks test events from data/interim/events.csv (default: 50 random 2026 events with 128+
entrants that are in the set data), and pulls every entrant's initial seed and final
placement from start.gg. About 5 minutes for 50 events. Resumable: finished events are skipped.

Outputs:
    data/raw/seeding/<event_id>.json     raw API response per event
    data/interim/seeding.csv             event_id, player_id, tag, seed, placement, dq

Run from the project root:
    python src/08_pull_seeding.py               # 50 events
    python src/08_pull_seeding.py --n 100       # more events
"""

from pathlib import Path
import argparse
import json
import os
import sys

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_pull import StartGG  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVENTS_CSV = ROOT / "data" / "interim" / "events.csv"
SETS_CSV = ROOT / "data" / "processed" / "sets.csv"
RAW = ROOT / "data" / "raw" / "seeding"
OUT = ROOT / "data" / "interim" / "seeding.csv"

TEST_FROM = "2026-01-01"
MIN_ENTRANTS = 128
PER_PAGE = 64
SEED = 0

QUERY = """
query EventEntrants($eventId: ID!, $page: Int!, $perPage: Int!) {
  event(id: $eventId) {
    id name numEntrants
    entrants(query: {page: $page, perPage: $perPage}) {
      pageInfo { total totalPages }
      nodes {
        id initialSeedNum isDisqualified
        standing { placement }
        participants { player { id gamerTag } }
      }
    }
  }
}
"""


def pick_events(n: int) -> pd.DataFrame:
    ev = pd.read_csv(EVENTS_CSV)
    ev = ev[(pd.to_datetime(ev.date) >= TEST_FROM) & (ev.num_entrants >= MIN_ENTRANTS)]
    played = pd.read_csv(SETS_CSV, usecols=["event_id", "status"])
    played = set(played.loc[played.status == "complete", "event_id"])
    ev = ev[ev.event_id.isin(played)]
    print(f"{len(ev)} eligible events (from {TEST_FROM}, {MIN_ENTRANTS}+ entrants, with set data)")
    return ev.sample(min(n, len(ev)), random_state=SEED).sort_values("date")


def pull_event(gg: StartGG, event_id: int) -> list[dict]:
    nodes, page = [], 1
    while True:
        data = gg.query(QUERY, {"eventId": int(event_id), "page": page, "perPage": PER_PAGE})
        ent = data["event"]["entrants"]
        nodes += ent["nodes"]
        if page >= (ent["pageInfo"]["totalPages"] or 0):
            return nodes
        page += 1


def flatten(event_id: int, nodes: list[dict]) -> list[dict]:
    rows = []
    for e in nodes:
        parts = e.get("participants") or []
        player = (parts[0].get("player") or {}) if parts else {}
        rows.append({
            "event_id": event_id,
            "entrant_id": e.get("id"),
            "player_id": player.get("id"),
            "tag": player.get("gamerTag"),
            "seed": e.get("initialSeedNum"),
            "placement": (e.get("standing") or {}).get("placement"),
            "dq": bool(e.get("isDisqualified")),
        })
    return rows


def main(n: int):
    RAW.mkdir(parents=True, exist_ok=True)
    events = pick_events(n)
    todo = [e for e in events.event_id if not (RAW / f"{int(e)}.json").exists()]
    print(f"{len(events)} test events, {len(events) - len(todo)} already pulled, {len(todo)} to pull")

    if todo:
        gg = StartGG(log=tqdm.write)
        for event_id in tqdm(todo, desc="Events", unit="event"):
            try:
                nodes = pull_event(gg, event_id)
            except RuntimeError as e:
                msg = str(e)
                tqdm.write(f"  event {event_id} failed: {msg[:200]}")
                if "Cannot query field" in msg:
                    tqdm.write("  -> a field name in QUERY doesn't exist in the start.gg schema; "
                               "check it in the API explorer (developer.start.gg) and edit QUERY.")
                    break
                continue
            tmp = RAW / f"{int(event_id)}.json.tmp"
            tmp.write_text(json.dumps(nodes), encoding="utf-8")
            os.replace(tmp, RAW / f"{int(event_id)}.json")

    rows = []
    for event_id in events.event_id:
        f = RAW / f"{int(event_id)}.json"
        if f.exists():
            rows += flatten(int(event_id), json.loads(f.read_text(encoding="utf-8")))
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    have = df.dropna(subset=["seed", "placement"])
    print(f"Wrote {OUT}: {df.event_id.nunique()} events, {len(df):,} entrants "
          f"({len(have):,} with both a seed and a placement)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50, help="number of test events")
    main(p.parse_args().n)
