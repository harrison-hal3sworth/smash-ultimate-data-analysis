"""
04_build_dataset.py

Turn the raw per-event set files from step 03 into analysis tables.
Safe to re-run any time: it rebuilds from whatever files exist so far.

Inputs:
    data/raw/sets/<event_id>.json     (step 03)
    data/interim/events.csv           (step 02: event / tournament metadata)

Outputs (data/processed/):
    sets.csv     one row per set   (who played, who won, how much was reported)
    games.csv    one row per game  (characters, stage, winner) -- games with
                 both characters reported only; this is the matchup dataset
    player_games.csv  games.csv seen from each player's side (2 rows per game):
                 player, char, opp, opp_char, won -> easy groupby for win rates
    audit.txt    reporting coverage summary (for the data-audit checkpoint)
"""

from pathlib import Path
import json
import os

import pandas as pd

ROOT       = Path(__file__).resolve().parents[1]
SETS_DIR   = ROOT / "data" / "raw" / "sets"
EVENTS_CSV = ROOT / "data" / "interim" / "events.csv"
OUT_DIR    = ROOT / "data" / "processed"


# ── helpers ────────────────────────────────────────────────────────────

def _entrant_player(slot: dict) -> tuple:
    """slot -> (entrant_id, player_id, gamer_tag). Any part may be None."""
    ent = (slot or {}).get("entrant") or {}
    parts = ent.get("participants") or []
    player = (parts[0].get("player") or {}) if parts else {}
    return ent.get("id"), player.get("id"), player.get("gamerTag")


def _set_status(s: dict, n_slots: int) -> str:
    score = str(s.get("displayScore") or "")
    if n_slots < 2:
        return "bye_or_incomplete"
    if s.get("winnerId") is None:
        return "unplayed"
    if "DQ" in score.upper() or "-1" in score:
        return "dq"
    return "complete"


def _char_by_entrant(game: dict) -> dict:
    """entrant_id -> character name (first selection per entrant)."""
    out = {}
    for sel in game.get("selections") or []:
        ent = (sel.get("entrant") or {}).get("id")
        char = (sel.get("character") or {}).get("name")
        if ent is not None and char and ent not in out:
            out[ent] = char
    return out


# ── parsing ────────────────────────────────────────────────────────────

def parse_event_file(path: Path) -> tuple[list[dict], list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    event_id = payload.get("event_id") or int(path.stem)
    set_rows, game_rows = [], []

    for s in payload.get("sets") or []:
        slots = [x for x in (s.get("slots") or []) if (x or {}).get("entrant")]
        players = [_entrant_player(x) for x in slots[:2]]
        (e1, p1, t1), (e2, p2, t2) = (players + [(None, None, None)] * 2)[:2]

        games = s.get("games") or []
        status = _set_status(s, len(slots))
        winner_ent = s.get("winnerId")
        winner_player = p1 if winner_ent == e1 else p2 if winner_ent == e2 else None

        n_games_with_chars = 0
        for g in sorted(games, key=lambda g: g.get("orderNum") or 0):
            chars = _char_by_entrant(g)
            c1, c2 = chars.get(e1), chars.get(e2)
            if c1 and c2:
                n_games_with_chars += 1
            g_win_ent = g.get("winnerId")
            game_rows.append({
                "event_id": event_id,
                "set_id": s.get("id"),
                "round": s.get("fullRoundText"),
                "game_num": g.get("orderNum"),
                "p1_id": p1, "p1_tag": t1, "p1_char": c1,
                "p2_id": p2, "p2_tag": t2, "p2_char": c2,
                "stage": (g.get("stage") or {}).get("name"),
                "winner_id": p1 if g_win_ent == e1 else p2 if g_win_ent == e2 else None,
                "p1_score": g.get("entrant1Score"),   # stocks left, as reported
                "p2_score": g.get("entrant2Score"),
            })

        set_rows.append({
            "event_id": event_id,
            "set_id": s.get("id"),
            "round": s.get("fullRoundText"),
            "status": status,
            "p1_id": p1, "p1_tag": t1,
            "p2_id": p2, "p2_tag": t2,
            "winner_id": winner_player,
            "display_score": s.get("displayScore"),
            "total_games": s.get("totalGames"),
            "n_games_reported": len(games),
            "n_games_with_chars": n_games_with_chars,
            "has_games": len(games) > 0,
            "has_chars": n_games_with_chars > 0,
        })
    return set_rows, game_rows


def build(sets_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    files = sorted(p for p in sets_dir.glob("*.json"))
    all_sets, all_games, bad = [], [], []
    for i, f in enumerate(files, 1):
        try:
            s, g = parse_event_file(f)
            all_sets.extend(s)
            all_games.extend(g)
        except Exception as e:                # report and keep going
            bad.append(f"{f.name}: {type(e).__name__}: {e}")
        if i % 500 == 0:
            print(f"  parsed {i:,}/{len(files):,} files", flush=True)
    sets_df, games_df = pd.DataFrame(all_sets), pd.DataFrame(all_games)
    for df in (sets_df, games_df):        # keep IDs as integers even when some are missing
        for col in ("event_id", "set_id", "p1_id", "p2_id", "winner_id", "game_num",
                    "total_games", "p1_score", "p2_score"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return sets_df, games_df, bad


def add_event_info(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    cols = ["event_id", "tournament_id", "tournament_name", "event_name",
            "date", "state", "num_entrants"]
    cols = [c for c in cols if c in events.columns]
    return df.merge(events[cols], on="event_id", how="left")


def player_view(games: pd.DataFrame) -> pd.DataFrame:
    """Each game twice, once from each player's side."""
    keep = [c for c in ["event_id", "tournament_name", "date", "num_entrants",
                        "set_id", "round", "game_num", "stage"] if c in games.columns]
    a = games[keep].assign(player_id=games.p1_id, player_tag=games.p1_tag, char=games.p1_char,
                           opp_id=games.p2_id, opp_tag=games.p2_tag, opp_char=games.p2_char,
                           won=games.winner_id == games.p1_id)
    b = games[keep].assign(player_id=games.p2_id, player_tag=games.p2_tag, char=games.p2_char,
                           opp_id=games.p1_id, opp_tag=games.p1_tag, opp_char=games.p1_char,
                           won=games.winner_id == games.p2_id)
    return pd.concat([a, b], ignore_index=True)


# ── audit ──────────────────────────────────────────────────────────────

def audit(sets: pd.DataFrame, games: pd.DataFrame, matchup_games: pd.DataFrame,
          n_files: int, bad: list[str]) -> str:
    played = sets[sets.status == "complete"]
    lines = [
        f"Event files parsed: {n_files:,}  (failed: {len(bad)})",
        f"Sets: {len(sets):,}  | completed: {len(played):,}",
        f"  status: " + ", ".join(f"{k} {v:,}" for k, v in sets.status.value_counts().items()),
        f"Completed sets with any game data:      {played.has_games.mean():.1%}",
        f"Completed sets with character data:     {played.has_chars.mean():.1%}",
        f"Games reported: {len(games):,}  | with both characters: {len(matchup_games):,}",
        f"Distinct characters seen: {pd.concat([matchup_games.p1_char, matchup_games.p2_char]).nunique()}",
        f"Distinct matchups (unordered pairs): "
        f"{matchup_games.apply(lambda r: tuple(sorted((r.p1_char, r.p2_char))), axis=1).nunique() if len(matchup_games) else 0:,}",
    ]
    if "num_entrants" in played.columns and played["num_entrants"].notna().any():
        bins = pd.cut(played["num_entrants"], [0, 63, 127, 255, 1e9],
                      labels=["32-63", "64-127", "128-255", "256+"])
        cov = played.groupby(bins, observed=False).agg(sets=("set_id", "size"),
                                                       pct_with_chars=("has_chars", "mean"))
        cov["pct_with_chars"] = (cov["pct_with_chars"] * 100).round(1)
        lines += ["", "Character reporting by event size (completed sets):", cov.to_string()]
    if "date" in played.columns and played["date"].notna().any():
        yr = pd.to_datetime(played["date"]).dt.year
        cov = played.groupby(yr).agg(sets=("set_id", "size"), pct_with_chars=("has_chars", "mean"))
        cov["pct_with_chars"] = (cov["pct_with_chars"] * 100).round(1)
        lines += ["", "Character reporting by year (completed sets):", cov.to_string()]
    if bad:
        lines += ["", "Files that failed to parse:"] + [f"  {b}" for b in bad[:20]]
    return "\n".join(lines)


# ── main ───────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_files = sum(1 for n in os.listdir(SETS_DIR) if n.endswith(".json"))
    print(f"Parsing {n_files:,} event files from {SETS_DIR} ...", flush=True)

    sets, games, bad = build(SETS_DIR)
    if sets.empty:
        print("No sets found. Has step 03 saved any events yet?")
        return

    events = pd.read_csv(EVENTS_CSV)
    sets = add_event_info(sets, events)
    games = add_event_info(games, events) if not games.empty else games

    # the matchup dataset: completed sets, both characters known, winner known
    complete_ids = set(sets.loc[sets.status == "complete", "set_id"])
    matchup = games[games.set_id.isin(complete_ids)
                    & games.p1_char.notna() & games.p2_char.notna()
                    & games.winner_id.notna()].copy() if not games.empty else games

    sets.to_csv(OUT_DIR / "sets.csv", index=False)
    matchup.to_csv(OUT_DIR / "games.csv", index=False)
    if not matchup.empty:
        player_view(matchup).to_csv(OUT_DIR / "player_games.csv", index=False)

    report = audit(sets, games, matchup, n_files, bad)
    (OUT_DIR / "audit.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\nWrote sets.csv ({len(sets):,}), games.csv ({len(matchup):,}), "
          f"player_games.csv ({2 * len(matchup):,}), audit.txt -> {OUT_DIR}")


if __name__ == "__main__":
    main()