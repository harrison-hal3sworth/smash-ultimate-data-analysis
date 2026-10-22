"""
clean.py  —  cleaning + summary functions for the Smash Ultimate analysis.

Used by EDA/cleaning.ipynb (and later the Quarto report), so the cleaning
rules live in one place. Re-running on new data from step 04 needs no edits.

    from clean import load_clean
    data = load_clean(ROOT)                 # dict with sets_all, sets, games, pg, log

Command line (writes cleaned tables to data/clean/):
    python src/clean.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ── configuration ──────────────────────────────────────────────────────

EXCLUDE_CHARS = {"Random Character"}     # not a real character pick
MIN_GAMES = 100                          # default "sufficient observations" cutoff per matchup
Z = 1.96                                 # 95% intervals
SIZE_BINS = [0, 63, 127, 255, 511, 1023, np.inf]
SIZE_LABELS = ["32-63", "64-127", "128-255", "256-511", "512-1023", "1024+"]


def find_root(start: Path | None = None) -> Path:
    """Project root = first folder upward that contains src/ and data/."""
    p = (start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / "src").is_dir() and (cand / "data").is_dir():
            return cand
    raise FileNotFoundError("Could not find project root (folder with src/ and data/)")


# ── loading ────────────────────────────────────────────────────────────

def load_processed(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = root / "data" / "processed"
    sets = pd.read_csv(d / "sets.csv", low_memory=False)
    games = pd.read_csv(d / "games.csv", low_memory=False)
    return sets, games


def _size_bucket(n: pd.Series) -> pd.Series:
    return pd.cut(n, SIZE_BINS, labels=SIZE_LABELS)


# ── cleaning ───────────────────────────────────────────────────────────

def clean_sets(sets: pd.DataFrame, log: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (sets_all, sets_complete).
    sets_all keeps every real set (for coverage / audit); sets_complete = played sets."""
    def step(df, msg):
        log.append(("sets", msg, len(df)))
        return df

    df = step(sets, "raw rows")
    df = step(df[df.set_id.notna()], "drop placeholder sets (no set_id)")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year.astype("Int64")
    df["size_bucket"] = _size_bucket(df["num_entrants"])
    for c in ("set_id", "event_id", "p1_id", "p2_id", "winner_id", "tournament_id"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    sets_all = df

    done = df[df.status == "complete"]
    done = step(done[done.winner_id.notna() & done.p1_id.notna() & done.p2_id.notna()],
                "keep completed sets with both players + winner")
    return sets_all, done.copy()


def clean_games(games: pd.DataFrame, sets_complete: pd.DataFrame, log: list) -> pd.DataFrame:
    def step(df, msg):
        log.append(("games", msg, len(df)))
        return df

    df = step(games, "raw rows (games with both characters)")
    df = step(df[df.set_id.isin(sets_complete.set_id)], "keep games from completed sets")
    df = step(df.drop_duplicates(["set_id", "game_num"]), "drop duplicate (set_id, game_num)")
    df = step(df[~df.p1_char.isin(EXCLUDE_CHARS) & ~df.p2_char.isin(EXCLUDE_CHARS)],
              f"drop {', '.join(sorted(EXCLUDE_CHARS))}")
    df = step(df[(df.winner_id == df.p1_id) | (df.winner_id == df.p2_id)],
              "keep games whose winner is one of the two players")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year.astype("Int64")
    df["size_bucket"] = _size_bucket(df["num_entrants"])
    df["is_mirror"] = df.p1_char == df.p2_char
    swap = df.p1_char > df.p2_char
    df["char_lo"] = df.p1_char.where(~swap, df.p2_char)          # unordered matchup key
    df["char_hi"] = df.p2_char.where(~swap, df.p1_char)
    log.append(("games", "  of which mirror matches (kept, flagged is_mirror)", int(df.is_mirror.sum())))
    return df


def player_view(games: pd.DataFrame) -> pd.DataFrame:
    """Each game twice, once from each player's side. Rebuilt from cleaned games,
    so it always matches games exactly (player_games.csv is not needed)."""
    keep = ["event_id", "tournament_name", "date", "year", "num_entrants", "size_bucket",
            "set_id", "round", "game_num", "stage", "is_mirror"]
    keep = [c for c in keep if c in games.columns]
    a = games[keep].assign(player_id=games.p1_id, player_tag=games.p1_tag, char=games.p1_char,
                           opp_id=games.p2_id, opp_tag=games.p2_tag, opp_char=games.p2_char,
                           won=games.winner_id == games.p1_id, side="p1")
    b = games[keep].assign(player_id=games.p2_id, player_tag=games.p2_tag, char=games.p2_char,
                           opp_id=games.p1_id, opp_tag=games.p1_tag, opp_char=games.p1_char,
                           won=games.winner_id == games.p2_id, side="p2")
    pg = pd.concat([a, b], ignore_index=True)
    pg["won"] = pg["won"].astype(bool)
    return pg


def load_clean(root: Path | None = None) -> dict:
    """Load step-04 output and apply every cleaning rule. Returns a dict:
    sets_all, sets, games, pg (player view), log (DataFrame of row counts per step)."""
    root = root or find_root()
    log: list = []
    sets_raw, games_raw = load_processed(root)
    sets_all, sets_c = clean_sets(sets_raw, log)
    games_c = clean_games(games_raw, sets_c, log)
    pg = player_view(games_c)
    return {
        "sets_all": sets_all,
        "sets": sets_c,
        "games": games_c,
        "pg": pg,
        "log": pd.DataFrame(log, columns=["table", "step", "rows"]),
    }


# ── summaries ──────────────────────────────────────────────────────────

def wilson(wins, n, z: float = Z):
    """Wilson score interval for a proportion (works on arrays)."""
    wins, n = np.asarray(wins, float), np.asarray(n, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = wins / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def coverage(sets_all: pd.DataFrame, by: str) -> pd.DataFrame:
    """Share of completed sets with game data / character data, grouped by `by`."""
    c = sets_all[sets_all.status == "complete"]
    out = c.groupby(by, observed=True).agg(
        sets=("set_id", "size"),
        events=("event_id", "nunique"),
        pct_with_games=("has_games", "mean"),
        pct_with_chars=("has_chars", "mean"),
    )
    out[["pct_with_games", "pct_with_chars"]] *= 100
    return out.round(1)


def event_coverage(sets_all: pd.DataFrame) -> pd.DataFrame:
    """One row per event: how much of it reported characters."""
    c = sets_all[sets_all.status == "complete"]
    return c.groupby("event_id").agg(
        tournament_name=("tournament_name", "first"),
        date=("date", "first"),
        num_entrants=("num_entrants", "first"),
        sets=("set_id", "size"),
        pct_with_chars=("has_chars", "mean"),
    ).assign(pct_with_chars=lambda d: (d.pct_with_chars * 100).round(1))


def char_stats(pg: pd.DataFrame) -> pd.DataFrame:
    """Per character: games played (usage) and win rate excluding mirrors, with 95% CI."""
    usage = pg.groupby("char").size().rename("games_all")
    nm = pg[~pg.is_mirror]
    w = nm.groupby("char").agg(games=("won", "size"), wins=("won", "sum"),
                               players=("player_id", "nunique"))
    out = usage.to_frame().join(w)
    out["usage_pct"] = out.games_all / out.games_all.sum() * 100
    out["win_rate"] = out.wins / out.games
    out["ci_low"], out["ci_high"] = wilson(out.wins, out.games)
    return out.sort_values("games_all", ascending=False)


def matchup_stats(pg: pd.DataFrame, min_games: int = MIN_GAMES,
                  bootstrap: int = 0, seed: int = 0) -> pd.DataFrame:
    """Directed matchups (char vs opp_char), mirrors excluded.
    Wilson CI treats games as independent; `bootstrap=N` adds a cluster bootstrap
    that resamples whole sets (games in a set share players, so this CI is wider)."""
    nm = pg[~pg.is_mirror]
    out = nm.groupby(["char", "opp_char"]).agg(
        games=("won", "size"), wins=("won", "sum"), sets=("set_id", "nunique"),
        players=("player_id", "nunique"))
    out["win_rate"] = out.wins / out.games
    out["ci_low"], out["ci_high"] = wilson(out.wins, out.games)
    out["sufficient"] = out.games >= min_games

    if bootstrap:
        rng = np.random.default_rng(seed)
        per_set = (nm.groupby(["char", "opp_char", "set_id"])
                     .won.agg(["sum", "size"]).reset_index())
        lo, hi = {}, {}
        for key, grp in per_set[per_set.set_index(["char", "opp_char"]).index
                                .isin(out.index[out.sufficient])].groupby(["char", "opp_char"]):
            w, n = grp["sum"].to_numpy(float), grp["size"].to_numpy(float)
            idx = rng.integers(0, len(w), size=(bootstrap, len(w)))
            rates = w[idx].sum(1) / n[idx].sum(1)
            lo[key], hi[key] = np.percentile(rates, [2.5, 97.5])
        out["boot_low"] = pd.Series(lo)
        out["boot_high"] = pd.Series(hi)
    return out.sort_values("games", ascending=False)


def undirected(m: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per matchup (char < opp_char alphabetically): A-vs-B and B-vs-A are mirror images."""
    m = m.reset_index()
    return m[m.char < m.opp_char].set_index(["char", "opp_char"])


def player_concentration(pg: pd.DataFrame) -> pd.Series:
    """How concentrated games are among the most active players (skill-confounding check)."""
    per = pg.groupby("player_id").size().sort_values(ascending=False)
    n = len(per)
    return pd.Series({
        "players": n,
        "median_games_per_player": float(per.median()),
        "share_top_1pct": per.iloc[: max(1, n // 100)].sum() / per.sum() * 100,
        "share_top_10pct": per.iloc[: max(1, n // 10)].sum() / per.sum() * 100,
    }).round(1)


def headline(d: dict, min_games: int = MIN_GAMES) -> pd.Series:
    """Key numbers for the data-audit checkpoint."""
    s_all, s, g, pg = d["sets_all"], d["sets"], d["games"], d["pg"]
    m = undirected(matchup_stats(pg, min_games))
    return pd.Series({
        "events": s_all.event_id.nunique(),
        "min_entrants": int(s_all.num_entrants.min()),
        "date_range": f"{s_all.date.min():%Y-%m-%d} to {s_all.date.max():%Y-%m-%d}",
        "completed_sets": len(s),
        "pct_sets_with_chars": round(s.has_chars.mean() * 100, 1),
        "events_reporting_chars": g.event_id.nunique(),
        "clean_games": len(g),
        "non_mirror_games": int((~g.is_mirror).sum()),
        "players_in_games": pd.concat([g.p1_id, g.p2_id]).nunique(),
        "characters": pd.concat([g.p1_char, g.p2_char]).nunique(),
        "matchups_observed": len(m),
        f"matchups_>={min_games}_games": int(m.sufficient.sum()),
        "pct_games_with_stage": round(g.stage.notna().mean() * 100, 1),
    })


# ── command line: write cleaned tables ─────────────────────────────────

if __name__ == "__main__":
    root = find_root(Path(__file__).parent)
    d = load_clean(root)
    out = root / "data" / "clean"
    out.mkdir(parents=True, exist_ok=True)
    d["sets"].to_csv(out / "sets_clean.csv", index=False)
    d["games"].to_csv(out / "games_clean.csv", index=False)
    d["log"].to_csv(out / "cleaning_log.csv", index=False)
    print(d["log"].to_string(index=False))
    print()
    print(headline(d).to_string())
    print(f"\nWrote cleaned tables to {out}")
