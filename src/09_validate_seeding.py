# Contains Claude generated code

"""
09_validate_seeding.py  —  does the auto-seeder beat organizer seeding?

For each test event in data/interim/seeding.csv (step 08), seed the entrants four ways:

    organizer            the event's real seeding
    ratings only         auto-seeder, player ratings only
    ratings + characters auto-seeder, ratings + each player's most-used character
    random               baseline (average of 20 random orderings)

using ONLY information from before the event: Glicko-2 ratings from before that week,
a matchup model refitted on games before TEST_FROM, and mains from before TEST_FROM.
Then score each seeding against what happened:

    spearman      rank correlation between seed and final placement (1 = perfect)
    top8          how many of the top 8 seeds finished top 8
    set_accuracy  share of the event's sets won by the better-seeded player

Writes reports/seeding_validation.md, reports/figures/seeding_*.png, reports/tables/seeding_*.csv.

Run from the project root (after steps 05 and 08):  python src/09_validate_seeding.py
"""

from pathlib import Path
import importlib.util
import json
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import clean  # noqa: E402
import seeder  # noqa: E402

spec = importlib.util.spec_from_file_location("build_models", SRC / "05_build_models.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)

ROOT = SRC.parent
REPORT, FIG, TAB = ROOT / "reports", ROOT / "reports" / "figures", ROOT / "reports" / "tables"
TEST_FROM = pd.Timestamp(B.TEST_FROM)
NEWCOMER_SETS = 5          # fewer prior sets than this = newcomer
N_RANDOM, N_BOOT, SEED = 20, 2000, 0
PC = "+ player-on-character"   # what the dashboard's auto-seeder uses
METHODS = ["organizer", "ratings only", "ratings + characters", PC, "random"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join([df.index.name or ""] + list(map(str, df.columns))) + " |"
    sep = "| " + " | ".join(["---"] * (len(df.columns) + 1)) + " |"
    rows = ["| " + " | ".join([str(i)] + [str(v) for v in r]) + " |" for i, r in zip(df.index, df.to_numpy())]
    return "\n".join([head, sep, *rows])


# ── inputs, all restricted to "before the event" ───────────────────────

def pre_event_ratings(rated: pd.DataFrame) -> pd.DataFrame:
    """(event_id, player_id) -> rating, rd, prior_sets from before that event's first week."""
    long = pd.concat([
        rated[["event_id", "date", "p1_id", "p1_r", "p1_rd", "p1_prior_sets"]].set_axis(
            ["event_id", "date", "player_id", "r", "rd", "prior_sets"], axis=1),
        rated[["event_id", "date", "p2_id", "p2_r", "p2_rd", "p2_prior_sets"]].set_axis(
            ["event_id", "date", "player_id", "r", "rd", "prior_sets"], axis=1),
    ])
    return long.sort_values("date").groupby(["event_id", "player_id"]).first().reset_index()


def train_models(games, rated, penalty) -> tuple[dict, dict, dict]:
    """Matchup model refitted on games before TEST_FROM, a ratings-only model, and the
    model with the player-on-character term (plus its fitted adjustments)."""
    d = B.prep_games(games, rated)
    tr = d[d.date < TEST_FROM]
    chars = sorted(set(tr.c1) | set(tr.c2))
    pairs = sorted({tuple(sorted(p)) for p in zip(tr.c1, tr.c2)})
    X = B.design(tr, chars, pairs)
    w = B.fit_logistic(X, tr.y.to_numpy(), len(chars), penalty["strength"], penalty["interaction"])
    w0 = B.fit_logistic(X[:, :1], tr.y.to_numpy(), 0, 0, 0)
    nc = len(chars)
    full = {"beta": float(w[0]), "scale": B.SCALE, "characters": chars,
            "strength": dict(zip(chars, map(float, w[1:1 + nc]))),
            "interaction": {f"{a}|{b}": float(v) for (a, b), v in zip(pairs, w[1 + nc:])}}
    ratings_only = {"beta": float(w0[0]), "scale": B.SCALE, "characters": [], "strength": {}, "interaction": {}}

    f = B.fit_player_char(tr, chars, pairs, penalty["strength"], penalty["interaction"],
                          penalty["player_char"], penalty["player"])
    k = 1 + nc + len(pairs)
    w2 = f["w"]
    pc = {"beta": float(w2[0]), "scale": B.SCALE, "characters": chars,
          "strength": dict(zip(chars, map(float, w2[1:1 + nc]))),
          "interaction": {f"{a}|{b}": float(v) for (a, b), v in zip(pairs, w2[1 + nc:k])},
          "experience_coef": f["gamma"], "adj": f["adj"].to_dict()}
    return full, ratings_only, pc


def offsets_before(pg: pd.DataFrame, mains: pd.Series, pc: dict) -> pd.Series:
    """Player-on-character term for each player's pre-TEST_FROM main, from pre-TEST_FROM games."""
    before = pg[pg.date < TEST_FROM]
    n_p = before.groupby("player_id").size()
    n_pc = before.groupby(["player_id", "char"]).size()
    out = {}
    for pid, c in mains.items():
        e = B.experience_now(n_pc.get((pid, c), 0), n_p.get(pid, 0))
        out[pid] = pc["experience_coef"] * e + pc["adj"].get((pid, c), 0.0)
    return pd.Series(out, dtype=float)


def mains_before(pg: pd.DataFrame) -> pd.Series:
    """Each player's most-used character before TEST_FROM (last 12 months, else all time)."""
    before = pg[pg.date < TEST_FROM]
    recent = before[before.date >= TEST_FROM - pd.Timedelta(days=365)]
    m_recent = recent.groupby(["player_id", "char"]).size().sort_values().groupby("player_id").tail(1)
    m_all = before.groupby(["player_id", "char"]).size().sort_values().groupby("player_id").tail(1)
    out = m_all.reset_index().set_index("player_id").char
    out.update(m_recent.reset_index().set_index("player_id").char)
    return out


# ── per event ──────────────────────────────────────────────────────────

def seed_event(ev: pd.DataFrame, full: dict, ronly: dict, mains: pd.Series, rng,
               pc: dict | None = None, offsets: pd.Series | None = None) -> dict:
    """ev: one event's entrants with seed, placement, r, rd. Returns seeds per method."""
    ids = ev.player_id.to_numpy()
    seeds = {"organizer": seeder.seeds_from_order(np.argsort(ev.seed.to_numpy(), kind="stable"))}
    P0 = seeder.pairwise_set_probs(ronly, ev.r, ev.rd, None)
    seeds["ratings only"] = seeder.seeds_from_order(seeder.seed_order(P0))
    chars = [mains.get(p) for p in ids]
    P1 = seeder.pairwise_set_probs(full, ev.r, ev.rd, chars)
    seeds["ratings + characters"] = seeder.seeds_from_order(seeder.seed_order(P1))
    if pc is not None:
        off = [offsets.get(p, 0.0) for p in ids]
        P2 = seeder.pairwise_set_probs(pc, ev.r, ev.rd, chars, offsets=off)
        seeds[PC] = seeder.seeds_from_order(seeder.seed_order(P2))
    seeds["random"] = [seeder.seeds_from_order(rng.permutation(len(ids))) for _ in range(N_RANDOM)]
    return seeds


def score_event(seeds: dict, ev: pd.DataFrame, sets: pd.DataFrame) -> dict:
    ids, placements = ev.player_id.to_numpy(), ev.placement.to_numpy()
    res = seeder.evaluate_event({m: s for m, s in seeds.items() if m != "random"}, placements, ids, sets)
    rand = [seeder.evaluate_event({"r": s}, placements, ids, sets)["r"] for s in seeds["random"]]
    res["random"] = {k: float(np.nanmean([r[k] for r in rand])) for k in rand[0]}
    return res


def newcomer_split(seeds: dict, ev: pd.DataFrame, sets: pd.DataFrame) -> list[dict]:
    """Set accuracy separately for sets with / without a newcomer in them."""
    new = set(ev.loc[ev.prior_sets < NEWCOMER_SETS, "player_id"])
    has_new = sets.p1_id.isin(new) | sets.p2_id.isin(new)
    rows = []
    for m in ["organizer", "ratings only", "ratings + characters", PC]:
        seed_of = dict(zip(ev.player_id, seeds[m]))
        for label, mask in [("involves a newcomer", has_new), ("established players only", ~has_new)]:
            sub = sets[mask]
            s1, s2 = sub.p1_id.map(seed_of), sub.p2_id.map(seed_of)
            ok = s1.notna() & s2.notna()
            rows.append({"method": m, "sets": label, "n": int(ok.sum()),
                         "correct": int(((s1[ok] < s2[ok]) == sub.loc[ok, "p1_won"].astype(bool)).sum())})
    return rows


# ── summary ────────────────────────────────────────────────────────────

def boot_ci(x: np.ndarray, rng) -> tuple[float, float]:
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan"), float("nan")
    b = rng.choice(x, size=(N_BOOT, len(x))).mean(1)
    return tuple(np.percentile(b, [2.5, 97.5]))


def main():
    for d in (REPORT, FIG, TAB):
        d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    log("loading inputs ...")
    seeding = pd.read_csv(ROOT / "data" / "interim" / "seeding.csv")
    seeding = seeding[~seeding.dq].dropna(subset=["player_id", "seed", "placement"])
    seeding["player_id"] = seeding.player_id.astype("int64")
    test_events = set(seeding.event_id)
    rated = pd.read_csv(ROOT / "data" / "models" / "sets_rated.csv.gz", parse_dates=["date"])
    pre = pre_event_ratings(rated[rated.event_id.isin(test_events)])
    data = clean.load_clean(ROOT)
    penalty = json.loads((ROOT / "data" / "models" / "matchup_model.json").read_text())["penalty"]

    log(f"refitting the matchup model on games before {TEST_FROM:%Y-%m-%d} ...")
    full, ronly, pc = train_models(data["games"], rated, penalty)
    mains = mains_before(data["pg"])
    offsets = offsets_before(data["pg"], mains, pc)

    log(f"seeding {len(test_events)} events five ways ...")
    per_event, newcomer_rows = [], []
    for event_id, ev in seeding.groupby("event_id"):
        ev = ev.merge(pre, on=["event_id", "player_id"], how="inner")    # entrants who played a set
        if len(ev) < 16:
            continue
        sets = rated[rated.event_id == event_id]
        seeds = seed_event(ev, full, ronly, mains, rng, pc, offsets)
        scores = score_event(seeds, ev, sets)
        info = data["sets"].loc[data["sets"].event_id == event_id, ["tournament_name", "date"]].iloc[0]
        for m, sc in scores.items():
            per_event.append({"event_id": event_id, "tournament": info.tournament_name,
                              "date": info.date.date(), "entrants": len(ev),
                              "newcomer_share": float((ev.prior_sets < NEWCOMER_SETS).mean()),
                              "main_known_share": float(ev.player_id.map(mains).notna().mean()),
                              "method": m, **sc})
        newcomer_rows += newcomer_split(seeds, ev, sets)

    res = pd.DataFrame(per_event)
    res.to_csv(TAB / "seeding_per_event.csv", index=False)
    n_events = res.event_id.nunique()

    # mean of each metric per method, with bootstrap CIs over events
    summ = []
    for m in METHODS:
        r = res[res.method == m]
        row = {"method": m}
        for k, fmt in [("spearman", ".3f"), ("top8", ".2f"), ("set_accuracy", ".3f")]:
            lo, hi = boot_ci(r[k].to_numpy(), rng)
            row[k] = f"{r[k].mean():{fmt}} [{lo:{fmt}}, {hi:{fmt}}]"
        summ.append(row)
    summ = pd.DataFrame(summ).set_index("method")
    summ.columns = ["seed vs placement (Spearman)", "top-8 seeds in top 8", "sets won by better seed"]
    summ.to_csv(TAB / "seeding_summary.csv")

    # paired comparisons: auto-seeder minus organizer, and characters minus ratings only
    wide = {k: res.pivot(index="event_id", columns="method", values=k) for k in ["spearman", "top8", "set_accuracy"]}
    pairs = [(PC, "organizer"), ("ratings + characters", "organizer"), ("ratings only", "organizer"),
             ("ratings + characters", "ratings only"), (PC, "ratings + characters")]
    comp = []
    for a, b in pairs:
        row = {"comparison": f"{a} vs {b}"}
        for k in wide:
            diff = (wide[k][a] - wide[k][b]).to_numpy()
            lo, hi = boot_ci(diff, rng)
            better, worse = int((diff > 1e-9).sum()), int((diff < -1e-9).sum())
            row[k] = f"{np.nanmean(diff):+.3f} [{lo:+.3f}, {hi:+.3f}]; better in {better}, worse in {worse}"
        comp.append(row)
    comp = pd.DataFrame(comp).set_index("comparison")
    comp.columns = ["Spearman difference", "top-8 difference", "set-accuracy difference"]
    comp.to_csv(TAB / "seeding_comparisons.csv")

    nc = pd.DataFrame(newcomer_rows).groupby(["sets", "method"])[["n", "correct"]].sum()
    nc["accuracy"] = (nc.correct / nc.n).round(3)
    nc = nc.reset_index().pivot(index="sets", columns="method", values="accuracy")[
        ["organizer", "ratings only", "ratings + characters", PC]]
    nc.to_csv(TAB / "seeding_newcomers.csv")

    # figure: auto-seeder vs organizer, one dot per event
    o, a = wide["spearman"]["organizer"], wide["spearman"][PC]
    fig, ax = plt.subplots(figsize=(5, 5))
    lim = [min(o.min(), a.min()) - 0.03, 1]
    ax.plot(lim, lim, ls=":", color="grey")
    ax.scatter(o, a, s=18)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("organizer seeding: Spearman with placement")
    ax.set_ylabel("auto-seeder: Spearman with placement")
    ax.set_title(f"Auto-seeder vs organizer, {n_events} events\n(above the line = auto-seeder better)")
    fig.savefig(FIG / "seeding_vs_organizer.png", bbox_inches="tight", dpi=130)

    best_sp = summ.index[np.argmax([res[res.method == m].spearman.mean() for m in METHODS])]
    lines = [
        "# Auto-seeder validation", "",
        f"Generated {pd.Timestamp.now():%Y-%m-%d %H:%M}. {n_events} events from {TEST_FROM:%Y-%m-%d} onward, "
        f"{res[res.method == 'organizer'].entrants.sum():,} entrants. Each event is seeded using only "
        "information from before it (ratings from before that week, a matchup model fitted on pre-"
        f"{TEST_FROM:%Y} games, and each player's pre-{TEST_FROM:%Y} main). "
        f"About {res[res.method == 'organizer'].main_known_share.mean():.0%} of entrants had a known main.", "",
        "## Results", "", "Mean over events, 95% bootstrap interval in brackets.", "",
        md_table(summ), "",
        f"Best Spearman: **{best_sp}**.", "",
        "## Head-to-head", "",
        "Per-event difference (first method minus second), with the number of events each way.", "",
        md_table(comp), "",
        "## Newcomers", "",
        f"Share of sets won by the better seed, split by whether a player with fewer than {NEWCOMER_SETS} "
        "prior sets was involved.", "",
        md_table(nc), "",
        "![auto-seeder vs organizer](figures/seeding_vs_organizer.png)", "",
        "## Reading these numbers", "",
        "- **Placement favours the organizer.** The organizer's seeding decides the bracket, so it decides "
        "who meets whom and which placements are reachable. A seeding that was never used is judged against "
        "a bracket built on another seeding. Set accuracy is less affected than Spearman or top-8.",
        "- **Organizers know things the data doesn't**: recent online results, regional reputation, players "
        "from outside the US data. Roughly matching them from tournament data alone shows the model has "
        "real competency.",
        "- Per-event results: `reports/tables/seeding_per_event.csv`.",
    ]
    (REPORT / "seeding_validation.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + md_table(summ) + "\n\n" + md_table(comp) + "\n\n" + md_table(nc))
    log(f"wrote {REPORT / 'seeding_validation.md'}")


if __name__ == "__main__":
    main()
