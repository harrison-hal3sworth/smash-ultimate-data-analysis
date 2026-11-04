"""
05_build_models.py  —  player table, Glicko-2 ratings, character usage, matchup model.

Reads the step-04 output (data/processed/) through clean.py and writes everything
the dashboard and the auto-seeder need to data/models/:

    players.csv          one row per player: current tag, past tags, state, activity, rank
    ratings_current.csv  latest Glicko-2 rating / RD per player + rank among active players
    sets_rated.csv.gz    every completed set with BOTH players' ratings *before* that week
    usage.csv            character usage per player (last 12 months, else all time)
    matchup_model.json   fitted model: rating coefficient, character strength, matchup terms
    char_strength.csv    characters ranked by skill-adjusted strength (data-driven tier list)
    matchup_matrix.csv   P(row character beats column character), equal-skill players, per game
    model_report.txt     validation numbers (train before 2026, test on 2026)

Run from the project root:  python src/05_build_models.py
"""

from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clean  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]      # project root (src/..)
OUT = ROOT / "data" / "models"

# Glicko-2 settings
SCALE = 173.7178          # converts between the 1500/350 scale and Glicko-2's internal scale
R0, RD0, VOL0 = 1500.0, 350.0, 0.06
TAU = 0.5                 # how fast volatility can change
ACTIVE_DAYS = 365         # "current" = played within the last 12 months
MIN_SETS_RANKED = 10      # sets in that window needed to get a rank
MAX_RD_RANKED = 100       # ...and the rating must be settled (RD at most this) to get a rank

TEST_FROM = "2026-01-01"  # model validation: train before, test on/after


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── 1. player table ────────────────────────────────────────────────────

def build_players(sets_all: pd.DataFrame) -> pd.DataFrame:
    """One row per start.gg player id: latest tag, other tags, home state, activity."""
    long = pd.concat([
        sets_all[["p1_id", "p1_tag", "date", "state", "tournament_name"]]
            .set_axis(["player_id", "tag", "date", "state", "tournament_name"], axis=1),
        sets_all[["p2_id", "p2_tag", "date", "state", "tournament_name"]]
            .set_axis(["player_id", "tag", "date", "state", "tournament_name"], axis=1),
    ]).dropna(subset=["player_id"])
    long = long.sort_values("date")

    end = long.date.max()
    recent = long[long.date >= end - pd.Timedelta(days=ACTIVE_DAYS)]

    g = long.groupby("player_id")
    players = pd.DataFrame({
        "tag": g.tag.last(),                                       # most recent tag
        "first_date": g.date.min().dt.date,
        "last_date": g.date.max().dt.date,
        "last_event": g.tournament_name.last(),
        "sets": g.size(),
        "state": g.state.agg(lambda s: s.mode().iat[0] if s.notna().any() else None),
    })
    tags = long.dropna(subset=["tag"]).groupby("player_id").tag.agg(lambda s: list(dict.fromkeys(s)))
    players["other_tags"] = [
        "; ".join(t for t in tags.get(pid, []) if t != cur) for pid, cur in players.tag.items()
    ]
    players["sets_last_12m"] = recent.groupby("player_id").size().reindex(players.index).fillna(0).astype(int)
    players.index = players.index.astype("int64")
    return players


# ── 2. Glicko-2 ratings ────────────────────────────────────────────────

def _volatility(sigma, phi, v, delta, tau=TAU):
    """Glicko-2 step 5 (Illinois algorithm), vectorized over players."""
    a = np.log(sigma ** 2)
    phi2 = phi ** 2

    def f(x):
        ex = np.exp(x)
        return ex * (delta ** 2 - phi2 - v - ex) / (2 * (phi2 + v + ex) ** 2) - (x - a) / tau ** 2

    A = a.copy()
    big = delta ** 2 > phi2 + v
    B = np.where(big, np.log(np.maximum(delta ** 2 - phi2 - v, 1e-300)), a - tau)
    need = ~big & (f(a - tau) < 0)
    k = 1
    while need.any() and k < 100:
        k += 1
        B = np.where(need, a - k * tau, B)
        need = need & (f(a - k * tau) < 0)

    fA, fB = f(A), f(B)
    for _ in range(100):
        todo = np.abs(B - A) > 1e-6
        if not todo.any():
            break
        C = A + (A - B) * fA / (fB - fA)
        fC = f(C)
        swap = fC * fB <= 0
        A = np.where(todo & swap, B, A)
        fA = np.where(todo & swap, fB, np.where(todo, fA / 2, fA))
        B = np.where(todo, C, B)
        fB = np.where(todo, fC, fB)
    return np.exp(A / 2)


def rate_period(mu, phi, sigma, a, b, sa, tau=TAU):
    """One Glicko-2 rating period (steps 3-8 of Glickman's paper), vectorized.
    mu/phi/sigma: arrays over all players (internal scale), RDs already inflated for
    idleness. a, b: player indices of each set; sa: 1 if a won. Returns the indices
    of players who played and their new (mu, phi, sigma)."""
    n = len(mu)
    u = np.unique(np.concatenate([a, b]))
    P = np.concatenate([a, b]); Q = np.concatenate([b, a]); S = np.concatenate([sa, 1 - sa])
    g = 1 / np.sqrt(1 + 3 * phi[Q] ** 2 / np.pi ** 2)
    E = 1 / (1 + np.exp(-g * (mu[P] - mu[Q])))
    v = 1 / np.bincount(P, g ** 2 * E * (1 - E), minlength=n)[u]
    dsum = np.bincount(P, g * (S - E), minlength=n)[u]
    sig_new = _volatility(sigma[u], phi[u], v, v * dsum, tau)
    phi_star = np.sqrt(phi[u] ** 2 + sig_new ** 2)
    phi_new = 1 / np.sqrt(1 / phi_star ** 2 + 1 / v)
    return u, mu[u] + phi_new ** 2 * dsum, phi_new, sig_new


def glicko2(sets: pd.DataFrame, tau: float = TAU) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weekly Glicko-2 over completed sets in date order.
    Returns (sets with pre-week ratings attached, current ratings per player)."""
    s = sets[["set_id", "event_id", "date", "p1_id", "p2_id", "winner_id"]].dropna().copy()
    s = s[s.p1_id != s.p2_id].sort_values("date").reset_index(drop=True)
    start = s.date.min()
    s["week"] = ((s.date - start).dt.days // 7).astype(int)

    ids, inv = np.unique(np.concatenate([s.p1_id.to_numpy("int64"), s.p2_id.to_numpy("int64")]),
                         return_inverse=True)
    n_sets, n_players = len(s), len(ids)
    a_idx, b_idx = inv[:n_sets], inv[n_sets:]
    score_a = (s.winner_id.to_numpy("int64") == s.p1_id.to_numpy("int64")).astype(float)

    mu = np.zeros(n_players)
    phi = np.full(n_players, RD0 / SCALE)
    sigma = np.full(n_players, VOL0)
    last = np.full(n_players, -1)            # last rating period each player was updated
    sets_seen = np.zeros(n_players, dtype=int)
    phi_max = RD0 / SCALE

    pre = np.zeros((n_sets, 6))              # p1 mu, phi, prior sets | p2 mu, phi, prior sets
    weeks = s.week.to_numpy()
    bounds = np.flatnonzero(np.diff(weeks)) + 1
    starts = np.concatenate([[0], bounds]); ends = np.concatenate([bounds, [n_sets]])

    for lo, hi in zip(starts, ends):
        w = weeks[lo]
        a, b, sa = a_idx[lo:hi], b_idx[lo:hi], score_a[lo:hi]
        u = np.unique(np.concatenate([a, b]))

        # pre-period RD: grows by one volatility step per idle period
        idle = np.where(last[u] < 0, 0, w - last[u] - 1)
        phi[u] = np.minimum(np.sqrt(phi[u] ** 2 + idle * sigma[u] ** 2), phi_max)

        pre[lo:hi] = np.column_stack([mu[a], phi[a], sets_seen[a], mu[b], phi[b], sets_seen[b]])

        _, mu[u], phi[u], sigma[u] = rate_period(mu, phi, sigma, a, b, sa, tau)
        last[u] = w
        sets_seen[u] += np.bincount(np.concatenate([a, b]), minlength=n_players)[u]

    # sets with pre-week ratings, on the familiar 1500 / 350 scale
    s["p1_r"] = R0 + SCALE * pre[:, 0]; s["p1_rd"] = SCALE * pre[:, 1]; s["p1_prior_sets"] = pre[:, 2].astype(int)
    s["p2_r"] = R0 + SCALE * pre[:, 3]; s["p2_rd"] = SCALE * pre[:, 4]; s["p2_prior_sets"] = pre[:, 5].astype(int)
    s["p1_won"] = score_a.astype(int)

    # current ratings, RD inflated to the last week in the data
    w_end = weeks.max()
    idle = np.where(last < 0, 0, w_end - last)
    phi_now = np.minimum(np.sqrt(phi ** 2 + idle * sigma ** 2), phi_max)
    cur = pd.DataFrame({
        "player_id": ids,
        "rating": R0 + SCALE * mu,
        "rd": SCALE * phi_now,
        "volatility": sigma,
        "sets_rated": sets_seen,
    }).set_index("player_id")
    return s.drop(columns="week"), cur


def rating_gap(r1, rd1, r2, rd2):
    """Rating difference in Glicko-2 units, discounted by both players' uncertainty
    (Glicko's g-factor): a gap between two unsettled ratings counts for less."""
    g = 1 / np.sqrt(1 + 3 * ((rd1 ** 2 + rd2 ** 2) / SCALE ** 2) / np.pi ** 2)
    return g * (r1 - r2) / SCALE


def expected(r1, rd1, r2, rd2):
    """P(player 1 beats player 2) under Glicko-2, accounting for both RDs."""
    g = 1 / np.sqrt(1 + 3 * ((rd1 ** 2 + rd2 ** 2) / SCALE ** 2) / np.pi ** 2)
    return 1 / (1 + np.exp(-g * (r1 - r2) / SCALE))


def scores(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "n": int(len(y)),
        "accuracy": float(((p > 0.5) == (y == 1)).mean()),
        "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
        "brier": float(((p - y) ** 2).mean()),
    }


# ── 3. character usage ─────────────────────────────────────────────────

def build_usage(pg: pd.DataFrame) -> pd.DataFrame:
    end = pg.date.max()
    recent = pg[pg.date >= end - pd.Timedelta(days=ACTIVE_DAYS)]
    has_recent = set(recent.player_id.unique())
    use = pd.concat([recent.assign(window="last 12 months"),
                     pg[~pg.player_id.isin(has_recent)].assign(window="all time")])
    u = use.groupby(["player_id", "char", "window"]).size().rename("games").reset_index()
    u["share"] = u.games / u.groupby("player_id").games.transform("sum")
    return u.sort_values(["player_id", "games"], ascending=[True, False]).reset_index(drop=True)


# ── 4. matchup model ───────────────────────────────────────────────────

def design(games: pd.DataFrame, chars: list, pairs: list):
    """Sparse design: [rating diff | character strength (+1/-1) | matchup interaction (+/-1)]."""
    ci = {c: i for i, c in enumerate(chars)}
    pi = {p: i for i, p in enumerate(pairs)}
    n, nc = len(games), len(chars)
    c1 = games.c1.map(ci).to_numpy(); c2 = games.c2.map(ci).to_numpy()
    lo = np.where(games.c1 < games.c2, games.c1, games.c2)
    hi = np.where(games.c1 < games.c2, games.c2, games.c1)
    k = pd.Series(list(zip(lo, hi))).map(pi).to_numpy()
    sign = np.where(games.c1.to_numpy() == lo, 1.0, -1.0)
    rows = np.arange(n)
    X_s = sparse.csr_matrix((np.r_[np.ones(n), -np.ones(n)], (np.r_[rows, rows], np.r_[c1, c2])), shape=(n, nc))
    X_i = sparse.csr_matrix((sign, (rows, k)), shape=(n, len(pairs)))
    x_r = sparse.csr_matrix(games.rdiff.to_numpy()[:, None])
    return sparse.hstack([x_r, X_s, X_i]).tocsr()


def fit_logistic(X, y, nc, lam_s, lam_i):
    """Ridge-penalised logistic regression (no intercept: the model is symmetric)."""
    pen = np.r_[0.0, np.full(nc, lam_s), np.full(X.shape[1] - 1 - nc, lam_i)]

    def fg(w):
        z = X @ w
        p = 1 / (1 + np.exp(-z))
        ll = np.logaddexp(0, z) - y * z
        loss = ll.sum() + 0.5 * (pen * w * w).sum()
        grad = X.T @ (p - y) + pen * w
        return loss, grad

    res = minimize(fg, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return res.x


def prep_games(games: pd.DataFrame, rated: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    g = games[~games.is_mirror].merge(
        rated[["set_id", "p1_id", "p1_r", "p1_rd", "p2_r", "p2_rd"]].rename(columns={"p1_id": "rated_p1"}), on="set_id")
    g = g[g.rated_p1 == g.p1_id]                       # orientation check
    # randomly swap sides so "player 1" (often the higher seed) carries no built-in advantage
    flip = np.random.default_rng(seed).random(len(g)) < 0.5
    out = pd.DataFrame({
        "event_id": g.event_id.to_numpy(),
        "date": g.date.to_numpy(),
        "c1": np.where(flip, g.p2_char, g.p1_char),
        "c2": np.where(flip, g.p1_char, g.p2_char),
        "rdiff": np.where(flip, -1, 1) * rating_gap(g.p1_r, g.p1_rd, g.p2_r, g.p2_rd).to_numpy(),
        "y": np.where(flip, g.winner_id == g.p2_id, g.winner_id == g.p1_id).astype(float),
    })
    return out


def fit_matchups(games: pd.DataFrame, rated: pd.DataFrame, report: list) -> dict:
    d = prep_games(games, rated)
    chars = sorted(set(d.c1) | set(d.c2))
    pairs = sorted({tuple(sorted(p)) for p in zip(d.c1, d.c2)})
    nc = len(chars)

    # choose penalties on a time split: fit before TEST_FROM, score on/after
    tr, te = d[d.date < pd.Timestamp(TEST_FROM)], d[d.date >= pd.Timestamp(TEST_FROM)]
    X_tr, X_te = design(tr, chars, pairs), design(te, chars, pairs)
    best = None
    report.append(f"Matchup model: {len(d):,} non-mirror games, {nc} characters, {len(pairs):,} matchups")
    report.append(f"  validation: train {len(tr):,} games before {TEST_FROM}, test {len(te):,} after")
    base = fit_logistic(X_tr[:, :1], tr.y.to_numpy(), 0, 0, 0)
    sc0 = scores(te.y.to_numpy(), 1 / (1 + np.exp(-(X_te[:, :1] @ base))))
    report.append(f"  ratings only:            log loss {sc0['log_loss']:.4f}  accuracy {sc0['accuracy']:.3f}")
    for lam_s, lam_i in [(10, 30), (10, 100), (30, 300), (100, 1000)]:
        w = fit_logistic(X_tr, tr.y.to_numpy(), nc, lam_s, lam_i)
        sc = scores(te.y.to_numpy(), 1 / (1 + np.exp(-(X_te @ w))))
        report.append(f"  ratings + characters (penalty {lam_s}/{lam_i}): log loss {sc['log_loss']:.4f}  accuracy {sc['accuracy']:.3f}")
        if best is None or sc["log_loss"] < best[0]["log_loss"]:
            best = (sc, lam_s, lam_i)
    sc, lam_s, lam_i = best
    report.append(f"  chosen penalty {lam_s}/{lam_i}; characters improve log loss by "
                  f"{sc0['log_loss'] - sc['log_loss']:.4f} over ratings alone")

    # refit on everything for the dashboard
    w = fit_logistic(design(d, chars, pairs), d.y.to_numpy(), nc, lam_s, lam_i)
    beta, s_eff, i_eff = w[0], w[1:1 + nc], w[1 + nc:]
    n_games = pd.Series(list(zip(np.minimum(d.c1, d.c2), np.maximum(d.c1, d.c2)))).value_counts()
    return {
        "beta": float(beta),                 # multiplies rating_gap() (Glicko-2 units, RD-discounted)
        "scale": SCALE,
        "penalty": {"strength": lam_s, "interaction": lam_i},
        "validation": {"ratings_only": sc0, "ratings_plus_characters": sc, "test_from": TEST_FROM},
        "characters": chars,
        "strength": dict(zip(chars, map(float, s_eff))),
        "interaction": {f"{a}|{b}": float(v) for (a, b), v in zip(pairs, i_eff)},
        "games": {f"{a}|{b}": int(n_games.get((a, b), 0)) for (a, b) in pairs},
    }


def matchup_logit(model, a, b):
    """Character part of the logit for character a vs character b (antisymmetric)."""
    s = model["strength"]
    lo, hi = sorted((a, b))
    inter = model["interaction"].get(f"{lo}|{hi}", 0.0)
    return s[a] - s[b] + (inter if a == lo else -inter)


# ── main ───────────────────────────────────────────────────────────────

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = []

    log("loading + cleaning data/processed ...")
    d = clean.load_clean(ROOT)
    sets_all, sets_c, games, pg = d["sets_all"], d["sets"], d["games"], d["pg"]
    log(f"  {len(sets_c):,} completed sets, {len(games):,} clean games")

    log("player table ...")
    players = build_players(sets_all)

    log("Glicko-2 ratings ...")
    rated, cur = glicko2(sets_c)
    rated.to_csv(OUT / "sets_rated.csv.gz", index=False)

    # rating validation: sets on/after TEST_FROM, using ratings from before each week
    t = rated[rated.date >= pd.Timestamp(TEST_FROM)]
    p = expected(t.p1_r, t.p1_rd, t.p2_r, t.p2_rd).to_numpy()
    sc = scores(t.p1_won.to_numpy(), p)
    est = t[(t.p1_prior_sets >= MIN_SETS_RANKED) & (t.p2_prior_sets >= MIN_SETS_RANKED)]
    sc_est = scores(est.p1_won.to_numpy(), expected(est.p1_r, est.p1_rd, est.p2_r, est.p2_rd).to_numpy())
    report += [f"Ratings: {len(cur):,} players rated from {len(rated):,} sets",
               f"  predicting sets from {TEST_FROM} with ratings from before each week:",
               f"  all sets:                 accuracy {sc['accuracy']:.3f}  log loss {sc['log_loss']:.4f}  ({sc['n']:,} sets)",
               f"  both players {MIN_SETS_RANKED}+ prior sets: accuracy {sc_est['accuracy']:.3f}  log loss {sc_est['log_loss']:.4f}  ({sc_est['n']:,} sets)",
               "  (a coin flip scores accuracy 0.500, log loss 0.6931)", ""]

    # current rank: players active in the last 12 months with enough sets, ordered by rating
    # (the same rating the win probabilities use, so "higher rank" and "favoured" always agree)
    players = players.join(cur, how="left")
    eligible = (players.sets_last_12m >= MIN_SETS_RANKED) & (players.rd <= MAX_RD_RANKED)
    players["rank"] = players.loc[eligible, "rating"].rank(ascending=False, method="first").astype("Int64")
    players = players.sort_values("rank")
    players.to_csv(OUT / "players.csv", index_label="player_id")
    players[["tag", "rating", "rd", "rank", "sets_last_12m"]].to_csv(OUT / "ratings_current.csv", index_label="player_id")
    report += [f"Players: {len(players):,} total, {int(eligible.sum()):,} ranked "
               f"({MIN_SETS_RANKED}+ sets in the last {ACTIVE_DAYS} days, RD <= {MAX_RD_RANKED})",
               "  top 10: " + ", ".join(players.dropna(subset=["rank"]).head(10).tag.astype(str)), ""]

    log("character usage ...")
    usage = build_usage(pg)
    usage.to_csv(OUT / "usage.csv", index=False)
    report.append(f"Character usage: {usage.player_id.nunique():,} players with character data "
                  f"({usage.player_id.nunique() / len(players):.0%} of all players)\n")

    log("matchup model (this is the slow part) ...")
    model = fit_matchups(games, rated, report)
    (OUT / "matchup_model.json").write_text(json.dumps(model, indent=1), encoding="utf-8")

    chars = model["characters"]
    strength = pd.Series(model["strength"]).sort_values(ascending=False)
    pd.DataFrame({
        "strength_logit": strength.round(4),
        "win_pct_vs_average": (100 / (1 + np.exp(-(strength - strength.mean())))).round(1),
        "rank": range(1, len(strength) + 1),
    }).to_csv(OUT / "char_strength.csv", index_label="char")
    mat = pd.DataFrame([[1 / (1 + np.exp(-matchup_logit(model, a, b))) if a != b else 0.5 for b in chars]
                        for a in chars], index=chars, columns=chars)
    mat.round(4).to_csv(OUT / "matchup_matrix.csv")
    report += ["", "Skill-adjusted character strength, top 5: " + ", ".join(strength.index[:5]),
               "  bottom 5: " + ", ".join(strength.index[-5:])]

    (OUT / "model_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n" + "\n".join(report))
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
