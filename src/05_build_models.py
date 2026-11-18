# Contains Claude generated code

"""
05_build_models.py  —  player table, Glicko-2 ratings, character usage, matchup model.

Reads the step-04 output (data/processed/) through clean.py and writes everything
the dashboard and the auto-seeder need to data/models/:

    players.csv          one row per player: current tag, past tags, state, activity, rank
    ratings_current.csv  latest Glicko-2 rating / RD per player + rank among active players
    sets_rated.csv.gz    every completed set with BOTH players' ratings *before* that week
    usage.csv            character usage per player (last 12 months, else all time)
    matchup_model.json   fitted model: rating coefficient, character strength, matchup terms,
                         familiarity coefficient
    player_char.csv      per player and character: all-time games + player-on-character adjustment
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
    return fit_penalised(X, y, pen)


def fit_penalised(X, y, pen):
    """Logistic regression with a separate ridge penalty per column."""
    def fg(w):
        z = X @ w
        p = 1 / (1 + np.exp(-z))
        ll = np.logaddexp(0, z) - y * z
        loss = ll.sum() + 0.5 * (pen * w * w).sum()
        grad = X.T @ (p - y) + pen * w
        return loss, grad

    res = minimize(fg, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return res.x


def char_experience(games: pd.DataFrame) -> pd.DataFrame:
    """Each player's familiarity with the character they used in each game, from games played
    on EARLIER dates only (no leakage): log((1 + games on this character) / (1 + all games)).
    0 = this is all they have played (or no history yet); very negative = a rare pick."""
    long = pd.concat([
        games[["date", "p1_id", "p1_char"]].set_axis(["date", "player_id", "char"], axis=1),
        games[["date", "p2_id", "p2_char"]].set_axis(["date", "player_id", "char"], axis=1)])
    long = long.dropna()
    pc = long.groupby(["player_id", "char", "date"]).size().rename("n").reset_index().sort_values("date")
    pc["prior_pc"] = pc.groupby(["player_id", "char"]).n.cumsum() - pc.n
    pp = long.groupby(["player_id", "date"]).size().rename("n").reset_index().sort_values("date")
    pp["prior_p"] = pp.groupby("player_id").n.cumsum() - pp.n
    out = games[["date", "p1_id", "p1_char", "p2_id", "p2_char"]].copy()
    for s in ("p1", "p2"):
        m = (out[["date", f"{s}_id", f"{s}_char"]].set_axis(["date", "player_id", "char"], axis=1)
             .merge(pc[["player_id", "char", "date", "prior_pc"]], on=["player_id", "char", "date"], how="left")
             .merge(pp[["player_id", "date", "prior_p"]], on=["player_id", "date"], how="left"))
        out[f"{s}_exp"] = np.log((1 + m.prior_pc.fillna(0).to_numpy()) / (1 + m.prior_p.fillna(0).to_numpy()))
    return out[["p1_exp", "p2_exp"]].set_index(games.index)


def experience_now(n_pc: float, n_p: float) -> float:
    """Same familiarity measure at prediction time, from all games so far."""
    return float(np.log((1 + n_pc) / (1 + n_p)))


def prep_games(games: pd.DataFrame, rated: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    if "p1_exp" not in games:
        games = games.join(char_experience(games))
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
        "pa": np.where(flip, g.p2_id, g.p1_id).astype("int64"),
        "pb": np.where(flip, g.p1_id, g.p2_id).astype("int64"),
        "ea": np.where(flip, g.p2_exp, g.p1_exp),
        "eb": np.where(flip, g.p1_exp, g.p2_exp),
    })
    return out


# player-on-character terms: [familiarity difference | one adjustment per (player, character)]
MIN_PC_GAMES = 5          # a (player, character) pair needs this many games to get its own adjustment


def pc_index(d: pd.DataFrame, min_games: int = MIN_PC_GAMES) -> dict:
    """(player_id, character) pairs with enough games in d to get their own adjustment."""
    both = pd.concat([d[["pa", "c1"]].set_axis(["p", "c"], axis=1), d[["pb", "c2"]].set_axis(["p", "c"], axis=1)])
    n = both.value_counts()
    keep = sorted(n[n >= min_games].index)
    return {k: i for i, k in enumerate(keep)}


def player_index(d: pd.DataFrame, min_games: int = MIN_PC_GAMES) -> dict:
    """Players with enough games in d to get a player-level adjustment."""
    n = pd.concat([d.pa, d.pb]).value_counts()
    return {p: i for i, p in enumerate(sorted(n[n >= min_games].index))}


def _signed_cols(ka, kb, n_cols):
    """Sparse +1 (side A) / -1 (side B) indicator columns from float index arrays (NaN = none)."""
    n = len(ka)
    rows = np.arange(n)
    ma, mb = ~np.isnan(ka), ~np.isnan(kb)
    return sparse.csr_matrix((np.r_[np.ones(ma.sum()), -np.ones(mb.sum())],
                              (np.r_[rows[ma], rows[mb]], np.r_[ka[ma], kb[mb]].astype(int))), shape=(n, n_cols))


def design_pc(d: pd.DataFrame, chars: list, pairs: list, pci: dict, with_u: bool = True, pidx: dict | None = None):
    """design() plus [familiarity difference | player-on-character columns | player columns]."""
    X = design(d, chars, pairs)
    blocks = [X, sparse.csr_matrix((d.ea - d.eb).to_numpy()[:, None])]
    if with_u:
        ka = pd.Series(list(zip(d.pa, d.c1))).map(pci).to_numpy(dtype=float)
        kb = pd.Series(list(zip(d.pb, d.c2))).map(pci).to_numpy(dtype=float)
        blocks.append(_signed_cols(ka, kb, len(pci)))
    if pidx:
        blocks.append(_signed_cols(d.pa.map(pidx).to_numpy(dtype=float), d.pb.map(pidx).to_numpy(dtype=float), len(pidx)))
    return sparse.hstack(blocks).tocsr()


def pc_penalty(nc: int, n_pairs: int, n_u: int, lam_s, lam_i, lam_u, n_p: int = 0, lam_p=0):
    return np.r_[0.0, np.full(nc, lam_s), np.full(n_pairs, lam_i), 0.0, np.full(n_u, lam_u), np.full(n_p, lam_p)]


def fit_player_char(d: pd.DataFrame, chars: list, pairs: list, lam_s, lam_i, lam_u, lam_p) -> dict:
    """Ratings + characters + familiarity + player-on-character adjustments, with a separate
    player-level adjustment so the character adjustments only describe differences WITHIN a
    player (better on Cloud than on their other characters). The player-level part soaks up
    "this player beats their rating" and is NOT used by the dashboard (player strength there
    stays the Glicko-2 rating, so rank and favourite always agree)."""
    pci, pidx = pc_index(d), player_index(d)
    X = design_pc(d, chars, pairs, pci, pidx=pidx)
    w = fit_penalised(X, d.y.to_numpy(), pc_penalty(len(chars), len(pairs), len(pci), lam_s, lam_i, lam_u, len(pidx), lam_p))
    k = 1 + len(chars) + len(pairs)
    return {"w": w, "pci": pci, "pidx": pidx, "gamma": float(w[k]),
            "adj": pd.Series(w[k + 1:k + 1 + len(pci)], name="adj",
                             index=pd.MultiIndex.from_tuples(list(pci), names=["player_id", "char"]))}


def logit_pc(d: pd.DataFrame, chars: list, pairs: list, fit: dict, as_used: bool = True) -> np.ndarray:
    """Log-odds from a fit_player_char() fit. as_used=True leaves out the player-level part,
    exactly as the dashboard and the auto-seeder do."""
    X = design_pc(d, chars, pairs, fit["pci"], pidx=fit["pidx"])
    w = fit["w"].copy()
    if as_used and fit["pidx"]:
        w[-len(fit["pidx"]):] = 0
    return X @ w


def fit_matchups(games: pd.DataFrame, rated: pd.DataFrame, report: list) -> tuple[dict, pd.Series]:
    """Fit the matchup model. Returns (model dict, player-on-character adjustments)."""
    d = prep_games(games, rated)
    chars = sorted(set(d.c1) | set(d.c2))
    pairs = sorted({tuple(sorted(p)) for p in zip(d.c1, d.c2)})
    nc, npairs = len(chars), len(pairs)

    # choose penalties on a time split: fit before TEST_FROM, score on/after
    tr, te = d[d.date < pd.Timestamp(TEST_FROM)], d[d.date >= pd.Timestamp(TEST_FROM)]
    y_tr, y_te = tr.y.to_numpy(), te.y.to_numpy()
    X_tr, X_te = design(tr, chars, pairs), design(te, chars, pairs)
    sig = lambda z: 1 / (1 + np.exp(-z))
    report.append(f"Matchup model: {len(d):,} non-mirror games, {nc} characters, {npairs:,} matchups")
    report.append(f"  validation: train {len(tr):,} games before {TEST_FROM}, test {len(te):,} after")
    base = fit_logistic(X_tr[:, :1], y_tr, 0, 0, 0)
    sc0 = scores(y_te, sig(X_te[:, :1] @ base))
    report.append(f"  ratings only:            log loss {sc0['log_loss']:.4f}  accuracy {sc0['accuracy']:.3f}")

    # (a) character matchups
    best = None
    for lam_s, lam_i in [(10, 30), (10, 100), (30, 300), (100, 1000)]:
        w = fit_logistic(X_tr, y_tr, nc, lam_s, lam_i)
        sc = scores(y_te, sig(X_te @ w))
        report.append(f"  + characters (penalty {lam_s}/{lam_i}): log loss {sc['log_loss']:.4f}  accuracy {sc['accuracy']:.3f}")
        if best is None or sc["log_loss"] < best[0]["log_loss"]:
            best = (sc, lam_s, lam_i, w)
    sc1, lam_s, lam_i, w1 = best

    # (b) + familiarity with the character picked
    Xe_tr, Xe_te = design_pc(tr, chars, pairs, {}, with_u=False), design_pc(te, chars, pairs, {}, with_u=False)
    we = fit_penalised(Xe_tr, y_tr, pc_penalty(nc, npairs, 0, lam_s, lam_i, 0))
    sc2 = scores(y_te, sig(Xe_te @ we))
    report.append(f"  + familiarity with the character: log loss {sc2['log_loss']:.4f}  accuracy {sc2['accuracy']:.3f}")

    # (c) + player-on-character adjustments (with a player-level term that the dashboard leaves out)
    best_u = None
    for lam_p, lam_u in [(3, 10), (10, 3), (10, 10), (10, 30)]:
        f = fit_player_char(tr, chars, pairs, lam_s, lam_i, lam_u, lam_p)
        sc_full = scores(y_te, sig(logit_pc(te, chars, pairs, f, as_used=False)))
        sc_used = scores(y_te, sig(logit_pc(te, chars, pairs, f, as_used=True)))
        report.append(f"  + player-on-character (penalty player {lam_p} / character {lam_u}): "
                      f"log loss {sc_used['log_loss']:.4f}  accuracy {sc_used['accuracy']:.3f} as used by the dashboard"
                      f"  [{sc_full['log_loss']:.4f} / {sc_full['accuracy']:.3f} with the player-level part]")
        if best_u is None or sc_full["log_loss"] < best_u[0]["log_loss"]:
            best_u = (sc_full, sc_used, lam_p, lam_u, f)
    sc_full, sc3, lam_p, lam_u, f_tr = best_u
    report.append(f"  chosen: characters {lam_s}/{lam_i}, familiarity, player-on-character penalty "
                  f"{lam_u} (player-level {lam_p}); {len(f_tr['pci']):,} player/character pairs with an adjustment")
    report.append(f"  log-loss gain over ratings alone: characters {sc0['log_loss'] - sc1['log_loss']:.4f}, "
                  f"+ familiarity {sc0['log_loss'] - sc2['log_loss']:.4f}, "
                  f"+ player-on-character {sc0['log_loss'] - sc3['log_loss']:.4f}")

    # how much of each prediction comes from characters (held-out games)
    z_new = logit_pc(te, chars, pairs, f_tr, as_used=True)
    r_old, c_old = np.abs(X_te[:, :1] @ w1[:1]).mean(), np.abs(X_te[:, 1:] @ w1[1:]).mean()
    r_new = np.abs(X_te[:, :1] @ f_tr["w"][:1]).mean()
    c_new = np.abs(z_new - X_te[:, :1] @ f_tr["w"][:1]).mean()
    weight = {"before": {"rating": float(r_old), "character": float(c_old)},
              "after": {"rating": float(r_new), "character": float(c_new)}}
    report.append(f"  average size of the character part of the log-odds: {c_old:.3f} -> {c_new:.3f} "
                  f"(rating part {r_old:.3f} -> {r_new:.3f})")

    # where the new term should matter: someone is on a character that is under half their games
    off = (np.minimum(te.ea, te.eb) < np.log(0.5)).to_numpy()
    sub = {
        "n": int(off.sum()),
        "ratings_only": scores(y_te[off], sig(X_te[off][:, :1] @ base)),
        "characters": scores(y_te[off], sig(X_te[off] @ w1)),
        "player_on_character": scores(y_te[off], sig(z_new[off])),
    }
    report.append(f"  games where a player is off their usual character ({sub['n']:,}): accuracy "
                  f"{sub['ratings_only']['accuracy']:.3f} ratings only, {sub['characters']['accuracy']:.3f} + characters, "
                  f"{sub['player_on_character']['accuracy']:.3f} + player-on-character")

    # refit on everything for the dashboard
    f = fit_player_char(d, chars, pairs, lam_s, lam_i, lam_u, lam_p)
    w = f["w"]
    beta, s_eff, i_eff = w[0], w[1:1 + nc], w[1 + nc:1 + nc + npairs]
    n_games = pd.Series(list(zip(np.minimum(d.c1, d.c2), np.maximum(d.c1, d.c2)))).value_counts()
    model = {
        "beta": float(beta),                 # multiplies rating_gap() (Glicko-2 units, RD-discounted)
        "scale": SCALE,
        "experience_coef": f["gamma"],       # multiplies (familiarity A - familiarity B), see experience_now()
        "penalty": {"strength": lam_s, "interaction": lam_i, "player_char": lam_u, "player": lam_p},
        "validation": {"ratings_only": sc0, "ratings_plus_characters": sc1, "plus_familiarity": sc2,
                       "plus_player_on_character": sc3, "with_player_level_part": sc_full,
                       "off_usual_character": sub, "logit_parts": weight, "test_from": TEST_FROM},
        "characters": chars,
        "strength": dict(zip(chars, map(float, s_eff))),
        "interaction": {f"{a}|{b}": float(v) for (a, b), v in zip(pairs, i_eff)},
        "games": {f"{a}|{b}": int(n_games.get((a, b), 0)) for (a, b) in pairs},
    }
    return model, f["adj"]


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
    model, adj = fit_matchups(games, rated, report)
    (OUT / "matchup_model.json").write_text(json.dumps(model, indent=1), encoding="utf-8")

    # per player and character: all-time games (familiarity) + fitted adjustment, for the dashboard
    long = pd.concat([games[["p1_id", "p1_char"]].set_axis(["player_id", "char"], axis=1),
                      games[["p2_id", "p2_char"]].set_axis(["player_id", "char"], axis=1)]).dropna()
    pc = long.value_counts().rename("games").to_frame()
    pc.index = pc.index.set_levels(pc.index.levels[0].astype("int64"), level=0)
    pc = pc.join(adj, how="left").fillna({"adj": 0.0})
    pc["adj"] = pc.adj.round(5)
    pc.sort_index().to_csv(OUT / "player_char.csv")
    report.append(f"Player-on-character table: {len(pc):,} player/character pairs, "
                  f"{int((pc.adj != 0).sum()):,} with their own adjustment\n")

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
