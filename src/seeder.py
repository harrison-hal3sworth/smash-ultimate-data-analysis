# Contains Claude generated code

"""
seeder.py  —  auto-seeding (PM deliverable 6).

Seeds a list of players by each one's average chance of beating everyone else in the
field, using the same model as the dashboard: rating gap (uncertainty-discounted) plus
the character matchup, with each player assumed to use their most-used character.

    from seeder import pairwise_set_probs, seed_order
    P = pairwise_set_probs(model, ratings, rds, chars)   # P[i, j] = chance i beats j in a set
    order = seed_order(P)                                # indices, best seed first

Evaluation helpers (used by 09_validate_seeding.py) live here too, so they can be unit-tested.
"""

import math

import numpy as np
import pandas as pd


# ── model pieces ───────────────────────────────────────────────────────

def matchup_matrix(model: dict) -> tuple[dict, np.ndarray]:
    """Character index and the antisymmetric character logit matrix M[a, b]."""
    chars = model["characters"]
    idx = {c: i for i, c in enumerate(chars)}
    s = np.array([model["strength"][c] for c in chars])
    M = s[:, None] - s[None, :]
    for key, v in model["interaction"].items():
        lo, hi = key.split("|")
        i, j = idx[lo], idx[hi]
        M[i, j] += v
        M[j, i] -= v
    return idx, M


def set_prob_array(p: np.ndarray, best_of: int = 3) -> np.ndarray:
    """Vectorized best-of-N set probability from per-game probability p."""
    need = best_of // 2 + 1
    q = 1 - p
    return sum(math.comb(need - 1 + k, k) * p ** need * q ** k for k in range(need))


def pairwise_set_probs(model: dict, ratings, rds, chars=None, best_of: int = 3, offsets=None) -> np.ndarray:
    """P[i, j] = chance player i beats player j in a best-of-N set.
    ratings / rds: arrays on the 1500 / 350 scale. chars: list of character names or None
    (None, or an unknown name, means 'no character effect' for that player).
    offsets: optional per-player log-odds terms (the player-on-character term for the
    character each player is assumed to use); 0 = no adjustment."""
    r, d = np.asarray(ratings, float), np.asarray(rds, float)
    S, beta = model["scale"], model["beta"]
    g = 1 / np.sqrt(1 + 3 * ((d[:, None] ** 2 + d[None, :] ** 2) / S ** 2) / np.pi ** 2)
    z = beta * g * (r[:, None] - r[None, :]) / S
    if chars is not None:
        idx, M = matchup_matrix(model)
        ci = np.array([idx.get(c, -1) if isinstance(c, str) else -1 for c in chars])
        known = ci >= 0
        both = known[:, None] & known[None, :]
        cc = np.where(known, ci, 0)
        z = z + np.where(both, M[cc[:, None], cc[None, :]], 0.0)
    if offsets is not None:
        o = np.nan_to_num(np.asarray(offsets, float))
        z = z + o[:, None] - o[None, :]
    P = set_prob_array(1 / (1 + np.exp(-z)), best_of)
    np.fill_diagonal(P, np.nan)
    return P


def expected_win_share(P: np.ndarray) -> np.ndarray:
    """Average chance of beating each other player in the field."""
    return np.nanmean(P, axis=1)


def seed_order(P: np.ndarray) -> np.ndarray:
    """Indices sorted best seed first (ties broken by original order, i.e. stable)."""
    return np.argsort(-expected_win_share(P), kind="stable")


def seeds_from_order(order: np.ndarray) -> np.ndarray:
    """seed number (1 = top) for each player index."""
    seeds = np.empty(len(order), dtype=int)
    seeds[order] = np.arange(1, len(order) + 1)
    return seeds


# ── evaluation ─────────────────────────────────────────────────────────

def spearman(a, b) -> float:
    """Rank correlation between two orderings (lower = better in both)."""
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def top_k_overlap(seeds, placements, k: int = 8) -> int:
    """How many of the k best-seeded players finished in the top k
    (placement <= k; start.gg shares placements, e.g. 5th, 5th, 7th, 7th)."""
    seeds, placements = np.asarray(seeds), np.asarray(placements)
    return int(((seeds <= k) & (placements <= k)).sum())


def set_accuracy(seed_of: dict, sets: pd.DataFrame) -> float:
    """Share of sets won by the better-seeded player. sets: columns p1_id, p2_id, p1_won.
    Sets involving players without a seed are skipped."""
    s1, s2 = sets.p1_id.map(seed_of), sets.p2_id.map(seed_of)
    ok = s1.notna() & s2.notna() & (s1 != s2)
    if not ok.any():
        return float("nan")
    better_p1 = s1[ok] < s2[ok]
    return float((better_p1 == sets.loc[ok, "p1_won"].astype(bool)).mean())


def evaluate_event(seeds_by_method: dict, placements, player_ids, sets: pd.DataFrame, k: int = 8) -> dict:
    """Metrics for one event, for every seeding method.
    seeds_by_method: {method: array of seed numbers aligned with player_ids}."""
    out = {}
    for method, seeds in seeds_by_method.items():
        seed_of = dict(zip(player_ids, seeds))
        out[method] = {
            "spearman": spearman(seeds, placements),
            "top8": top_k_overlap(seeds, placements, k),
            "set_accuracy": set_accuracy(seed_of, sets),
        }
    return out
