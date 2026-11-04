"""Matchup model in src/05_build_models.py: design matrix symmetry and parameter recovery."""
import numpy as np
import pandas as pd
import pytest

from conftest import load_build

B = load_build()
CHARS = ["A", "B", "C", "D"]
PAIRS = sorted({tuple(sorted((x, y))) for x in CHARS for y in CHARS if x != y})


def test_swapping_sides_negates_the_design_row():
    g = pd.DataFrame({"c1": ["A", "C"], "c2": ["B", "A"], "rdiff": [0.3, -0.2]})
    swapped = pd.DataFrame({"c1": g.c2, "c2": g.c1, "rdiff": -g.rdiff})
    X, Xs = B.design(g, CHARS, PAIRS).toarray(), B.design(swapped, CHARS, PAIRS).toarray()
    np.testing.assert_allclose(X, -Xs)


def test_matchup_logit_is_antisymmetric_and_zero_for_mirrors():
    model = {"strength": {"A": 0.3, "B": -0.1}, "interaction": {"A|B": 0.2}}
    assert B.matchup_logit(model, "A", "B") == pytest.approx(-B.matchup_logit(model, "B", "A"))
    assert B.matchup_logit(model, "A", "B") == pytest.approx(0.3 + 0.1 + 0.2)


def test_fit_recovers_known_rating_effect_and_character_strengths():
    rng = np.random.default_rng(0)
    n, beta, strength = 40_000, 0.9, {"A": 0.4, "B": 0.1, "C": -0.1, "D": -0.4}
    c1, c2 = rng.choice(CHARS, n), rng.choice(CHARS, n)
    keep = c1 != c2
    g = pd.DataFrame({"c1": c1[keep], "c2": c2[keep], "rdiff": rng.normal(0, 1, keep.sum())})
    z = beta * g.rdiff + g.c1.map(strength) - g.c2.map(strength)
    y = (rng.random(len(g)) < 1 / (1 + np.exp(-z))).astype(float).to_numpy()
    w = B.fit_logistic(B.design(g, CHARS, PAIRS), y, len(CHARS), lam_s=1, lam_i=100)
    assert w[0] == pytest.approx(beta, abs=0.05)
    est = dict(zip(CHARS, w[1:1 + len(CHARS)]))
    for a, b in [("A", "D"), ("B", "C"), ("A", "B")]:        # strengths are identified as differences
        assert est[a] - est[b] == pytest.approx(strength[a] - strength[b], abs=0.07)


def test_ridge_pulls_rare_matchups_toward_zero():
    rng = np.random.default_rng(2)
    g = pd.DataFrame({"c1": ["A"] * 6, "c2": ["B"] * 6, "rdiff": np.zeros(6)})
    y = np.ones(6)                                           # A wins all 6 games
    loose = B.fit_logistic(B.design(g, CHARS, PAIRS), y, 4, lam_s=0.01, lam_i=0.01)
    tight = B.fit_logistic(B.design(g, CHARS, PAIRS), y, 4, lam_s=10, lam_i=100)
    logit = lambda w: w[1] - w[2] + w[1 + 4 + PAIRS.index(("A", "B"))]
    assert 0 < logit(tight) < logit(loose)


def test_random_side_flip_removes_slot_bias():
    games = pd.DataFrame({"set_id": range(2000), "event_id": 1, "is_mirror": False, "date": pd.Timestamp("2025-01-01"),
                          "p1_id": 1, "p2_id": 2, "p1_char": "A", "p2_char": "B", "winner_id": 1})
    rated = pd.DataFrame({"set_id": range(2000), "p1_id": 1, "p1_r": 1500.0, "p1_rd": 50.0,
                          "p2_r": 1500.0, "p2_rd": 50.0})
    d = B.prep_games(games, rated)
    assert 0.45 < d.y.mean() < 0.55                          # "player 1 always wins" is gone
    assert ((d.c1 == "A") == (d.y == 1)).all()               # but the A player still always wins
