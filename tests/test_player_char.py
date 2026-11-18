# Contains Claude generated code

"""Player-on-character term: leakage-free familiarity, design symmetry, recovery, dashboard + seeder use."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from conftest import load_build, make_model_dir
from predict import SmashModel
import seeder

B = load_build()
CHARS = ["A", "B", "C", "D"]
PAIRS = sorted({tuple(sorted((x, y))) for x in CHARS for y in CHARS if x != y})


# ── familiarity is computed from earlier dates only ──

def _games(rows):
    return pd.DataFrame(rows, columns=["date", "p1_id", "p1_char", "p2_id", "p2_char"]).assign(
        date=lambda d: pd.to_datetime(d.date))


def test_experience_uses_only_earlier_dates():
    g = _games([("2025-01-01", 1, "A", 9, "B"),
                ("2025-01-01", 1, "A", 9, "B"),    # same day: must not count each other
                ("2025-01-08", 1, "A", 9, "B"),
                ("2025-01-15", 1, "C", 9, "B")])
    e = B.char_experience(g)
    assert e.p1_exp.iloc[0] == 0 and e.p1_exp.iloc[1] == 0          # no history yet -> neutral
    assert e.p1_exp.iloc[2] == pytest.approx(math.log(3 / 3))        # 2 prior games, both on A
    assert e.p1_exp.iloc[3] == pytest.approx(math.log(1 / 4))        # 3 prior games, none on C


def test_experience_now_matches_training_formula():
    assert B.experience_now(0, 0) == 0
    assert B.experience_now(30, 100) == pytest.approx(math.log(31 / 101))


# ── design matrix ──

def _d(n=6):
    return pd.DataFrame({"c1": ["A", "B", "C", "A", "D", "B"][:n], "c2": ["B", "C", "A", "D", "A", "A"][:n],
                         "rdiff": np.linspace(-1, 1, n), "pa": [1, 2, 3, 1, 4, 2][:n], "pb": [2, 3, 1, 4, 1, 1][:n],
                         "ea": np.linspace(-2, 0, n), "eb": np.linspace(0, -1, n)})


def test_swapping_sides_negates_the_full_design_row():
    d = _d()
    sw = pd.DataFrame({"c1": d.c2, "c2": d.c1, "rdiff": -d.rdiff, "pa": d.pb, "pb": d.pa, "ea": d.eb, "eb": d.ea})
    pci = B.pc_index(pd.concat([d, sw]), min_games=1)
    X, Xs = B.design_pc(d, CHARS, PAIRS, pci).toarray(), B.design_pc(sw, CHARS, PAIRS, pci).toarray()
    np.testing.assert_allclose(X, -Xs)


def test_pairs_below_the_game_threshold_get_no_adjustment():
    d = _d()
    pci = B.pc_index(d, min_games=3)
    assert (2, "B") in pci and (3, "C") not in pci                  # player 2 used B 3 times, player 3 used C twice


def test_fit_recovers_the_familiarity_effect():
    rng = np.random.default_rng(1)
    n, gamma = 40_000, 0.25
    d = pd.DataFrame({"c1": rng.choice(CHARS, n), "c2": rng.choice(CHARS, n), "rdiff": rng.normal(0, 1, n),
                      "pa": 1, "pb": 2, "ea": -rng.exponential(1, n), "eb": -rng.exponential(1, n)})
    d = d[d.c1 != d.c2]
    z = 0.8 * d.rdiff + gamma * (d.ea - d.eb)
    y = (rng.random(len(d)) < 1 / (1 + np.exp(-z))).astype(float).to_numpy()
    X = B.design_pc(d, CHARS, PAIRS, {}, with_u=False)
    w = B.fit_penalised(X, y, B.pc_penalty(len(CHARS), len(PAIRS), 0, 10, 100, 0))
    assert w[1 + len(CHARS) + len(PAIRS)] == pytest.approx(gamma, abs=0.04)


# ── dashboard ──

@pytest.fixture
def m(tmp_path):
    root = make_model_dir(tmp_path)
    d = root / "data" / "models"
    model = json.loads((d / "matchup_model.json").read_text())
    model["experience_coef"] = 0.2
    (d / "matchup_model.json").write_text(json.dumps(model))
    pd.DataFrame([(1, "Steve", 70, 0.1), (1, "Cloud", 20, -0.05), (1, "Fox", 10, 0.0),
                  (2, "Kirby", 40, 0.0)], columns=["player_id", "char", "games", "adj"]).to_csv(
        d / "player_char.csv", index=False)
    return SmashModel(root)


def test_player_char_logit_formula(m):
    assert m.player_char_logit(1, "Steve") == pytest.approx(0.2 * math.log(71 / 101) + 0.1)
    assert m.player_char_logit(1, "Kirby") == pytest.approx(0.2 * math.log(1 / 101))
    assert m.player_char_logit(4, "Steve") == 0 and m.player_char_logit(None, "Steve") == 0


def test_unfamiliar_character_lowers_the_chance_and_is_flagged(m):
    main = m.predict(1, 2, "Steve", "Kirby")
    off = m.predict(1, 2, "Kirby", "Kirby")                         # mirror, but player 1 never plays Kirby
    assert off["p_game"] < 0.5 + (main["p_game"] - 0.5) and off["p_game"] < m.predict(1, 2)["p_game"]
    assert any("Alpha has played Kirby in 0 of 100" in n for n in off["notes"])


def test_player_term_is_antisymmetric(m):
    p = m.predict(1, 2, "Cloud", "Kirby")["p_game"]
    q = m.predict(2, 1, "Kirby", "Cloud")["p_game"]
    assert p + q == pytest.approx(1)


def test_characters_only_ignores_the_player_term(m):
    assert m.predict(char_a="Steve", char_b="Kirby")["p_game"] == pytest.approx(1 / (1 + math.exp(-0.3)))


def test_model_folder_without_player_char_still_works(tmp_path):
    m0 = SmashModel(make_model_dir(tmp_path))
    assert m0.player_char_logit(1, "Steve") == 0
    assert 0 < m0.predict(1, 2, "Steve", "Kirby")["p_game"] < 1


# ── seeder ──

def test_seeder_offsets_shift_the_pairwise_probabilities():
    model = {"beta": 1.0, "scale": 173.7178, "characters": [], "strength": {}, "interaction": {}}
    even = seeder.pairwise_set_probs(model, [1600, 1600], [80, 80])
    tilt = seeder.pairwise_set_probs(model, [1600, 1600], [80, 80], offsets=[0.3, 0.0])
    assert even[0, 1] == pytest.approx(0.5) and tilt[0, 1] > 0.5
    assert tilt[0, 1] + tilt[1, 0] == pytest.approx(1)


def test_player_level_effect_goes_to_the_player_term_not_the_character_adjustments():
    """A player who beats their rating on EVERY character should get a player-level adjustment
    (left out of the dashboard), not inflated character adjustments; a player who is only better
    on one character should get a character adjustment for that one."""
    rng = np.random.default_rng(3)
    n = 30_000
    pa, pb = rng.integers(1, 21, n), rng.integers(21, 41, n)
    d = pd.DataFrame({"c1": rng.choice(["A", "B"], n), "c2": rng.choice(["C", "D"], n),
                      "rdiff": rng.normal(0, 1, n), "pa": pa, "pb": pb, "ea": 0.0, "eb": 0.0})
    z = 0.8 * d.rdiff + 0.8 * (d.pa == 1) + 0.8 * ((d.pa == 2) & (d.c1 == "A"))
    d["y"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(float)
    f = B.fit_player_char(d, CHARS, PAIRS, 10, 100, lam_u=10, lam_p=3)
    adj = f["adj"]
    w_player = f["w"][-len(f["pidx"]):]
    assert w_player[f["pidx"][1]] > 0.5                              # player 1: player-level
    assert abs(adj[(1, "A")] - adj[(1, "B")]) < 0.15                  # ...no character difference
    assert adj[(2, "A")] - adj[(2, "B")] > 0.4                        # player 2: better on A only
    used = B.logit_pc(d, CHARS, PAIRS, f, as_used=True)
    full = B.logit_pc(d, CHARS, PAIRS, f, as_used=False)
    assert np.abs(used - full)[d.pa.to_numpy() == 1].mean() > 0.4     # the dashboard drops the player part
