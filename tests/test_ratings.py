# Contains Claude generated code

"""Glicko-2 implementation in src/05_build_models.py."""
import numpy as np
import pandas as pd
import pytest

from conftest import load_build

B = load_build()


def test_matches_glickmans_worked_example():
    """Glickman (2013), 'Example of the Glicko-2 system': 1500/200/0.06 beats a 1400/30,
    loses to 1550/100 and 1700/300, tau = 0.5 -> 1464.06 / 151.52 / 0.05999."""
    S = B.SCALE
    r = np.array([1500, 1400, 1550, 1700.0]); rd = np.array([200, 30, 100, 300.0])
    mu, phi, sigma = (r - 1500) / S, rd / S, np.full(4, 0.06)
    u, mu_new, phi_new, sig_new = B.rate_period(mu, phi, sigma, np.array([0, 0, 0]),
                                                np.array([1, 2, 3]), np.array([1.0, 0, 0]), tau=0.5)
    i = list(u).index(0)
    assert 1500 + S * mu_new[i] == pytest.approx(1464.06, abs=0.05)
    assert S * phi_new[i] == pytest.approx(151.52, abs=0.05)
    assert sig_new[i] == pytest.approx(0.05999, abs=1e-5)


def _sets(rows):
    """rows: (date, winner, loser) -> sets frame as step 05 expects."""
    df = pd.DataFrame(rows, columns=["date", "w", "l"])
    return pd.DataFrame({"set_id": range(len(df)), "event_id": range(len(df)),
                         "date": pd.to_datetime(df.date), "p1_id": df.w, "p2_id": df.l, "winner_id": df.w})


def test_winner_gains_and_loser_drops():
    rated, cur = B.glicko2(_sets([("2025-01-01", 1, 2)]))
    assert cur.loc[1, "rating"] > 1500 > cur.loc[2, "rating"]
    assert cur.loc[1, "rating"] - 1500 == pytest.approx(1500 - cur.loc[2, "rating"])


def test_first_set_uses_default_ratings_no_leakage():
    rated, _ = B.glicko2(_sets([("2025-01-01", 1, 2), ("2025-01-20", 1, 2)]))
    first = rated.iloc[0]
    assert (first.p1_r, first.p1_rd, first.p2_r) == (1500, 350, 1500)
    assert rated.iloc[1].p1_r > 1500                         # week-1 result shows up in week 3


def test_future_results_do_not_change_past_pre_event_ratings():
    base = [("2025-01-01", 1, 2), ("2025-01-08", 2, 3), ("2025-01-15", 3, 1)]
    flipped = base[:2] + [("2025-01-15", 1, 3)]               # only the LAST result changes
    a, _ = B.glicko2(_sets(base)); b, _ = B.glicko2(_sets(flipped))
    cols = ["p1_r", "p1_rd", "p2_r", "p2_rd"]
    early = a.date < pd.Timestamp("2025-01-15")
    pd.testing.assert_frame_equal(a.loc[early, cols].reset_index(drop=True),
                                  b.loc[b.date < pd.Timestamp("2025-01-15"), cols].reset_index(drop=True))


def test_same_week_sets_are_rated_together():
    rated, _ = B.glicko2(_sets([("2025-01-01", 1, 2), ("2025-01-02", 1, 3)]))
    assert (rated.p1_r == 1500).all()                         # neither set sees the other's result


def test_rd_shrinks_with_play_and_grows_when_idle():
    busy, cur_busy = B.glicko2(_sets([("2025-01-01", 1, 2), ("2025-01-08", 1, 3)]))
    idle, _ = B.glicko2(_sets([("2025-01-01", 1, 2), ("2025-06-01", 1, 3)]))
    assert cur_busy.loc[1, "rd"] < 350
    assert idle.iloc[1].p1_rd > busy.iloc[1].p1_rd             # 21 idle weeks -> less certain


def test_stronger_player_ends_higher_on_simulated_data():
    rng = np.random.default_rng(1)
    skill = {1: 2.0, 2: 1.0, 3: 0.0, 4: -1.0}
    rows, day = [], pd.Timestamp("2024-01-01")
    for i in range(1200):
        a, b = rng.choice(list(skill), 2, replace=False)
        p = 1 / (1 + np.exp(-(skill[a] - skill[b])))
        w, l = (a, b) if rng.random() < p else (b, a)
        rows.append((day + pd.Timedelta(days=i // 10), w, l))
    _, cur = B.glicko2(_sets(rows))
    assert list(cur.rating.sort_values(ascending=False).index) == [1, 2, 3, 4]


def test_expected_score_is_symmetric_and_discounted_by_uncertainty():
    p = B.expected(1700, 50, 1500, 50)
    assert p == pytest.approx(1 - B.expected(1500, 50, 1700, 50))
    assert 0.5 < B.expected(1700, 300, 1500, 300) < p          # high RD pulls toward 50%
    assert B.rating_gap(1700, 300, 1500, 300) < B.rating_gap(1700, 50, 1500, 50)
