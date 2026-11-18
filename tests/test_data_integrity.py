# Contains Claude generated code

"""Checks on the REAL pipeline outputs. Skipped automatically when the data isn't present.

    pytest -m data          # only these
    pytest -m "not data"    # only the fast unit tests
"""
import json

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT

PROC, MODELS = ROOT / "data" / "processed", ROOT / "data" / "models"
pytestmark = pytest.mark.data
need_proc = pytest.mark.skipif(not (PROC / "sets.csv").exists(), reason="no data/processed")
need_models = pytest.mark.skipif(not (MODELS / "players.csv").exists(), reason="no data/models")
need_rated = pytest.mark.skipif(not (MODELS / "sets_rated.csv.gz").exists(),
                                reason="no data/models/sets_rated.csv.gz (git-ignored; made by step 05)")


@pytest.fixture(scope="module")
def sets():
    return pd.read_csv(PROC / "sets.csv", low_memory=False,
                       usecols=["set_id", "status", "p1_id", "p2_id", "winner_id", "date"])


@pytest.fixture(scope="module")
def games():
    return pd.read_csv(PROC / "games.csv", low_memory=False,
                       usecols=["set_id", "game_num", "p1_id", "p2_id", "winner_id", "p1_char", "p2_char"])


@need_proc
def test_set_ids_are_unique(sets):
    assert not sets.set_id.dropna().duplicated().any()


@need_proc
def test_completed_sets_have_a_winner_who_played(sets):
    c = sets[sets.status == "complete"].dropna(subset=["p1_id", "p2_id"])
    assert ((c.winner_id == c.p1_id) | (c.winner_id == c.p2_id)).mean() > 0.999


@need_proc
def test_dates_are_inside_ultimates_lifetime(sets):
    d = pd.to_datetime(sets.date)
    assert d.min() >= pd.Timestamp("2018-12-07")               # Ultimate's release
    assert d.max() <= pd.Timestamp.now() + pd.Timedelta(days=1)


@need_proc
def test_games_have_sane_numbers_and_characters(games):
    # raw start.gg data has a few reporting errors (e.g. a 94-"game" set); clean.py drops those sets.
    # This fails if they ever become more than a trace.
    bad = ~games.game_num.between(1, 9)
    assert bad.mean() < 1e-4, f"{bad.sum()} games with numbers outside 1-9 in {games.loc[bad, 'set_id'].nunique()} sets"
    assert pd.concat([games.p1_char, games.p2_char]).nunique() <= 87   # 86 + "Random Character"


@need_proc
def test_cleaning_removes_impossible_game_numbers(games):
    import clean
    g = games.assign(event_id=0, p1_tag="", p2_tag="", date="2025-01-01", num_entrants=32)
    out = clean.clean_games(g, pd.DataFrame({"set_id": games.set_id.unique()}), [])
    assert out.game_num.between(1, 9).all()


@need_models
def test_players_and_ranks_are_consistent():
    p = pd.read_csv(MODELS / "players.csv", low_memory=False)
    assert p.player_id.is_unique
    r = p.dropna(subset=["rank"]).sort_values("rank")
    assert list(r["rank"].astype(int)) == list(range(1, len(r) + 1))   # 1..N, no gaps or ties
    assert (r.rd <= 100).all() and (r.sets_last_12m >= 10).all()
    assert r.rating.is_monotonic_decreasing                    # rank order = rating order


@need_models
def test_usage_shares_sum_to_one():
    u = pd.read_csv(MODELS / "usage.csv")
    assert np.allclose(u.groupby("player_id").share.sum(), 1)


@need_models
def test_matchup_matrix_is_antisymmetric():
    mat = pd.read_csv(MODELS / "matchup_matrix.csv", index_col=0)
    v = mat.to_numpy()
    assert np.allclose(v + v.T, 1, atol=1e-3)
    assert np.allclose(np.diag(v), 0.5)


@need_models
def test_model_file_covers_every_character_and_matchup():
    m = json.loads((MODELS / "matchup_model.json").read_text())
    n = len(m["characters"])
    assert set(m["strength"]) == set(m["characters"])
    assert len(m["interaction"]) <= n * (n - 1) // 2
    assert 0 < m["beta"] < 5


@need_rated
def test_every_players_first_set_uses_starting_ratings():
    """No leakage: before a player's first set they must be at 1500 / 350."""
    s = pd.read_csv(MODELS / "sets_rated.csv.gz", usecols=["date", "p1_id", "p1_r", "p1_rd",
                                                           "p1_prior_sets"], nrows=500_000)
    first = s[s.p1_prior_sets == 0]
    assert np.allclose(first.p1_r, 1500) and np.allclose(first.p1_rd, 350)
