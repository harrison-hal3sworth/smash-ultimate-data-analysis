"""Cleaning rules in src/clean.py, on hand-made rows."""
import numpy as np
import pandas as pd

import clean


def _sets():
    return pd.DataFrame({
        "event_id": [1, 1, 1, 1, 1],
        "set_id": [10, 11, 12, np.nan, 14],
        "status": ["complete", "complete", "dq", "unplayed", "complete"],
        "p1_id": [100, 100, 100, np.nan, 101],
        "p2_id": [200, 300, 200, np.nan, np.nan],          # set 14 has no second player
        "winner_id": [100, 300, 200, np.nan, 101],
        "has_games": [True, True, False, False, False],
        "has_chars": [True, False, False, False, False],
        "date": ["2025-01-05"] * 5,
        "num_entrants": [64] * 5,
        "tournament_id": [7] * 5,
    })


def _games():
    return pd.DataFrame({
        "event_id": [1, 1, 1, 1, 1],
        "set_id": [10, 10, 10, 11, 10],
        "game_num": [1, 2, 2, 1, 3],                       # (10, 2) is a duplicate
        "p1_id": [100] * 4 + [100], "p1_tag": ["A"] * 5, "p1_char": ["Steve", "Steve", "Steve", "Random Character", "Fox"],
        "p2_id": [200, 200, 200, 300, 200], "p2_tag": ["B"] * 5, "p2_char": ["Kirby", "Kirby", "Kirby", "Fox", "Fox"],
        "winner_id": [100, 200, 200, 300, 999],             # 999 is not one of the two players
        "stage": [None] * 5, "date": ["2025-01-05"] * 5, "num_entrants": [64] * 5,
        "round": ["W1"] * 5, "tournament_name": ["T"] * 5,
    })


def test_clean_sets_keeps_only_completed_sets_with_two_players():
    log = []
    sets_all, done = clean.clean_sets(_sets(), log)
    assert sets_all.set_id.notna().all()                    # placeholder dropped
    assert list(done.set_id) == [10, 11]                    # DQ, unplayed, one-player set dropped
    assert [r for _, _, r in log] == [5, 4, 2]


def test_clean_games_drops_duplicates_random_and_bad_winners():
    log = []
    _, done = clean.clean_sets(_sets(), log)
    g = clean.clean_games(_games(), done, log)
    assert not g.duplicated(["set_id", "game_num"]).any()
    assert "Random Character" not in set(g.p1_char) | set(g.p2_char)
    assert ((g.winner_id == g.p1_id) | (g.winner_id == g.p2_id)).all()
    assert len(g) == 2                                      # games (10,1) and (10,2)


def test_mirror_flag_and_unordered_matchup_key():
    g = pd.DataFrame({"set_id": [1, 2], "game_num": [1, 1], "p1_id": [1, 1], "p2_id": [2, 2],
                      "p1_char": ["Steve", "Fox"], "p2_char": ["Kirby", "Fox"], "winner_id": [1, 2],
                      "p1_tag": ["a", "a"], "p2_tag": ["b", "b"], "date": ["2025-01-01"] * 2,
                      "num_entrants": [40, 40]})
    out = clean.clean_games(g, pd.DataFrame({"set_id": [1, 2]}), [])
    assert list(out.is_mirror) == [False, True]
    assert (out.char_lo <= out.char_hi).all()
    assert list(out.char_lo) == ["Kirby", "Fox"]


def test_player_view_has_two_rows_per_game_and_one_winner_each():
    g = clean.clean_games(_games(), clean.clean_sets(_sets(), [])[1], [])
    pg = clean.player_view(g)
    assert len(pg) == 2 * len(g)
    assert pg.groupby(["set_id", "game_num"]).won.sum().eq(1).all()
    assert pg.won.mean() == 0.5


def test_wilson_interval_contains_the_estimate_and_stays_in_0_1():
    wins, n = np.array([0, 5, 50, 100]), np.array([10, 10, 100, 100])
    lo, hi = clean.wilson(wins, n)
    p = wins / n
    assert ((lo <= p + 1e-12) & (p <= hi + 1e-12)).all()
    assert (lo >= 0).all() and (hi <= 1).all()
    assert (hi - lo)[2] < (hi - lo)[1]                      # more games -> narrower


def test_matchup_stats_directed_rows_are_complements():
    rng = np.random.default_rng(0)
    n = 400
    g = pd.DataFrame({"set_id": np.arange(n), "game_num": 1, "p1_id": 1, "p2_id": 2,
                      "p1_char": rng.choice(["A", "B", "C"], n), "p2_char": rng.choice(["A", "B", "C"], n),
                      "p1_tag": "x", "p2_tag": "y", "date": "2025-01-01", "num_entrants": 40})
    g["winner_id"] = np.where(rng.random(n) < 0.5, 1, 2)
    g = clean.clean_games(g, pd.DataFrame({"set_id": g.set_id}), [])
    m = clean.matchup_stats(clean.player_view(g), min_games=1).reset_index()
    x = m.merge(m, left_on=["char", "opp_char"], right_on=["opp_char", "char"])
    assert (x.games_x == x.games_y).all()
    assert (x.wins_x + x.wins_y == x.games_x).all()


def test_sets_with_impossible_game_numbers_are_dropped():
    g = pd.DataFrame({"set_id": [1, 1, 2, 2, 2, 3, 3], "game_num": [1, 2, 0, 2, 3, 9, 10],
                      "p1_id": 1, "p2_id": 2, "p1_char": "Fox", "p2_char": "Steve", "winner_id": 1,
                      "p1_tag": "a", "p2_tag": "b", "date": "2025-01-01", "num_entrants": 40})
    log = []
    out = clean.clean_games(g, pd.DataFrame({"set_id": [1, 2, 3]}), log)
    assert set(out.set_id) == {1}                            # set 2 starts at 0, set 3 reaches 10
    assert any("outside 1-9" in step for _, step, _ in log)
