"""Dashboard logic in src/predict.py, against a tiny known model (conftest.make_model_dir)."""
import math

import numpy as np
import pytest

from conftest import make_model_dir
from predict import SmashModel, set_prob, split


@pytest.fixture
def m(tmp_path):
    return SmashModel(make_model_dir(tmp_path))


# ── set probability and display ──

@pytest.mark.parametrize("best_of", [1, 3, 5, 7])
def test_set_prob_symmetry_and_fixed_points(best_of):
    assert set_prob(0.5, best_of) == pytest.approx(0.5)
    assert set_prob(0.0, best_of) == 0 and set_prob(1.0, best_of) == pytest.approx(1)
    for p in np.linspace(0.05, 0.95, 19):
        assert set_prob(p, best_of) + set_prob(1 - p, best_of) == pytest.approx(1)


def test_set_prob_known_values_and_longer_sets_favour_the_better_player():
    assert set_prob(0.6, 3) == pytest.approx(0.648)
    assert set_prob(0.6, 5) == pytest.approx(0.68256)
    assert set_prob(0.6, 1) == pytest.approx(0.6)
    ps = np.linspace(0.51, 0.99, 25)
    assert all(set_prob(p, 5) > set_prob(p, 3) > p for p in ps)
    assert all(np.diff([set_prob(p, 3) for p in ps]) > 0)      # monotone


@pytest.mark.parametrize("p", [0.004, 0.333, 0.5, 0.645, 0.996])
def test_split_adds_to_100(p):
    a, b = map(int, split(p).split("-"))
    assert a + b == 100


# ── search and usage ──

def test_search_ranks_exact_matches_first_and_finds_past_tags(m):
    res = m.search("alpha")
    assert list(res.index[:2]) == [1, 3]                       # both exact "Alpha"s, more active first
    assert 5 in res.index                                      # prefix match "Alphabet"
    assert 2 in m.search("oldbeta").index                      # past tag
    assert m.search("   ").empty and m.search("zzz").empty


def test_label_tells_same_tag_players_apart(m):
    assert m.label(1) != m.label(3)
    assert "unranked" in m.label(3) and "#1" in m.label(1)


def test_usage_is_top_two_plus_other_and_sums_to_one(m):
    u = m.usage(1)
    assert [c for c, _ in u] == ["Steve", "Cloud", "Other"]
    assert sum(s for _, s in u) == pytest.approx(1)
    assert m.usage(2) == [("Kirby", 1.0)]                      # no 'Other' when nothing is left
    assert m.usage(4) is None
    assert m.main_char(1) == "Steve" and m.main_char(4) is None


# ── predict(): every case in the PM's spec ──

def test_characters_only_is_antisymmetric(m):
    for a, b in [("Steve", "Kirby"), ("Cloud", "Fox"), ("Cloud", "Kirby")]:
        p = m.predict(char_a=a, char_b=b)
        q = m.predict(char_a=b, char_b=a)
        assert p["mode"] == "characters only"
        assert p["p_game"] + q["p_game"] == pytest.approx(1)


def test_characters_only_matches_the_model_formula(m):
    z = 0.2 - (-0.2) + (-0.1)                                  # Steve vs Kirby; Kirby is first alphabetically
    assert m.predict(char_a="Steve", char_b="Kirby")["p_game"] == pytest.approx(1 / (1 + math.exp(-z)))


def test_mirror_is_even_with_a_note(m):
    r = m.predict(char_a="Fox", char_b="Fox")
    assert r["p_game"] == 0.5 and any("Mirror" in n for n in r["notes"])


def test_rare_matchup_is_flagged(m):
    assert any("Only 40 games" in n for n in m.predict(char_a="Cloud", char_b="Fox")["notes"])
    assert not m.predict(char_a="Steve", char_b="Kirby")["notes"]


def test_players_only_favours_the_higher_rank_and_agrees_with_the_number(m):
    r = m.predict(1, 2)
    assert r["mode"] == "players only" and r["favoured"] == 1 and r["p_game"] > 0.5
    r2 = m.predict(2, 1)
    assert r2["favoured"] == 1 and r2["p_game"] < 0.5
    assert r["p_game"] + r2["p_game"] == pytest.approx(1)


def test_unrated_player_gets_default_rating(m):
    r = m.predict(4, 1)                                        # player 4 has no rating
    assert r["p_game"] < 0.5 and r["favoured"] == 1


def test_uncertain_ratings_give_less_extreme_odds(m):
    gap_sure = m.rating_gap(1, 2)                              # RDs 60 / 70
    m.players.loc[2, "rd"] = 300
    assert 0 < m.rating_gap(1, 2) < gap_sure


def test_players_and_characters_combine(m):
    base = m.predict(1, 2)["p_game"]
    good = m.predict(1, 2, "Steve", "Kirby")["p_game"]         # Steve is favoured vs Kirby
    bad = m.predict(1, 2, "Kirby", "Steve")["p_game"]
    assert bad < base < good


def test_one_missing_character_is_filled_with_that_players_main(m):
    r = m.predict(1, 2, char_a="Cloud")
    assert (r["char_a"], r["char_b"]) == ("Cloud", "Kirby")
    r = m.predict(1, 4, char_a="Cloud")                        # player 4 has no main -> players only
    assert r["mode"] == "players only"


def test_nothing_selected_raises(m):
    with pytest.raises(ValueError):
        m.predict()
    with pytest.raises(ValueError):
        m.predict(a=1, char_a="Steve")                         # one player + one character is not enough


@pytest.mark.parametrize("best_of", [3, 5])
def test_outputs_are_valid_probabilities(m, best_of):
    for args in [(1, 2, None, None), (None, None, "Fox", "Steve"), (1, 5, "Fox", "Cloud")]:
        r = m.predict(*args, best_of=best_of)
        assert 0 < r["p_game"] < 1 and 0 < r["p_set"] < 1
        assert (r["p_set"] - 0.5) * (r["p_game"] - 0.5) >= 0   # set and game agree on the favourite
