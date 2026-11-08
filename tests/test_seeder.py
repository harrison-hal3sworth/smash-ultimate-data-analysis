"""Auto-seeder (src/seeder.py and SmashModel.seed)."""
import numpy as np
import pandas as pd
import pytest

import seeder
from conftest import make_model_dir
from predict import SmashModel, set_prob

MODEL = {"beta": 1.0, "scale": 173.7178, "characters": ["Fox", "Kirby", "Steve"],
         "strength": {"Fox": 0.0, "Kirby": -0.3, "Steve": 0.3}, "interaction": {"Kirby|Steve": 0.2}}


def test_pairwise_probabilities_are_complementary():
    P = seeder.pairwise_set_probs(MODEL, [1800, 1600, 1500, 1400], [60, 80, 200, 350], ["Steve", "Fox", None, "Kirby"])
    off = ~np.eye(4, dtype=bool)
    assert np.allclose((P + P.T)[off], 1)
    assert np.isnan(np.diag(P)).all()
    assert ((P[off] > 0) & (P[off] < 1)).all()


def test_matchup_matrix_matches_the_dashboard_formula(tmp_path):
    m = SmashModel(make_model_dir(tmp_path))
    idx, M = seeder.matchup_matrix(m.model)
    for a in m.characters:
        for b in m.characters:
            if a != b:
                assert M[idx[a], idx[b]] == pytest.approx(m.matchup_logit(a, b))


@pytest.mark.parametrize("p", [0.1, 0.5, 0.62, 0.9])
@pytest.mark.parametrize("best_of", [3, 5])
def test_vectorized_set_prob_matches_predict(p, best_of):
    assert seeder.set_prob_array(np.array([p]), best_of)[0] == pytest.approx(set_prob(p, best_of))


def test_without_characters_seeding_follows_rating():
    r = [1500, 1900, 1700, 1600]
    order = seeder.seed_order(seeder.pairwise_set_probs(MODEL, r, [80] * 4, None))
    assert list(order) == [1, 2, 3, 0]


def test_characters_break_a_tie_between_equal_players():
    P = seeder.pairwise_set_probs(MODEL, [1700, 1700], [80, 80], ["Kirby", "Steve"])
    assert list(seeder.seed_order(P)) == [1, 0]                  # Steve beats Kirby


def test_unknown_or_missing_character_has_no_effect():
    a = seeder.pairwise_set_probs(MODEL, [1700, 1600], [80, 80], None)
    b = seeder.pairwise_set_probs(MODEL, [1700, 1600], [80, 80], ["Not A Character", None])
    np.testing.assert_allclose(a, b)


def test_seeds_from_order_inverts_the_order():
    assert list(seeder.seeds_from_order(np.array([2, 0, 1]))) == [2, 3, 1]


def test_spearman_extremes():
    assert seeder.spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1)
    assert seeder.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1)


def test_top_k_overlap_with_shared_placements():
    seeds = np.arange(1, 13)
    placements = np.array([1, 2, 3, 4, 5, 5, 7, 7, 9, 9, 9, 9])
    assert seeder.top_k_overlap(seeds, placements, 8) == 8
    assert seeder.top_k_overlap(seeds[::-1], placements, 8) == 4


def test_set_accuracy_counts_wins_by_the_better_seed():
    sets = pd.DataFrame({"p1_id": [10, 20, 10, 99], "p2_id": [20, 30, 30, 10], "p1_won": [1, 0, 1, 1]})
    seed_of = {10: 1, 20: 2, 30: 3}                              # player 99 is unseeded -> skipped
    assert seeder.set_accuracy(seed_of, sets) == pytest.approx(2 / 3)


def test_evaluate_event_scores_every_method():
    ids = np.array([10, 20, 30])
    sets = pd.DataFrame({"p1_id": [10, 20], "p2_id": [20, 30], "p1_won": [1, 1]})
    res = seeder.evaluate_event({"good": np.array([1, 2, 3]), "bad": np.array([3, 2, 1])},
                                np.array([1, 2, 3]), ids, sets)
    assert res["good"]["spearman"] > res["bad"]["spearman"]
    assert res["good"]["set_accuracy"] == 1 and res["bad"]["set_accuracy"] == 0


def test_smashmodel_seed_orders_the_field_and_handles_unknown_players(tmp_path):
    m = SmashModel(make_model_dir(tmp_path))
    t = m.seed([2, None, 1, 5], names=["Beta", "Newcomer", "Alpha", "Alphabet"])
    assert list(t.player) == ["Alpha", "Beta", "Alphabet", "Newcomer"]
    assert list(t.seed) == [1, 2, 3, 4]
    assert t.avg_win_chance.is_monotonic_decreasing
    assert t.loc[t.player == "Alpha", "main"].iat[0] == "Steve"
    no_chars = m.seed([2, 1], use_chars=False)
    assert no_chars["main"].isna().all() and list(no_chars.player) == ["Alpha", "Beta"]
