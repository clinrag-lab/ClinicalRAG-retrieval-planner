import pytest

from clinrag.planner.order import candidates_to_fetch, choose_order, search_first_candidates


def test_selective_filter_goes_filter_first():
    # 5 * 10000 / 20 = 2500 candidates > 1000
    d = choose_order(k=5, total_notes=10_000, matching_notes=20, max_candidates=1000)
    assert d.order == "filter_first"
    assert d.candidates == 2500


def test_loose_filter_goes_search_first():
    # 5 * 10000 / 5000 = 10 candidates <= 1000
    d = choose_order(k=5, total_notes=10_000, matching_notes=5000, max_candidates=1000)
    assert d.order == "search_first"
    assert d.candidates == 10


def test_zero_matches_goes_filter_first():
    d = choose_order(k=5, total_notes=10_000, matching_notes=0, max_candidates=1000)
    assert d.order == "filter_first"
    assert d.candidates is None


def test_threshold_boundary():
    # 5 * 10000 / 50 = exactly 1000 candidates
    at_limit = choose_order(k=5, total_notes=10_000, matching_notes=50, max_candidates=1000)
    over_limit = choose_order(k=5, total_notes=10_000, matching_notes=50, max_candidates=999)
    assert at_limit.candidates == 1000
    assert at_limit.order == "search_first"
    assert over_limit.order == "filter_first"


def test_decision_keeps_inputs():
    d = choose_order(k=3, total_notes=100, matching_notes=10, max_candidates=50)
    assert (d.k, d.total_notes, d.matching_notes, d.max_candidates) == (3, 100, 10, 50)


def test_candidates_round_up():
    # 5 * 10 / 3 = 16.67
    assert search_first_candidates(5, 10, 3) == 17


def test_candidates_to_fetch_caps_at_total():
    assert candidates_to_fetch(5, 10, 3) == 10
    assert candidates_to_fetch(5, 10, 0) == 10
    assert candidates_to_fetch(5, 10_000, 5000) == 10


@pytest.mark.parametrize(
    "k,total,matching",
    [(0, 10, 5), (5, 10, 11), (5, -1, 0), (5, 10, -1)],
)
def test_invalid_counts_raise(k, total, matching):
    with pytest.raises(ValueError):
        choose_order(k=k, total_notes=total, matching_notes=matching, max_candidates=100)
