import pytest

from clinrag.planner.order import candidates_to_fetch, choose_order, search_first_candidates, switch_matching_notes


def test_selective_filter_goes_filter_first():
    # 5 * 10000 / 20 = 2500 candidates > 0.1 * 10000
    d = choose_order(k=5, total_notes=10_000, matching_notes=20, c=0.1)
    assert d.order == "filter_first"
    assert d.candidates == 2500
    assert d.limit == 1000


def test_loose_filter_goes_search_first():
    # 5 * 10000 / 5000 = 10 candidates <= 0.1 * 10000
    d = choose_order(k=5, total_notes=10_000, matching_notes=5000, c=0.1)
    assert d.order == "search_first"
    assert d.candidates == 10


def test_zero_matches_goes_filter_first():
    d = choose_order(k=5, total_notes=10_000, matching_notes=0)
    assert d.order == "filter_first"
    assert d.candidates is None


def test_threshold_boundary_is_relative():
    # 5 * 10000 / 50 = exactly 1000 = 0.1 * 10000: search_first at the boundary.
    at_limit = choose_order(k=5, total_notes=10_000, matching_notes=50, c=0.1)
    over_limit = choose_order(k=5, total_notes=10_000, matching_notes=50, c=0.0999)
    assert at_limit.candidates == 1000
    assert at_limit.order == "search_first"
    assert over_limit.order == "filter_first"


def test_same_fraction_switches_at_the_same_matching_count_for_any_index_size():
    # search_first iff f >= beta * k / c = 1 * 5 / 0.25 = 20 matching notes.
    assert switch_matching_notes(5, 0.25, 1.0) == 20
    for total in (100, 1_000, 100_000):
        assert choose_order(5, total, 20, c=0.25).order == "search_first"
        assert choose_order(5, total, 19, c=0.25).order == "filter_first"


def test_beta_scales_candidates_and_moves_the_switch():
    # beta 2: candidates ceil(2 * 5 * 1000 / 30) = 334 > 0.25 * 1000 -> filter_first;
    # beta 1: 167 <= 250 -> search_first.
    plain = choose_order(5, 1000, 30, c=0.25, beta=1.0)
    corrected = choose_order(5, 1000, 30, c=0.25, beta=2.0)
    assert (plain.candidates, plain.order) == (167, "search_first")
    assert (corrected.candidates, corrected.order) == (334, "filter_first")
    assert search_first_candidates(5, 1000, 30, beta=1.5) == 250


def test_decision_keeps_inputs():
    d = choose_order(k=3, total_notes=100, matching_notes=10, c=0.5, beta=1.5)
    assert (d.k, d.total_notes, d.matching_notes, d.c, d.beta) == (3, 100, 10, 0.5, 1.5)


def test_candidates_round_up():
    # 5 * 10 / 3 = 16.67
    assert search_first_candidates(5, 10, 3) == 17


def test_candidates_to_fetch_caps_at_total_and_applies_beta():
    assert candidates_to_fetch(5, 10, 3) == 10
    assert candidates_to_fetch(5, 10, 0) == 10
    assert candidates_to_fetch(5, 10_000, 5000) == 10
    assert candidates_to_fetch(5, 10_000, 5000, beta=3.0) == 30


@pytest.mark.parametrize(
    "k,total,matching",
    [(0, 10, 5), (5, 10, 11), (5, -1, 0), (5, 10, -1)],
)
def test_invalid_counts_raise(k, total, matching):
    with pytest.raises(ValueError):
        choose_order(k=k, total_notes=total, matching_notes=matching)


def test_invalid_c_or_beta_raise():
    with pytest.raises(ValueError):
        choose_order(5, 100, 10, c=0)
    with pytest.raises(ValueError):
        search_first_candidates(5, 100, 10, beta=0)
