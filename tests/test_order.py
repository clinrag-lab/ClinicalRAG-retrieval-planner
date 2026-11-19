import pytest

from clinrag.planner.order import candidates_to_fetch, choose_order, search_first_candidates


def test_selective_filter_goes_filter_first():
    # 20 / 10000 = 0.2% of the index < s = 10%
    d = choose_order(k=5, total_notes=10_000, matching_notes=20, s=0.1)
    assert d.order == "filter_first"
    assert d.candidates == 2500
    assert d.selectivity == pytest.approx(0.002)


def test_loose_filter_goes_search_first():
    d = choose_order(k=5, total_notes=10_000, matching_notes=5000, s=0.1)
    assert d.order == "search_first"
    assert d.candidates == 10


def test_zero_matches_goes_filter_first():
    d = choose_order(k=5, total_notes=10_000, matching_notes=0)
    assert d.order == "filter_first"
    assert d.candidates is None


def test_threshold_boundary():
    # Exactly s of the index counts as search_first.
    assert choose_order(5, 1000, 100, s=0.1).order == "search_first"
    assert choose_order(5, 1000, 99, s=0.1).order == "filter_first"


def test_switch_is_the_same_selectivity_at_any_index_size():
    # 5% of the index: filter_first at s = 0.1 whether the index has 200 or 200,000 notes,
    # unlike a fixed matching-note count.
    for total in (200, 2_000, 200_000):
        assert choose_order(5, total, total // 20, s=0.1).order == "filter_first"
        assert choose_order(5, total, total // 5, s=0.1).order == "search_first"


def test_beta_only_scales_the_candidate_count():
    plain = choose_order(5, 1000, 30, s=0.02, beta=1.0)
    corrected = choose_order(5, 1000, 30, s=0.02, beta=2.0)
    assert plain.order == corrected.order == "search_first"
    assert (plain.candidates, corrected.candidates) == (167, 334)
    assert search_first_candidates(5, 1000, 30, beta=1.5) == 250


def test_decision_keeps_inputs():
    d = choose_order(k=3, total_notes=100, matching_notes=10, s=0.2, beta=1.5)
    assert (d.k, d.total_notes, d.matching_notes, d.s, d.beta) == (3, 100, 10, 0.2, 1.5)


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


def test_invalid_s_or_beta_raise():
    with pytest.raises(ValueError):
        choose_order(5, 100, 10, s=0)
    with pytest.raises(ValueError):
        choose_order(5, 100, 10, s=1.5)
    with pytest.raises(ValueError):
        search_first_candidates(5, 100, 10, beta=0)
