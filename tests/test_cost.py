"""Cost estimates checked against hand-computed numbers.

Config: k=5, alpha=0.1, 10 tokens/row, 200 tokens/chunk, relational 20 ms,
graph 2 ms, scan 0.5 ms per 1000 notes, fetch 2 ms per 1000 candidates,
id filter 1 ms per 1000 ids.
Probes: 30 relational rows over 12 admissions, 10000 notes, 40 matching notes,
6 graph codes.
"""

import pytest

from clinrag.config import PlannerConfig
from clinrag.planner.cost import estimate_cost
from clinrag.planner.plan import GraphExpand, NoteSearch, Plan, RelationalFilter
from clinrag.planner.probes import Probes
from clinrag.question import StructuredFilter

CONFIG = PlannerConfig(
    top_k=5,
    alpha=0.1,
    avg_tokens_per_row=10,
    avg_tokens_per_chunk=200,
    relational_ms=20,
    graph_ms=2,
    note_scan_ms_per_1k_notes=0.5,
    note_fetch_ms_per_1k_candidates=2.0,
    id_filter_ms_per_1k_ids=1.0,
)
PROBES = Probes(
    relational_rows=30,
    matching_admissions=12,
    total_notes=10_000,
    matching_notes=40,
    graph_codes=6,
)
LAB = StructuredFilter(50912, ">", 2.0)
ROWS = RelationalFilter(LAB)
FILTER_ONLY = RelationalFilter(LAB, as_evidence=False)
FILTER_FIRST = Plan("rn:ff", (FILTER_ONLY, NoteSearch("q", "filter_first")))
SEARCH_FIRST = Plan("rn:sf", (FILTER_ONLY, NoteSearch("q", "search_first")))
NOTES_ONLY = Plan("notes_only", (NoteSearch("q"),))


def note_latency(plan, probes):
    return estimate_cost(plan, probes, CONFIG).steps[-1].est_latency_ms


def test_relational_only():
    # latency 20; tokens 30 * 10 = 300; total 20 + 0.1 * 300 = 50
    cost = estimate_cost(Plan("relational_only", (ROWS,)), PROBES, CONFIG)
    assert cost.est_latency_ms == pytest.approx(20)
    assert cost.est_tokens == pytest.approx(300)
    assert cost.total == pytest.approx(50)


def test_notes_only():
    # scan(10000) = 5.0, fetch(5) = 0.01 -> 5.01 ms; tokens 5 * 200 = 1000; total 5.01 + 100
    cost = estimate_cost(NOTES_ONLY, PROBES, CONFIG)
    assert cost.est_latency_ms == pytest.approx(5.01)
    assert cost.est_tokens == pytest.approx(1000)
    assert cost.total == pytest.approx(105.01)


def test_filter_first():
    # relational 20 ms, no tokens
    # id_filter(12) = 0.012, scan(40) = 0.02, fetch(5) = 0.01 -> 0.042 ms; 5 chunks -> 1000 tokens
    cost = estimate_cost(FILTER_FIRST, PROBES, CONFIG)
    assert cost.est_latency_ms == pytest.approx(20.042)
    assert cost.est_tokens == pytest.approx(1000)
    assert cost.total == pytest.approx(120.042)
    assert cost.steps[1].details == {
        "step": "notes:filter_first",
        "scanned": 40,
        "fetched": 5,
        "filter_ids": 12,
    }


def test_search_first():
    # ceil(5 * 10000 / 40) = 1250 candidates
    # scan(10000) = 5.0, fetch(1250) = 2.5 -> 7.5 ms; latency 27.5; total 27.5 + 100
    cost = estimate_cost(SEARCH_FIRST, PROBES, CONFIG)
    assert cost.steps[1].details["scanned"] == 10_000
    assert cost.steps[1].details["fetched"] == 1250
    assert cost.est_latency_ms == pytest.approx(27.5)
    assert cost.total == pytest.approx(127.5)


def test_both_unfiltered():
    # latency 20 + 5.01; tokens 300 + 1000; total 25.01 + 130
    plan = Plan("rn:unf", (ROWS, NoteSearch("q")))
    cost = estimate_cost(plan, PROBES, CONFIG)
    assert cost.est_latency_ms == pytest.approx(25.01)
    assert cost.est_tokens == pytest.approx(1300)
    assert cost.total == pytest.approx(155.01)


def test_graph_then_relational():
    # latency 2 + 20; tokens 300; total 22 + 30
    plan = Plan("g>r", (GraphExpand("renal"), RelationalFilter(None, use_graph_codes=True)))
    cost = estimate_cost(plan, PROBES, CONFIG)
    assert cost.est_latency_ms == pytest.approx(22)
    assert cost.total == pytest.approx(52)
    assert cost.steps[0].est_rows == 6
    assert cost.steps[0].est_tokens == 0


def test_search_first_with_no_matches_scans_and_fetches_everything():
    # 0 matching notes: scan(10000) = 5.0 + fetch(10000) = 20.0; keep nothing
    probes = Probes(relational_rows=0, matching_admissions=0, total_notes=10_000, matching_notes=0)
    cost = estimate_cost(SEARCH_FIRST, probes, CONFIG)
    assert cost.est_latency_ms == pytest.approx(20 + 25)
    assert cost.est_tokens == 0


def test_unfiltered_latency_grows_with_total_notes():
    # scan(1000) + fetch(5) = 0.51 ms vs scan(100000) + fetch(5) = 50.01 ms
    small = note_latency(NOTES_ONLY, Probes(total_notes=1_000))
    large = note_latency(NOTES_ONLY, Probes(total_notes=100_000))
    assert small == pytest.approx(0.51)
    assert large == pytest.approx(50.01)
    assert large > small


def test_selective_filter_filter_first_is_faster():
    # 40 of 10000 notes match: filter_first 0.042 ms vs search_first 7.5 ms
    ff = note_latency(FILTER_FIRST, PROBES)
    sf = note_latency(SEARCH_FIRST, PROBES)
    assert ff == pytest.approx(0.042)
    assert sf == pytest.approx(7.5)
    assert ff < sf


def test_loose_filter_search_first_is_faster():
    # 9000 of 10000 notes match, over 1800 admissions.
    # filter_first: id_filter(1800) = 1.8 + scan(9000) = 4.5 + fetch(5) = 0.01 -> 6.31 ms
    # search_first: ceil(50000 / 9000) = 6; scan(10000) = 5.0 + fetch(6) = 0.012 -> 5.012 ms
    probes = Probes(relational_rows=1800, matching_admissions=1800, total_notes=10_000, matching_notes=9000)
    ff = note_latency(FILTER_FIRST, probes)
    sf = note_latency(SEARCH_FIRST, probes)
    assert ff == pytest.approx(6.31)
    assert sf == pytest.approx(5.012)
    assert sf < ff


def test_alpha_weights_tokens():
    cfg = PlannerConfig(alpha=2.0, avg_tokens_per_row=10, relational_ms=20)
    cost = estimate_cost(Plan("relational_only", (ROWS,)), PROBES, cfg)
    assert cost.total == pytest.approx(20 + 2.0 * 300)


def test_missing_probe_raises():
    with pytest.raises(ValueError, match="matching_notes"):
        estimate_cost(FILTER_FIRST, Probes(relational_rows=1, total_notes=10), CONFIG)
    with pytest.raises(ValueError, match="matching_admissions"):
        estimate_cost(FILTER_FIRST, Probes(relational_rows=1, total_notes=10, matching_notes=5), CONFIG)


# ---- per-call fixed costs --------------------------------------------------------------

FIXED = PlannerConfig(
    **{**CONFIG.__dict__, "relational_fixed_ms": 3.0, "graph_fixed_ms": 0.5, "note_search_fixed_ms": 1.5}
)


def test_fixed_costs_default_to_zero():
    config = PlannerConfig()
    assert (config.relational_fixed_ms, config.graph_fixed_ms, config.note_search_fixed_ms) == (0.0, 0.0, 0.0)


def test_fixed_costs_added_once_per_step():
    # graph 0.5 + 2; relational 3 + 20; filter_first note search
    # 1.5 + id_filter(12) 0.012 + scan(40) 0.02 + fetch(5) 0.01 = 1.542
    plan = Plan("g>rn:ff", (GraphExpand("renal"), RelationalFilter(LAB, use_graph_codes=True, as_evidence=False),
                            NoteSearch("q", "filter_first")))
    cost = estimate_cost(plan, PROBES, FIXED)
    assert [s.est_latency_ms for s in cost.steps] == pytest.approx([2.5, 23.0, 1.542])
    assert cost.est_latency_ms == pytest.approx(2.5 + 23.0 + 1.542)


def test_fixed_note_cost_applies_to_every_search_mode():
    # Each mode's latency rises by exactly note_search_fixed_ms.
    for plan in (NOTES_ONLY, FILTER_FIRST, SEARCH_FIRST):
        plain = estimate_cost(plan, PROBES, CONFIG).steps[-1].est_latency_ms
        fixed = estimate_cost(plan, PROBES, FIXED).steps[-1].est_latency_ms
        assert fixed - plain == pytest.approx(1.5)
    # Tokens are unaffected.
    assert estimate_cost(NOTES_ONLY, PROBES, FIXED).est_tokens == estimate_cost(NOTES_ONLY, PROBES, CONFIG).est_tokens
