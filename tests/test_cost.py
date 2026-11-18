"""Cost estimates checked against hand-computed numbers.

Config: k=5, alpha=0.1, 10 tokens/row, 200 tokens/chunk, relational 20 ms,
graph 2 ms, note search 8 ms per 1000 candidates.
Probes: 30 relational rows, 10000 notes, 40 matching notes, 6 graph codes.
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
    note_search_ms_per_1k_candidates=8,
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


def test_relational_only():
    # latency 20; tokens 30 * 10 = 300; total 20 + 0.1 * 300 = 50
    cost = estimate_cost(Plan("relational_only", (ROWS,)), PROBES, CONFIG)
    assert cost.est_latency_ms == pytest.approx(20)
    assert cost.est_tokens == pytest.approx(300)
    assert cost.total == pytest.approx(50)


def test_notes_only():
    # 5 candidates -> 8 * 5 / 1000 = 0.04 ms; tokens 5 * 200 = 1000; total 0.04 + 100
    cost = estimate_cost(Plan("notes_only", (NoteSearch("q"),)), PROBES, CONFIG)
    assert cost.est_latency_ms == pytest.approx(0.04)
    assert cost.est_tokens == pytest.approx(1000)
    assert cost.total == pytest.approx(100.04)


def test_filter_first():
    # relational 20 ms, no tokens; 40 candidates -> 0.32 ms; 5 chunks -> 1000 tokens
    plan = Plan("rn:ff", (FILTER_ONLY, NoteSearch("q", "filter_first")))
    cost = estimate_cost(plan, PROBES, CONFIG)
    assert cost.est_latency_ms == pytest.approx(20.32)
    assert cost.est_tokens == pytest.approx(1000)
    assert cost.total == pytest.approx(120.32)
    assert cost.steps[1].details["candidates"] == 40


def test_search_first():
    # ceil(5 * 10000 / 40) = 1250 candidates -> 10 ms; latency 30; total 30 + 100
    plan = Plan("rn:sf", (FILTER_ONLY, NoteSearch("q", "search_first")))
    cost = estimate_cost(plan, PROBES, CONFIG)
    assert cost.steps[1].details["candidates"] == 1250
    assert cost.est_latency_ms == pytest.approx(30)
    assert cost.total == pytest.approx(130)


def test_both_unfiltered():
    # latency 20 + 0.04; tokens 300 + 1000; total 20.04 + 130
    plan = Plan("rn:unf", (ROWS, NoteSearch("q")))
    cost = estimate_cost(plan, PROBES, CONFIG)
    assert cost.est_tokens == pytest.approx(1300)
    assert cost.total == pytest.approx(150.04)


def test_graph_then_relational():
    # latency 2 + 20; tokens 300; total 22 + 30
    plan = Plan("g>r", (GraphExpand("renal"), RelationalFilter(None, use_graph_codes=True)))
    cost = estimate_cost(plan, PROBES, CONFIG)
    assert cost.est_latency_ms == pytest.approx(22)
    assert cost.total == pytest.approx(52)
    assert cost.steps[0].est_rows == 6
    assert cost.steps[0].est_tokens == 0


def test_search_first_with_no_matches_scans_everything():
    # 0 matching notes: fetch all 10000 -> 80 ms, keep nothing
    probes = Probes(relational_rows=0, total_notes=10_000, matching_notes=0)
    plan = Plan("rn:sf", (FILTER_ONLY, NoteSearch("q", "search_first")))
    cost = estimate_cost(plan, probes, CONFIG)
    assert cost.est_latency_ms == pytest.approx(20 + 80)
    assert cost.est_tokens == 0


def test_alpha_weights_tokens():
    cfg = PlannerConfig(alpha=2.0, avg_tokens_per_row=10, relational_ms=20)
    cost = estimate_cost(Plan("relational_only", (ROWS,)), PROBES, cfg)
    assert cost.total == pytest.approx(20 + 2.0 * 300)


def test_missing_probe_raises():
    with pytest.raises(ValueError, match="matching_notes"):
        estimate_cost(
            Plan("rn:ff", (FILTER_ONLY, NoteSearch("q", "filter_first"))),
            Probes(relational_rows=1, total_notes=10),
            CONFIG,
        )
