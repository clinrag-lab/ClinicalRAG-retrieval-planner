import pytest

from clinrag.config import PlannerConfig
from clinrag.planner.enumerate import enumerate_plans
from clinrag.planner.execute import execute
from clinrag.planner.plan import NoteSearch, Plan, RelationalFilter
from clinrag.question import QuestionSpec, StructuredFilter
from clinrag.sources.base import Sources
from clinrag.sources.tokens import count_tokens
from tests.fakes import CREATININE, FakeGraph, FakeNote, FakeNotes, FakeRelational, creatinine_labs, grid_notes

CONFIG = PlannerConfig(top_k=3)
HIGH = StructuredFilter(CREATININE, ">", 2.0)


def plans_by_name(question):
    return {p.name: p for p in enumerate_plans(question)}


@pytest.fixture
def sources():
    # 10 admissions x 4 notes; admissions 7, 8, 9 have high creatinine.
    # Admissions 1 and 3 have renal codes.
    return Sources(
        relational=FakeRelational(creatinine_labs(10, high={7, 8, 9}), [(1, "N179"), (3, "N186"), (5, "I10")]),
        notes=FakeNotes(grid_notes(10, 4, text="patient started on dialysis today")),
        graph=FakeGraph({"renal": ["N179", "N186"]}),
    )


@pytest.fixture
def mixed():
    return QuestionSpec(
        qid="q",
        text="text",
        qclass="C",
        relational_filter=HIGH,
        note_query="dialysis",
        required_sources=frozenset({"relational", "notes"}),
    )


def test_graph_codes_flow_into_relational(sources):
    q = QuestionSpec(qid="g", text="t", qclass="B", category="renal", required_sources=frozenset({"graph", "relational"}))
    result = execute(plans_by_name(q)["graph>relational"], sources, q, CONFIG)

    assert [s.step for s in result.steps] == ["graph", "relational"]
    assert {item["hadm_id"] for item in result.items} == {1, 3}
    assert result.sources_queried == {"graph", "relational"}


def test_filter_first_only_searches_matching_admissions(sources, mixed):
    result = execute(plans_by_name(mixed)["relational+notes:filter_first"], sources, mixed, CONFIG)

    assert sources.notes.search_args == [(3, (7, 8, 9))]
    note_items = [i for i in result.items if i["source"] == "notes"]
    assert len(note_items) == 3
    assert {i["hadm_id"] for i in note_items} <= {7, 8, 9}
    # Relational rows were only a filter, so they are not evidence.
    assert all(i["source"] == "notes" for i in result.items)


def test_search_first_fetches_candidates_then_filters(sources, mixed):
    plan = plans_by_name(mixed)["relational+notes:search_first"].with_candidates(40)
    result = execute(plan, sources, mixed, CONFIG)

    assert sources.notes.search_args == [(40, None)]
    assert len(result.items) == 3
    assert {i["hadm_id"] for i in result.items} <= {7, 8, 9}
    # Highest-scoring matching notes: admission 7's first three.
    assert [i["note_id"] for i in result.items] == ["7-0", "7-1", "7-2"]


def test_search_first_without_candidates_raises(sources, mixed):
    with pytest.raises(ValueError, match="n_candidates"):
        execute(plans_by_name(mixed)["relational+notes:search_first"], sources, mixed, CONFIG)


def test_tokens_add_up(sources, mixed):
    result = execute(plans_by_name(mixed)["query_all"], sources, mixed, CONFIG)

    per_item = sum(count_tokens(i["text"]) for i in result.items)
    assert result.tokens == per_item == sum(s.tokens for s in result.steps)
    # 3 lab rows ("lab itemid 50912 value 3.0" = 5 words -> 7 tokens)
    # + 3 notes (5 words -> 7 tokens each)
    assert result.tokens == 3 * 7 + 3 * 7


def test_latency_recorded_per_step_and_total(sources, mixed):
    result = execute(plans_by_name(mixed)["relational+notes:unfiltered"], sources, mixed, CONFIG)
    assert len(result.steps) == 2
    assert all(s.latency_ms >= 0 for s in result.steps)
    assert result.total_latency_ms >= sum(s.latency_ms for s in result.steps)


def test_filtered_search_without_relational_step_is_rejected():
    with pytest.raises(ValueError):
        Plan("bad", (NoteSearch("q", "filter_first"),))
    with pytest.raises(ValueError):
        Plan("bad", (RelationalFilter(None, use_graph_codes=True),))


def test_empty_filter_returns_no_notes():
    sources = Sources(
        relational=FakeRelational(creatinine_labs(3, high=set())),
        notes=FakeNotes([FakeNote("1-0", 1, "a b c", 0.9)]),
    )
    q = QuestionSpec(
        qid="e", text="t", qclass="C", relational_filter=HIGH, note_query="x",
        required_sources=frozenset({"relational", "notes"}),
    )
    result = execute(plans_by_name(q)["relational+notes:filter_first"], sources, q, CONFIG)
    assert result.items == []
    assert result.tokens == 0


def test_note_hadm_id_scopes_search_to_one_admission(sources):
    q = QuestionSpec(
        qid="b",
        text="According to the discharge summary for admission 5, was dialysis performed?",
        qclass="B",
        note_query="dialysis",
        note_hadm_id=5,
        required_sources=frozenset({"notes"}),
    )
    result = execute(plans_by_name(q)["notes_only"], sources, q, CONFIG)

    assert sources.notes.search_args == [(3, (5,))]
    assert {i["hadm_id"] for i in result.items} == {5}
    assert result.sources_queried == {"notes"}
    assert sources.relational.calls == []


def test_note_hadm_id_only_on_notes_only_questions():
    with pytest.raises(ValueError, match="note_hadm_id"):
        QuestionSpec(
            qid="x",
            text="t",
            qclass="C",
            relational_filter=HIGH,
            note_query="dialysis",
            note_hadm_id=5,
            required_sources=frozenset({"relational", "notes"}),
        )
