import pytest

from clinrag.planner.enumerate import QUERY_ALL, enumerate_plans
from clinrag.planner.plan import GraphExpand, NoteSearch, RelationalFilter
from clinrag.question import QuestionSpec, StructuredFilter

LAB = StructuredFilter(50912, ">", 2.0)


def question(sources, **kwargs):
    return QuestionSpec(qid="q", text="question text", qclass="A", required_sources=frozenset(sources), **kwargs)


QUESTIONS = {
    "relational": question({"relational"}, relational_filter=LAB),
    "notes": question({"notes"}, note_query="dialysis"),
    "relational+notes": question({"relational", "notes"}, relational_filter=LAB, note_query="dialysis"),
    "graph+relational": question({"graph", "relational"}, category="renal"),
    "graph+relational+notes": question({"graph", "relational", "notes"}, category="renal", note_query="dialysis"),
}

EXPECTED_NAMES = {
    "relational": ["relational_only", QUERY_ALL],
    "notes": ["notes_only", QUERY_ALL],
    "relational+notes": [
        "relational+notes:filter_first",
        "relational+notes:search_first",
        "relational+notes:unfiltered",
        QUERY_ALL,
    ],
    "graph+relational": ["graph>relational", QUERY_ALL],
    "graph+relational+notes": [
        "graph>relational+notes:filter_first",
        "graph>relational+notes:search_first",
        QUERY_ALL,
    ],
}


@pytest.mark.parametrize("key", QUESTIONS)
def test_plan_names_per_source_set(key):
    assert [p.name for p in enumerate_plans(QUESTIONS[key])] == EXPECTED_NAMES[key]


@pytest.mark.parametrize("key", QUESTIONS)
def test_query_all_last_and_unfiltered(key):
    plans = enumerate_plans(QUESTIONS[key])
    baseline = plans[-1]
    assert baseline.name == QUERY_ALL
    assert "notes" in baseline.sources
    assert baseline.note_search.mode == "unfiltered"


@pytest.mark.parametrize("key", QUESTIONS)
def test_every_plan_covers_required_sources(key):
    q = QUESTIONS[key]
    for plan in enumerate_plans(q):
        assert q.required_sources <= plan.sources, plan.name


def test_total_plan_count():
    assert sum(len(enumerate_plans(q)) for q in QUESTIONS.values()) == 13


def test_query_all_uses_question_text_without_note_query():
    baseline = enumerate_plans(QUESTIONS["relational"])[-1]
    assert baseline.sources == {"relational", "notes"}
    assert baseline.note_search.query == "question text"


def test_graph_pipeline_steps():
    plan = enumerate_plans(QUESTIONS["graph+relational+notes"])[0]
    graph, relational, notes = plan.steps
    assert isinstance(graph, GraphExpand) and graph.category == "renal"
    assert isinstance(relational, RelationalFilter)
    assert relational.use_graph_codes and not relational.as_evidence
    assert isinstance(notes, NoteSearch) and notes.mode == "filter_first"


def test_question_rejects_inconsistent_sources():
    with pytest.raises(ValueError):
        question({"notes"})  # notes without a note_query
    with pytest.raises(ValueError):
        question({"graph", "notes"}, category="renal", note_query="x")  # graph without relational
    with pytest.raises(ValueError):
        question({"relational"}, relational_filter=LAB, category="renal")  # category without graph
