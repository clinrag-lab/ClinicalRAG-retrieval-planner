"""Enumerate every candidate plan for a question's required sources."""

from __future__ import annotations

from clinrag.planner.plan import GraphExpand, NoteSearch, Plan, RelationalFilter, Step
from clinrag.question import GRAPH, NOTES, RELATIONAL, QuestionSpec

QUERY_ALL = "query_all"


def enumerate_plans(question: QuestionSpec) -> list[Plan]:
    """Candidate plans that cover every required source, then the query_all baseline.

    required sources          plans
    relational                relational_only
    notes                     notes_only
    relational, notes         relational+notes:filter_first, :search_first, :unfiltered
    graph, relational         graph>relational
    graph, relational, notes  graph>relational+notes:filter_first, :search_first
    (always)                  query_all

    Targeted plans come before query_all, so the planner prefers them on cost ties.
    """
    required = question.required_sources
    uses_graph = GRAPH in required
    note_query = question.note_query or ""

    rows_as_evidence = RelationalFilter(question.relational_filter, use_graph_codes=uses_graph, as_evidence=True)
    rows_as_filter = RelationalFilter(question.relational_filter, use_graph_codes=uses_graph, as_evidence=False)
    graph = (GraphExpand(question.category),) if question.category is not None else ()

    plans: list[Plan] = []
    if required == {RELATIONAL}:
        plans.append(Plan("relational_only", (rows_as_evidence,)))
    elif required == {NOTES}:
        plans.append(Plan("notes_only", (NoteSearch(note_query),)))
    elif required == {RELATIONAL, NOTES}:
        plans += [
            Plan("relational+notes:filter_first", (rows_as_filter, NoteSearch(note_query, "filter_first"))),
            Plan("relational+notes:search_first", (rows_as_filter, NoteSearch(note_query, "search_first"))),
            Plan("relational+notes:unfiltered", (rows_as_evidence, NoteSearch(note_query))),
        ]
    elif required == {GRAPH, RELATIONAL}:
        plans.append(Plan("graph>relational", (*graph, rows_as_evidence)))
    elif required == {GRAPH, RELATIONAL, NOTES}:
        plans += [
            Plan("graph>relational+notes:filter_first", (*graph, rows_as_filter, NoteSearch(note_query, "filter_first"))),
            Plan("graph>relational+notes:search_first", (*graph, rows_as_filter, NoteSearch(note_query, "search_first"))),
        ]
    else:  # QuestionSpec validation rules out other combinations
        raise ValueError(f"{question.qid}: unsupported source combination {sorted(required)}")

    plans.append(query_all_plan(question))
    return plans


def query_all_plan(question: QuestionSpec) -> Plan:
    """Baseline: query every source the question gives an input for, notes unfiltered.

    Graph codes still feed the relational step (nothing else can use them), but
    nothing restricts the note search, and the note search always runs, using
    the question text when there is no note_query. Without a category the
    baseline has no graph step, since there is nothing to expand.
    """
    steps: list[Step] = []
    if question.category is not None:
        steps.append(GraphExpand(question.category))
    if question.relational_filter is not None or question.category is not None:
        steps.append(
            RelationalFilter(
                question.relational_filter,
                use_graph_codes=question.category is not None,
                as_evidence=True,
            )
        )
    steps.append(NoteSearch(question.note_query or question.text))
    return Plan(QUERY_ALL, tuple(steps))
