"""Cheap probes that feed the cost model. Nothing here fetches evidence.

Probes are lazy: only questions with an order choice (a structured filter or
category combined with a note search) pay for relational counts, note counts
and the trial note search. Other questions read only index metadata
(total_notes, and the note count of a per-admission question's one admission),
and their plans are costed with what is known.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from clinrag.config import PlannerConfig
from clinrag.question import GRAPH, NOTES, RELATIONAL, QuestionSpec
from clinrag.sources.base import RelationalQuery, Sources

# Names recorded in Probes.ran.
GRAPH_EXPAND = "graph_expand"
RELATIONAL_COUNTS_SQL = "relational_counts_sql"  # rows, admissions and notes in one query
RELATIONAL_COUNT = "relational_count"
MATCHING_IDS = "matching_ids"
NOTE_COUNT_SQL = "note_count_sql"
NOTE_COUNT_FOR_IDS = "note_count_for_ids"
TOTAL_NOTES = "total_notes"
SCOPED_NOTES = "scoped_notes"
TRIAL_SEARCH = "trial_search"


@dataclass(frozen=True)
class Probes:
    """Every number the cost model uses for one question. None means not probed.

    relational_rows:     rows the question's relational query would return.
    matching_admissions: distinct admissions in those rows.
    total_notes:         note chunks in the index.
    matching_notes:      note chunks belonging to the matching admissions.
    scoped_notes:        note chunks of the question's note_hadm_id admission, for
                         per-admission note questions.
    probe_scores:        top probe_k similarity scores for the note query.
    graph_codes:         ICD codes under the question's category.
    probe_ms:            wall time spent probing.
    ran:                 names of the probes that ran, in order.
    """

    relational_rows: int | None = None
    matching_admissions: int | None = None
    total_notes: int | None = None
    matching_notes: int | None = None
    scoped_notes: int | None = None
    probe_scores: tuple[float, ...] = ()
    graph_codes: int | None = None
    probe_ms: float = 0.0
    ran: tuple[str, ...] = ()


def has_order_choice(question: QuestionSpec) -> bool:
    """True when a structured filter or category restricts a note search, so filter-first vs search-first is open."""
    return {RELATIONAL, NOTES} <= question.required_sources


def run_probes(question: QuestionSpec, sources: Sources, config: PlannerConfig) -> Probes:
    """Probe what the decision needs.

    Questions with an order choice:
      - graph:      expand the category (cached in the graph source); graph_codes
                    is the number of codes.
      - relational: rows, matching admissions and matching notes. In one query
                    when the source implements count_matches; otherwise count plus
                    matching_ids, with notes from count_notes (one SQL count, no
                    id list) or, failing that, notes.count_notes_for(ids).
      - notes:      total chunk count and a trial probe_search with probe_k.
    Questions without one (a single source, or no filter combined with notes):
      no relational counts, note counts or trial search. Only total_notes (index
      metadata, needed to cost unfiltered searches and query_all) and, for a
      per-admission question, that admission's note count are read.
    """
    start = time.perf_counter()
    required = question.required_sources
    ran: list[str] = []
    graph_codes = relational_rows = matching_admissions = None
    total_notes = matching_notes = scoped_notes = None
    probe_scores: tuple[float, ...] = ()

    if has_order_choice(question):
        codes: tuple[str, ...] = ()
        if GRAPH in required:
            assert question.category is not None
            codes = tuple(sources.require_graph().expand(question.category))
            graph_codes = len(codes)
            ran.append(GRAPH_EXPAND)

        relational = sources.require_relational()
        notes = sources.require_notes()
        query = RelationalQuery(lab=question.relational_filter, icd_codes=codes)
        try:
            relational_rows, matching_admissions, matching_notes = relational.count_matches(query)
            ran.append(RELATIONAL_COUNTS_SQL)
        except NotImplementedError:
            relational_rows = relational.count(query)
            ids = relational.matching_ids(query)
            matching_admissions = len(ids)
            ran += [RELATIONAL_COUNT, MATCHING_IDS]
            try:
                matching_notes = relational.count_notes(query)
                ran.append(NOTE_COUNT_SQL)
            except NotImplementedError:
                matching_notes = notes.count_notes_for(ids)
                ran.append(NOTE_COUNT_FOR_IDS)

        assert question.note_query is not None
        total_notes = notes.total_notes()
        probe_scores = tuple(notes.probe_search(question.note_query, config.probe_k))
        ran += [TOTAL_NOTES, TRIAL_SEARCH]
    elif sources.notes is not None:
        total_notes = sources.notes.total_notes()
        ran.append(TOTAL_NOTES)
        if NOTES in required and question.note_hadm_id is not None:
            scoped_notes = sources.notes.count_notes_for([question.note_hadm_id])
            ran.append(SCOPED_NOTES)

    return Probes(
        relational_rows=relational_rows,
        matching_admissions=matching_admissions,
        total_notes=total_notes,
        matching_notes=matching_notes,
        scoped_notes=scoped_notes,
        probe_scores=probe_scores,
        graph_codes=graph_codes,
        probe_ms=(time.perf_counter() - start) * 1000.0,
        ran=tuple(ran),
    )
