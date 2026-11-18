"""Cheap probes that feed the cost model. Nothing here fetches evidence."""

from __future__ import annotations

import time
from dataclasses import dataclass

from clinrag.config import PlannerConfig
from clinrag.question import GRAPH, NOTES, RELATIONAL, QuestionSpec
from clinrag.sources.base import RelationalQuery, Sources


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
    """

    relational_rows: int | None = None
    matching_admissions: int | None = None
    total_notes: int | None = None
    matching_notes: int | None = None
    scoped_notes: int | None = None
    probe_scores: tuple[float, ...] = ()
    graph_codes: int | None = None
    probe_ms: float = 0.0


def run_probes(question: QuestionSpec, sources: Sources, config: PlannerConfig) -> Probes:
    """Probe only the sources the question needs, using count-style operations.

    - graph:      expand the category (in memory) to get the codes the relational
                  count needs; graph_codes is their number.
    - relational: count rows and list matching admission ids.
    - notes:      total chunk count (also needed to cost the query_all baseline,
                  so read whenever a notes source exists), probe_search with
                  probe_k, and count_notes_for on the matching ids when the
                  question also has a relational filter, or on [note_hadm_id]
                  when the question is scoped to one admission's notes.
    """
    start = time.perf_counter()
    required = question.required_sources

    codes: tuple[str, ...] = ()
    graph_codes = None
    if GRAPH in required:
        assert question.category is not None
        codes = tuple(sources.require_graph().expand(question.category))
        graph_codes = len(codes)

    relational_rows = matching_admissions = None
    matching_ids: list[int] = []
    if RELATIONAL in required:
        relational = sources.require_relational()
        query = RelationalQuery(lab=question.relational_filter, icd_codes=codes)
        relational_rows = relational.count(query)
        matching_ids = relational.matching_ids(query)
        matching_admissions = len(matching_ids)

    total_notes = matching_notes = scoped_notes = None
    probe_scores: tuple[float, ...] = ()
    if NOTES in required:
        notes = sources.require_notes()
        assert question.note_query is not None
        total_notes = notes.total_notes()
        probe_scores = tuple(notes.probe_search(question.note_query, config.probe_k))
        if RELATIONAL in required:
            matching_notes = notes.count_notes_for(matching_ids)
        if question.note_hadm_id is not None:
            scoped_notes = notes.count_notes_for([question.note_hadm_id])
    elif sources.notes is not None:
        total_notes = sources.notes.total_notes()

    return Probes(
        relational_rows=relational_rows,
        matching_admissions=matching_admissions,
        total_notes=total_notes,
        matching_notes=matching_notes,
        scoped_notes=scoped_notes,
        probe_scores=probe_scores,
        graph_codes=graph_codes,
        probe_ms=(time.perf_counter() - start) * 1000.0,
    )
