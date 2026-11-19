"""Note searches return the top k distinct admissions, for every search mode and any NotesSource."""

from pathlib import Path

from clinrag.config import PlannerConfig
from clinrag.planner.cost import estimate_cost
from clinrag.planner.execute import distinct_admissions, distinct_search, execute
from clinrag.planner.plan import NoteSearch, Plan, RelationalFilter
from clinrag.planner.probes import Probes
from clinrag.question import QuestionSpec, StructuredFilter
from clinrag.sources.base import Sources
from clinrag.sources.reference import build_reference_sources
from clinrag.workload import notes_gen
from tests.fakes import CREATININE, FakeNotes, FakeRelational, creatinine_labs, grid_notes

FIXTURES = Path(__file__).parent / "fixtures"
HIGH = StructuredFilter(CREATININE, ">", 2.0)


def test_distinct_admissions_keeps_the_best_note_of_each():
    items = [{"hadm_id": h, "score": s} for h, s in [(1, 0.9), (1, 0.8), (2, 0.7), (1, 0.6), (3, 0.5), (4, 0.4)]]
    assert [(i["hadm_id"], i["score"]) for i in distinct_admissions(items, 3)] == [(1, 0.9), (2, 0.7), (3, 0.5)]
    assert len(distinct_admissions(items, 10)) == 4


def test_unfiltered_search_overfetches_until_k_admissions():
    # 10 admissions x 4 notes, scores fall by admission: the first 3 notes are all admission 1.
    notes = FakeNotes(grid_notes(10, 4))
    top = distinct_search(notes, "q", 3, None)
    assert [i["hadm_id"] for i in top] == [1, 2, 3]
    assert notes.search_args == [(3, None), (12, None)]


def test_restricted_search_targets_at_most_the_given_admissions():
    notes = FakeNotes(grid_notes(10, 4))
    top = distinct_search(notes, "q", 5, [8, 9])
    assert [i["hadm_id"] for i in top] == [8, 9]
    assert notes.search_args == [(2, (8, 9)), (8, (8, 9))]
    assert distinct_search(notes, "q", 5, [99]) == []


def test_search_first_counts_candidates_in_notes_and_dedupes_after_filtering():
    sources = Sources(relational=FakeRelational(creatinine_labs(10, {2, 3})), notes=FakeNotes(grid_notes(10, 4)))
    question = QuestionSpec(qid="c", text="c", qclass="C", relational_filter=HIGH, note_query="q",
                            required_sources=frozenset({"relational", "notes"}))
    plan = Plan("sf", (RelationalFilter(HIGH, as_evidence=False), NoteSearch("q", "search_first", n_candidates=12)))
    result = execute(plan, sources, question, PlannerConfig(top_k=5))
    # 12 candidate notes = admissions 1-3; admissions 2 and 3 pass the filter, one note each.
    assert sources.notes.search_args == [(12, None)]
    assert [i["note_id"] for i in result.items] == ["2-0", "3-0"]


def test_token_estimates_use_one_item_per_admission():
    config = PlannerConfig(top_k=5, avg_tokens_per_chunk=100)
    probes = Probes(relational_rows=8, matching_admissions=3, total_notes=1000, matching_notes=60, scoped_notes=20)
    ff = Plan("ff", (RelationalFilter(HIGH, as_evidence=False), NoteSearch("q", "filter_first")))
    assert estimate_cost(ff, probes, config).est_tokens == 3 * 100
    scoped = Plan("notes_only", (NoteSearch("q"),))
    assert estimate_cost(scoped, probes, config).est_tokens == 1 * 100  # scoped_notes set: one admission
    unscoped = Probes(total_notes=1000)
    assert estimate_cost(scoped, unscoped, config).est_tokens == 5 * 100


def test_reference_tfidf_with_several_notes_per_admission(tmp_path):
    notes_gen.main(["--tables-dir", str(FIXTURES), "--out-dir", str(tmp_path), "--seed", "0", "--scale", "3"])
    sources = build_reference_sources(FIXTURES, tmp_path / "scale_3")
    question = QuestionSpec(qid="b", text="b", qclass="B", note_query="dialysis", required_sources=frozenset({"notes"}))
    result = execute(Plan("notes_only", (NoteSearch("dialysis"),)), sources, question, PlannerConfig(top_k=4))
    hadms = [i["hadm_id"] for i in result.items]
    assert len(hadms) == 4 and len(set(hadms)) == 4
    scores = [i["score"] for i in result.items]
    assert scores == sorted(scores, reverse=True)
