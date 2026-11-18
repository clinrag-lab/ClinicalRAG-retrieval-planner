"""Calibration recovers planted timings from fake sources driven by a fake clock."""

import pytest

from clinrag.config import PlannerConfig, load_planner_config
from clinrag.eval.calibrate import LineFit, calibrate, write_outputs
from clinrag.question import QuestionSpec, StructuredFilter
from clinrag.sources.base import RelationalQuery, Sources
from clinrag.sources.tokens import count_tokens
from clinrag.workload.schema import WorkloadQuestion
from tests.fakes import CREATININE, FakeGraph, FakeNotes, FakeRelational, creatinine_labs, grid_notes

# Planted costs, in ms.
RELATIONAL_MS = 7.0
COUNT_MS = 2.0
GRAPH_MS = 1.5
SEARCH_BASE_MS = 1.2  # fixed cost per search, not a PlannerConfig constant
ID_MS_PER_1K = 0.4
SCAN_MS_PER_1K = 2.5
FETCH_MS_PER_1K = 30.0


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def add_ms(self, ms):
        self.t += ms / 1000.0


class TimedRelational(FakeRelational):
    def __init__(self, clock, *args):
        super().__init__(*args)
        self.clock = clock

    def run(self, query):
        self.clock.add_ms(RELATIONAL_MS)
        return super().run(query)

    def count(self, query):
        self.clock.add_ms(COUNT_MS)
        return super().count(query)


class TimedGraph(FakeGraph):
    def __init__(self, clock, hierarchy):
        super().__init__(hierarchy)
        self.clock = clock

    def expand(self, category):
        self.clock.add_ms(GRAPH_MS)
        return super().expand(category)


class TimedNotes(FakeNotes):
    """search costs base + id_filter(ids) + scan(notes searched) + fetch(min(k, notes searched))."""

    def __init__(self, clock, notes):
        super().__init__(notes)
        self.clock = clock

    def search(self, query, k, hadm_ids=None):
        if hadm_ids is None:
            n_ids, scanned = 0, len(self.notes)
        else:
            wanted = set(hadm_ids)
            n_ids, scanned = len(hadm_ids), sum(n.hadm_id in wanted for n in self.notes)
        self.clock.add_ms(
            SEARCH_BASE_MS + ID_MS_PER_1K * n_ids / 1000 + SCAN_MS_PER_1K * scanned / 1000
            + FETCH_MS_PER_1K * min(k, scanned) / 1000
        )
        return super().search(query, k, hadm_ids)


def wq(qid, qclass, sources, **kw):
    spec = QuestionSpec(qid=qid, text=qid, qclass=qclass, required_sources=frozenset(sources), **kw)
    return WorkloadQuestion(spec, {"answer": [1], "template_id": qid})


def setup():
    clock = Clock()
    sources = Sources(
        relational=TimedRelational(clock, creatinine_labs(50, set(range(1, 20))), [(1, "N179"), (2, "N186")]),
        notes=TimedNotes(clock, grid_notes(50, 4)),
        graph=TimedGraph(clock, {"renal": ["N179", "N186"]}),
    )
    questions = [
        wq("a1", "A", {"relational"}, relational_filter=StructuredFilter(CREATININE, hadm_id=3)),
        wq("c1", "C", {"relational", "notes"}, relational_filter=StructuredFilter(CREATININE, ">", 2.0), note_query="dialysis"),
        wq("c2", "C", {"relational", "notes"}, relational_filter=StructuredFilter(CREATININE, ">", 0.5), note_query="sepsis"),
        wq("d1", "D", {"graph", "relational", "notes"}, category="renal", note_query="insulin"),
    ]
    return clock, sources, questions


def test_calibration_recovers_planted_constants():
    clock, sources, questions = setup()
    config, report = calibrate(sources, questions, repeats=3, clock=clock)

    assert config.relational_ms == pytest.approx(RELATIONAL_MS)
    assert config.graph_ms == pytest.approx(GRAPH_MS)
    assert config.id_filter_ms_per_1k_ids == pytest.approx(ID_MS_PER_1K)
    assert config.note_scan_ms_per_1k_notes == pytest.approx(SCAN_MS_PER_1K)
    assert config.note_fetch_ms_per_1k_candidates == pytest.approx(FETCH_MS_PER_1K)
    for fit in report["fits"].values():
        assert fit["r2"] == pytest.approx(1.0)
    # Unfiltered searches also scan all 200 notes: intercept = base + scan(200).
    assert report["fits"]["fetch_ms_vs_1k_candidates"]["intercept"] == pytest.approx(SEARCH_BASE_MS + SCAN_MS_PER_1K * 0.2)
    assert report["relational_count_ms"]["median"] == pytest.approx(COUNT_MS)


def test_calibration_measures_token_averages():
    clock, sources, questions = setup()
    config, _ = calibrate(sources, questions, repeats=1, clock=clock)
    assert config.avg_tokens_per_chunk == count_tokens("note text here")
    rows = [
        count_tokens(item["text"])
        for q in questions if "relational" in q.spec.required_sources
        for item in sources.relational.run(
            RelationalQuery(
                lab=q.spec.relational_filter,
                icd_codes=tuple(sources.graph.expand(q.spec.category)) if q.spec.category else (),
            )
        ).items
    ]
    assert config.avg_tokens_per_row == pytest.approx(sum(rows) / len(rows))


def test_min_similarity_fit_separates_tagged_notes():
    clock, sources, questions = setup()
    # Fake scores ignore the query and fall with admission id: admissions 1-10 score highest.
    tags = {"dialysis": set(range(1, 11))}
    config, report = calibrate(sources, questions, repeats=1, clock=clock, concept_tags=tags)
    fit = report["min_similarity_fit"]
    lowest_tagged = min(n.score for n in sources.notes.notes if n.hadm_id <= 10)
    assert fit["balanced_accuracy"] == 1.0
    assert config.min_similarity == pytest.approx(lowest_tagged)


def test_unchanged_keys_and_yaml_round_trip(tmp_path):
    clock, sources, questions = setup()
    base = PlannerConfig(top_k=7, alpha=0.5)
    config, report = calibrate(sources, questions, base=base, repeats=1, clock=clock)
    assert config.top_k == 7 and config.alpha == 0.5 and config.min_similarity == base.min_similarity
    assert "top_k" in report["unchanged"]
    write_outputs(config, report, tmp_path / "calibrated_config.yaml")
    assert load_planner_config(tmp_path / "calibrated_config.yaml") == config
    assert (tmp_path / "calibration_report.md").read_text(encoding="utf-8").startswith("# Calibration")
    assert (tmp_path / "calibration_report.json").exists()


def test_load_planner_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("top_k: 3\ndata_dir: data/\nnot_a_key: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not_a_key"):
        load_planner_config(path)
    path.write_text("top_k: 3\ndata_dir: data/\n", encoding="utf-8")
    assert load_planner_config(path).top_k == 3


def test_line_fit():
    fit = LineFit.fit([1, 2, 3], [3, 5, 7])
    assert fit.slope == pytest.approx(2.0) and fit.intercept == pytest.approx(1.0) and fit.r2 == pytest.approx(1.0)
    flat = LineFit.fit([2, 2], [1, 3])
    assert flat.slope == 0.0 and flat.intercept == 2.0
