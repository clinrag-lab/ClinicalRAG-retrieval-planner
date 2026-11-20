"""FaissNotesSource wired into the runner, on the demo tables and a built index.

Runs only where faiss, sentence-transformers, the cached BGE model, the demo
tables, the workload and data/faiss_index (scripts/build_vector_index.py) are
all present; skipped otherwise, so CI stays green without data.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# Check availability without importing sentence_transformers (and torch) at collection:
# tests/test_classifier.py asserts the TF-IDF path never has them in sys.modules.
if importlib.util.find_spec("sentence_transformers") is None:
    pytest.skip("sentence-transformers not installed", allow_module_level=True)
pytest.importorskip("faiss")

from clinrag.config import PlannerConfig  # noqa: E402
from clinrag.eval import runner  # noqa: E402
from clinrag.planner.execute import distinct_search  # noqa: E402
from clinrag.planner.fallback import run_with_fallback  # noqa: E402
from clinrag.sources.faiss_notes import FaissNotesSource  # noqa: E402
from clinrag.workload.schema import read_jsonl  # noqa: E402

INDEX = runner.FAISS_INDEX
TABLES = runner.DEMO_TABLES
WORKLOAD = Path("data/workload")

pytestmark = pytest.mark.skipif(
    not (INDEX / "notes.faiss").exists()
    or not (INDEX / "chunks.json").exists()
    or not TABLES.exists()
    or not (WORKLOAD / "test.jsonl").exists(),
    reason="needs data/faiss_index (scripts/build_vector_index.py), the demo tables and the workload",
)


@pytest.fixture(scope="module")
def sources():
    from sentence_transformers import SentenceTransformer

    try:
        encoder = SentenceTransformer(runner.BGE_MODEL, local_files_only=True)
    except Exception as exc:  # model not cached locally
        pytest.skip(f"BGE model not available offline: {exc}")
    return runner.build_sources("faiss", encoder=encoder)


def test_build_sources_uses_faiss_notes_and_one_encoder(sources):
    assert isinstance(sources.notes, FaissNotesSource)
    assert sources.relational is not None and sources.graph is not None
    # The chunk metadata registered with DuckDB matches the index, one row per chunk.
    n = sources.relational.con.execute("SELECT count(*) FROM note_meta").fetchone()[0]
    assert n == sources.notes.total_notes() == sources.notes.index.ntotal


def test_index_has_several_chunks_for_some_admissions(sources):
    assert max(sources.notes.counts.values()) > 1


def test_distinct_search_returns_k_distinct_admissions(sources):
    notes = sources.notes
    items = distinct_search(notes, "heart failure", 5, None)
    hadms = [int(i["hadm_id"]) for i in items]
    assert len(hadms) == 5 == len(set(hadms))
    scores = [i["score"] for i in items]
    assert scores == sorted(scores, reverse=True)

    # Restricted to admissions: still at most one chunk per admission.
    some = sorted(notes.by_hadm)[:3]
    restricted = distinct_search(notes, "heart failure", 5, some)
    got = [int(i["hadm_id"]) for i in restricted]
    assert len(got) == len(set(got)) == 3 and set(got) <= set(some)


@pytest.mark.parametrize("router_name", ["oracle", "p1", "query_all"])
def test_runner_note_evidence_is_distinct_admissions(sources, router_name):
    config = PlannerConfig()
    questions = read_jsonl(WORKLOAD / "test.jsonl")[:12]
    router = runner.build_router(router_name, sources, config, WORKLOAD, 0.5)
    for question in questions:
        run = run_with_fallback(router.decide(question.spec), sources, question.spec, config)
        note_hadms = [int(i["hadm_id"]) for i in run.items if i.get("source") == "notes"]
        assert len(note_hadms) == len(set(note_hadms))
        assert len(note_hadms) <= config.top_k
