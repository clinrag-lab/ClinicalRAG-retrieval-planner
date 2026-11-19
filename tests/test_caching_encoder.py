"""CachingEncoder: one embedding per question for the probe and the search."""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pytest

from clinrag.sources.caching_encoder import CachingEncoder, clear_query_cache
from tests.test_classifier import workload  # noqa: F401  (module-scoped fixture workload)

FIXTURES = Path(__file__).parent / "fixtures"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class CountingEncoder:
    """Deterministic hashed bag-of-words vectors; counts the texts it is asked to encode."""

    dim = 32

    def __init__(self) -> None:
        self.calls = 0
        self.texts: list[str] = []

    def encode(self, texts, **kwargs):
        texts = [texts] if isinstance(texts, str) else list(texts)
        self.calls += 1
        self.texts += texts
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            vectors[i, 0] = 1.0  # never all zero
            for word in text.lower().split():
                vectors[i, 1 + zlib.crc32(word.encode()) % (self.dim - 1)] += 1.0
        if kwargs.get("normalize_embeddings"):
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors

    def query_encodes(self) -> int:
        return sum(t.startswith(QUERY_PREFIX) for t in self.texts)


# ---- the wrapper on its own -----------------------------------------------------------------


def test_same_text_and_arguments_encode_once():
    inner = CountingEncoder()
    enc = CachingEncoder(inner)
    a = enc.encode(["heart failure"], normalize_embeddings=True)
    b = enc.encode(["heart failure"], normalize_embeddings=True)
    assert inner.calls == 1 and (enc.hits, enc.misses) == (1, 1)
    np.testing.assert_array_equal(a, b)
    enc.encode(["heart failure"], normalize_embeddings=False)  # different arguments: a new entry
    assert inner.calls == 2


def test_list_input_encodes_only_missing_texts_and_keeps_order():
    inner = CountingEncoder()
    enc = CachingEncoder(inner)
    enc.encode(["b"])
    out = enc.encode(["a", "b", "a"])
    assert inner.texts == ["b", "a"]
    np.testing.assert_array_equal(out, inner.encode(["a", "b", "a"]))
    assert enc.encode("a").shape == (CountingEncoder.dim,)


def test_lru_bound_and_clear():
    inner = CountingEncoder()
    enc = CachingEncoder(inner, maxsize=2)
    for text in ("a", "b", "a", "c"):  # "a" was used last before "c", so "b" is evicted
        enc.encode([text])
    assert len(enc) == 2
    enc.encode(["a"])
    assert inner.calls == 3
    enc.encode(["b"])
    assert inner.calls == 4
    enc.clear()
    assert len(enc) == 0
    enc.encode(["a"])
    assert inner.calls == 5
    with pytest.raises(ValueError):
        CachingEncoder(inner, maxsize=0)


def test_results_are_copies():
    enc = CachingEncoder(CountingEncoder())
    first = enc.encode(["x"])
    first[:] = 0
    assert enc.encode(["x"]).any()


# ---- with FaissNotesSource and the runner ---------------------------------------------------


@pytest.fixture(scope="module")
def index_dir(tmp_path_factory):
    faiss_notes = pytest.importorskip("clinrag.sources.faiss_notes")
    out = tmp_path_factory.mktemp("faiss_index")
    faiss_notes.FaissNotesSource.build(FIXTURES / "discharge.csv", out, encoder=CountingEncoder(),
                                       chunk_words=40, overlap_words=10)
    return out


def test_probe_and_search_share_one_encode(index_dir):
    from clinrag.sources.faiss_notes import FaissNotesSource

    inner = CountingEncoder()
    notes = FaissNotesSource(index_dir, encoder=CachingEncoder(inner))
    notes.probe_search("chest pain", 5)
    notes.search("chest pain", 3)
    notes.search("chest pain", 3, hadm_ids=sorted(notes.by_hadm)[:2])
    assert inner.query_encodes() == 1


def test_runner_encodes_each_question_once_per_execution(index_dir, workload):  # noqa: F811
    from clinrag.config import PlannerConfig
    from clinrag.eval import runner

    inner = CountingEncoder()
    sources = runner.build_faiss_sources(FIXTURES, index_dir, encoder=inner)
    assert isinstance(sources.notes.model, CachingEncoder)
    config = PlannerConfig()
    router = runner.build_router("oracle", sources, config, Path("."), 0.5)
    # A mixed question (relational filter + notes): only those run probe_search during planning.
    question = next(q for q in workload.test if {"relational", "notes"} <= q.spec.required_sources)

    record = runner.run_question("oracle", router, question, sources, config, repeats=2)

    assert "notes" in record["sources_queried"]
    # oracle probes and then searches; the cache is cleared before each repeat,
    # so each of the 2 executions encodes the query exactly once.
    assert inner.query_encodes() == 2
    clear_query_cache(sources)
    assert len(sources.notes.model) == 0

    # Without the cache the same run encodes at least twice per execution (probe, then each search call).
    raw = CountingEncoder()
    uncached = runner.build_faiss_sources(FIXTURES, index_dir, encoder=raw, cache_queries=False)
    router = runner.build_router("oracle", uncached, config, Path("."), 0.5)
    runner.run_question("oracle", router, question, uncached, config, repeats=2)
    assert raw.query_encodes() >= 4


def test_build_without_cache_keeps_raw_encoder(index_dir):
    from clinrag.eval import runner

    inner = CountingEncoder()
    sources = runner.build_faiss_sources(FIXTURES, index_dir, encoder=inner, cache_queries=False)
    assert sources.notes.model is inner
    clear_query_cache(sources)  # no-op without a cache
