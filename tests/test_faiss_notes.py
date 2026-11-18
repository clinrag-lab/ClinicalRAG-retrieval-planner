
from pathlib import Path

import numpy as np
import pytest

from clinrag.sources.faiss_notes import FaissNotesSource


# Use a fake encoder so tests don't download the BGE model.
class FakeEncoder:

    def encode(self, texts, **kwargs):
        keywords = [
            "diabetes",
            "pneumonia",
            "heart",
            "kidney",
            "blood",
        ]

        vectors = []

        for text in texts:
            text = text.lower()

            vector = [
                float(text.count(word))
                for word in keywords
            ]

            vectors.append(vector)

        vectors = np.array(vectors, dtype=np.float32)

        # Normalize vectors like the real embedding model.
        if kwargs.get("normalize_embeddings", False):
            norms = np.linalg.norm(
                vectors, axis=1, keepdims=True
            )
            vectors = vectors / np.maximum(norms, 1e-12)

        return vectors


# Build a small FAISS index before each test.
@pytest.fixture
def notes_source(tmp_path):

    fixture_path = (
        Path(__file__).parent
        / "fixtures"
        / "discharge.csv"
    )

    source = FaissNotesSource.build(
        csv_path=fixture_path,
        index_dir=tmp_path / "faiss_index",
        encoder=FakeEncoder(),
        chunk_words=100,
        overlap_words=10,
    )

    return source


# Test 1: The index builds successfully.
def test_build_index(notes_source):

    assert notes_source.index.ntotal > 0

    assert (
        notes_source.index.ntotal
        == len(notes_source.chunks)
    )


# Test 2: Searching returns clinical notes.
def test_search(notes_source):

    result = notes_source.search(
        query="diabetes",
        k=5,
    )

    assert len(result.items) > 0

    assert len(result.items) <= 5

    assert all(
        "hadm_id" in item
        for item in result.items
    )


# Test 3: Filtering only returns allowed admissions.
def test_filtered_search(notes_source):

    admission_id = notes_source.chunks[0]["hadm_id"]

    result = notes_source.search(
        query="diabetes",
        k=5,
        hadm_ids=[admission_id],
    )

    assert len(result.items) > 0

    assert all(
        item["hadm_id"] == admission_id
        for item in result.items
    )


# Test 4: Invalid admission IDs return no results.
def test_invalid_admission(notes_source):

    result = notes_source.search(
        query="diabetes",
        k=5,
        hadm_ids=[-999999],
    )

    assert result.items == []

    assert result.ids == []

    assert result.tokens == 0


# Test 5: Filtering all admissions matches full search.
def test_filtered_vs_unfiltered(notes_source):

    all_admissions = list({
        chunk["hadm_id"]
        for chunk in notes_source.chunks
    })

    normal = notes_source.search(
        query="diabetes",
        k=5,
    )

    filtered = notes_source.search(
        query="diabetes",
        k=5,
        hadm_ids=all_admissions,
    )

    # Compare the returned chunk IDs.
    normal_ids = {
        item["chunk_id"]
        for item in normal.items
    }

    filtered_ids = {
        item["chunk_id"]
        for item in filtered.items
    }

    assert normal_ids == filtered_ids
