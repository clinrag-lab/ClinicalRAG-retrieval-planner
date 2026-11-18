
from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from clinrag.sources.base import NotesSource, SourceResult
from clinrag.sources.tokens import count_tokens


# This class handles the vector retrieval for the clinical notes.
# It uses BGE to create the embeddings and FAISS to search through them.
class FaissNotesSource(NotesSource):

    def __init__(
        self,
        index_dir: str | Path,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: str | None = None,
        encoder=None,
    ):
        # Load the embedding model and the FAISS index that was already built.
        self.index_dir = Path(index_dir)

        if encoder is not None:
            self.model = encoder
        else:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, device=device)
        self.index = faiss.read_index(str(self.index_dir / "notes.faiss"))

        # Load the text chunks and their information.
        # FAISS only stores the vectors, so we need this to know
        # which admission and note each result came from.
        with (self.index_dir / "chunks.json").open(encoding="utf-8") as f:
            self.chunks = json.load(f)

        # Make sure every vector has a matching chunk.
        if self.index.ntotal != len(self.chunks):
            raise ValueError("Index and chunk metadata have different lengths")

        # Count how many chunks belong to each hospital admission.
        self.counts = Counter(int(c["hadm_id"]) for c in self.chunks)

        # Store the positions of chunks for each admission.
        # This will help when the planner wants to search only certain admissions.
        self.by_hadm: dict[int, list[int]] = {}

        for i, chunk in enumerate(self.chunks):
            self.by_hadm.setdefault(int(chunk["hadm_id"]), []).append(i)

    @staticmethod
    def _split(
        text: str,
        chunk_words: int,
        overlap_words: int,
    ) -> list[str]:
        """
        Splits a clinical note into smaller chunks.
        The overlap helps keep information from getting cut off
        between two different chunks.
        """

        if chunk_words <= 0 or not 0 <= overlap_words < chunk_words:
            raise ValueError(
                "Require chunk_words > 0 and 0 <= overlap_words < chunk_words"
            )

        words = text.split()

        if not words:
            return []

        # Figure out how far to move before starting the next chunk.
        stride = chunk_words - overlap_words

        return [
            " ".join(words[start:start + chunk_words])
            for start in range(0, len(words), stride)
            if start == 0 or start + overlap_words < len(words)
        ]

    @classmethod
    def build(
        cls,
        csv_path: str | Path,
        index_dir: str | Path,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: str | None = None,
        chunk_words: int = 180,
        overlap_words: int = 30,
        batch_size: int = 32,
        encoder=None,
    ) -> "FaissNotesSource":
        """
        Builds the vector database from the generated discharge notes.

        The notes are split into chunks, converted into embeddings,
        and saved in a FAISS index so we can load them again later.
        """

        # Read the generated MIMIC-IV demo discharge notes.
        frame = pd.read_csv(
            csv_path,
            compression="infer",
            dtype={"note_id": str},
        )

        # These columns are needed to connect the notes to admissions.
        required = {"note_id", "hadm_id", "text"}

        if not required.issubset(frame.columns):
            raise ValueError(
                f"Missing discharge columns: "
                f"{sorted(required - set(frame.columns))}"
            )

        chunks: list[dict] = []

        # Go through each discharge note and split it into smaller pieces.
        for row in frame.itertuples(index=False):

            # Skip notes that do not have an admission ID or usable text.
            if (
                pd.isna(row.hadm_id)
                or not isinstance(row.text, str)
                or not row.text.strip()
            ):
                continue

            for part, text in enumerate(
                cls._split(row.text, chunk_words, overlap_words)
            ):
                # Keep the original note and admission information.
                # The chunk ID lets us identify each individual piece.
                chunks.append({
                    "note_id": str(row.note_id),
                    "hadm_id": int(row.hadm_id),
                    "chunk_id": f"{row.note_id}:{part}",
                    "text": text,
                })

        if not chunks:
            raise ValueError("No usable note chunks found")

        # Load BGE and generate embeddings for all of the chunks.
        if encoder is not None:
            model = encoder
        else:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name, device=device)

        vectors = model.encode(
            [c["text"] for c in chunks],
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")

        # Create the FAISS index.
        # Inner product works as cosine similarity since the vectors
        # were normalized when we generated them.
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(np.ascontiguousarray(vectors))

        # Create the folder where the index will be stored.
        dest = Path(index_dir)
        dest.mkdir(parents=True, exist_ok=True)

        # Save the embeddings in FAISS.
        faiss.write_index(index, str(dest / "notes.faiss"))

        # Save the actual text and metadata separately.
        with (dest / "chunks.json").open("w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)

        # Return the source so we can start searching immediately.
        return cls(
            dest,
            model_name=model_name,
            device=device,
            encoder=model,
        )

    def _query(self, query: str) -> np.ndarray:
        """
        Converts a search question into an embedding using BGE.
        This embedding is compared against the stored note embeddings.
        """

        # Add the retrieval instruction used by the BGE model.
        instructed = (
            "Represent this sentence for searching relevant passages: "
            + query
        )

        vector = self.model.encode(
            [instructed],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        return np.ascontiguousarray(vector)

    def total_notes(self) -> int:
        """
        Returns the total number of indexed chunks.
        This is used by the planner when estimating retrieval costs.
        """
        return len(self.chunks)

    def count_notes_for(self, hadm_ids: Sequence[int]) -> int:
        """
        Counts how many chunks belong to the given admission IDs.
        The planner can use this to estimate how much data it needs
        to search after filtering.
        """

        return sum(
            self.counts.get(h, 0)
            for h in set(int(x) for x in hadm_ids)
        )

    def probe_search(self, query: str, k_probe: int) -> list[float]:
        """
        Runs a small vector search and returns only the similarity scores.
        The planner uses these scores to estimate how useful vector
        retrieval might be for a question.
        """

        if k_probe <= 0 or not self.chunks:
            return []

        scores, _ = self.index.search(
            self._query(query),
            min(k_probe, len(self.chunks)),
        )

        return [float(x) for x in scores[0]]

    def search(self,query: str, k: int, hadm_ids: Sequence[int] | None = None,) -> SourceResult:
        """
        Searches the clinical notes for the most relevant chunks.

        If admission IDs are provided, only those admissions are searched.
        Returns the matching text, admission IDs, token count,
        and the time it took to complete the search.
        """

        # Start the timer so we can measure retrieval latency.
        start = time.perf_counter()

        # If k is zero or negative, there is nothing to retrieve.
        if k <= 0:
            return SourceResult(
                ids=[],
                items=[],
                tokens=0,
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        # Generate the embedding for the search question.
        vector = self._query(query)

        if hadm_ids is None:
            # No admission filter was given, so search the entire index.
            scores, positions = self.index.search(vector, min(k, len(self.chunks)),)

            # Match the FAISS positions with their similarity scores.
            hits = [
                (int(i), float(s))
                for i, s in zip(positions[0], scores[0])
                if i >= 0
            ]

        else:
            # The planner provided admission IDs, so only look at
            # chunks that belong to those admissions.
            positions = [
                i
                for h in set(int(x) for x in hadm_ids)
                for i in self.by_hadm.get(h, [])
            ]

            # Return an empty result if none of the admissions have chunks.
            if not positions:
                return SourceResult(
                    ids=[],
                    items=[],
                    tokens=0,
                    latency_ms=(time.perf_counter() - start) * 1000,
                )

            positions.sort()

            # Get the stored vectors for the selected admissions.
            all_vectors = self.index.reconstruct_n(0, self.index.ntotal)
            candidates = all_vectors[positions]

            # Compare the question embedding with the selected chunks.
            similarities = candidates @ vector[0]

            # Sort by similarity and keep the top k results.
            best = np.argsort( -similarities, kind="stable",)[:k]

            hits = [
                (positions[j], float(similarities[j]))
                for j in best
            ]

        # Add the similarity score to each retrieved chunk.
        items = [
            {**self.chunks[i], "score": score}
            for i, score in hits
        ]

        # Return everything in the format expected by the planner.
        return SourceResult(
            ids=sorted({x["hadm_id"] for x in items}),
            items=items,
            tokens=sum(count_tokens(x["text"]) for x in items),
            latency_ms=(time.perf_counter() - start) * 1000,
        )
