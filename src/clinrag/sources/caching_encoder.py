"""Memoize query embeddings so one question's probe and search share one encode.

The planner's probe_search and the executor's search embed the same note
query; with a SentenceTransformer behind FaissNotesSource that is two model
calls per question. CachingEncoder wraps any object with an
encode(sentences, **kwargs) method and returns the cached vector when the same
text is encoded again with the same arguments.

The runner clears the cache before each question execution (clear_query_cache),
so latencies stay per question: the first encode is paid inside planning or
retrieval, whichever runs first, and nothing carries over between questions or
repeats.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

import numpy as np

from clinrag.sources.base import Sources


class CachingEncoder:
    """encode() with a bounded LRU cache keyed by (exact text, keyword arguments)."""

    def __init__(self, encoder: Any, maxsize: int = 256) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be at least 1")
        self.encoder = encoder
        self.maxsize = maxsize
        self._cache: OrderedDict[tuple[str, tuple], np.ndarray] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _kwargs_key(kwargs: dict[str, Any]) -> tuple:
        return tuple(sorted((k, repr(v)) for k, v in kwargs.items()))

    def encode(self, sentences: str | Sequence[str], **kwargs: Any) -> np.ndarray:
        single = isinstance(sentences, str)
        texts = [sentences] if single else list(sentences)
        kw = self._kwargs_key(kwargs)
        rows: list[np.ndarray | None] = []
        missing: dict[str, list[int]] = {}
        for i, text in enumerate(texts):
            key = (text, kw)
            if key in self._cache:
                self._cache.move_to_end(key)
                self.hits += 1
                rows.append(self._cache[key])
            else:
                rows.append(None)
                missing.setdefault(text, []).append(i)
        if missing:
            self.misses += len(missing)
            fresh = np.asarray(self.encoder.encode(list(missing), **kwargs))
            for text, vector in zip(missing, fresh):
                for i in missing[text]:
                    rows[i] = vector
                self._cache[(text, kw)] = vector
                self._cache.move_to_end((text, kw))
                while len(self._cache) > self.maxsize:
                    self._cache.popitem(last=False)
        # Copies, so a caller that edits its result in place cannot change the cache.
        out = np.stack([np.array(r, copy=True) for r in rows]) if rows else np.empty((0,))
        return out[0] if single else out

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)

    def __getattr__(self, name: str) -> Any:
        # Anything else (e.g. get_sentence_embedding_dimension) goes to the wrapped encoder.
        return getattr(self.encoder, name)


def clear_query_cache(sources: Sources | None) -> None:
    """Clear the notes source's query cache, if it encodes through a CachingEncoder."""
    encoder = getattr(getattr(sources, "notes", None), "model", None)
    if isinstance(encoder, CachingEncoder):
        encoder.clear()
