""" 
Vector store for news embeddings.

Uses FAISS for fast similarity search.
Global singleton instance is created on first import.
"""

from __future__ import annotations

from typing import Any

import faiss
import numpy as np

# --------------------------------------------------------------------------- #
# 🔍  Global singleton                                                         #
# --------------------------------------------------------------------------- #
_STORE: VectorStore | None = None


def get_store() -> VectorStore:
    """Get global vector store instance (create if needed)."""
    global _STORE
    if _STORE is None:
        _STORE = VectorStore()
    return _STORE


def add_vector(vector: np.ndarray, metadata: Any) -> None:
    """Add vector to global store."""
    get_store().add(vector, metadata)


def search_vectors(query: np.ndarray, k: int = 3) -> list[tuple[Any, float]]:
    """Search global store."""
    return get_store().search(query, k)


# --------------------------------------------------------------------------- #
# 🗄️  Main class                                                              #
# --------------------------------------------------------------------------- #
class VectorStore:
    """FAISS-backed vector store with metadata."""

    def __init__(self) -> None:
        self._index = faiss.IndexFlatL2(1536)      # 1536-D embeddings
        self._metadata: list[Any] = []

    def add(self, vector: np.ndarray, metadata: Any) -> None:
        """Add vector & metadata to store."""
        self._index.add(vector.reshape(1, -1))
        self._metadata.append(metadata)

    def search(self, query: np.ndarray, k: int = 3) -> list[tuple[Any, float]]:
        """Find k nearest neighbours & distances."""
        if len(self._metadata) == 0:
            return []  # Return empty list if no vectors in store
        
        D, I = self._index.search(query.reshape(1, -1), k)
        # Filter out invalid indices (FAISS returns -1 for missing results)
        return [(self._metadata[i], d) for i, d in zip(I[0], D[0]) if i >= 0 and i < len(self._metadata)]