"""Reranker service scaffolding.

Provides interfaces and helpers to plug a cross-encoder reranker (e.g., BGE).

Implementation notes:
- Default is disabled. Enable via settings or explicit flag on the hybrid search.
- Dependencies (sentence-transformers/transformers/torch) are optional; import lazily.
"""

