from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Optional, Any, Dict
from sentence_transformers import CrossEncoder


@dataclass
class RerankConfig:
    model_name: str = "BAAI/bge-reranker-base"
    device: str = "cpu"
    batch_size: int = 32
    max_length: int = 384


class BgeReranker:
    """Scaffold for a BGE cross-encoder reranker.

    Usage (after installing sentence-transformers):
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(config.model_name, device=config.device, max_length=config.max_length)
        scores = model.predict([(query, doc) for doc in docs], batch_size=config.batch_size)
    """

    def __init__(self, config: Optional[RerankConfig] = None) -> None:
        self.config = config or RerankConfig()
        self._model = None  # lazy

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder  
        except Exception as e:  
            raise RuntimeError(
                "sentence-transformers not installed. Install it to enable reranking (e.g., pip install sentence-transformers)."
            ) from e
        self._model = CrossEncoder(self.config.model_name, device=self.config.device, max_length=self.config.max_length)

    def score(self, query: str, docs: Sequence[str]) -> List[float]:
        """Return per-doc relevance scores for (query, doc) pairs.

        Higher means more relevant. Uses CrossEncoder predict when available.
        """
        if not docs:
            return []
        self._ensure_model()
        assert self._model is not None
        pairs = [(query, d) for d in docs]
        scores = self._model.predict(pairs, batch_size=self.config.batch_size)
        return list(map(float, scores))

    def rerank(self, query: str, candidates: Sequence[Dict[str, Any]], *, text_key: str = "text") -> List[Dict[str, Any]]:
        """Return candidates sorted by reranker score (desc), with 'rerank_score' attached.

        candidates: list of dicts that include a textual field under `text_key`.
        """
        if not candidates:
            return []
        docs = [str(c.get(text_key, "")) for c in candidates]
        scores = self.score(query, docs)
        out: List[Dict[str, Any]] = []
        for c, s in zip(candidates, scores):
            cc = dict(c)
            cc["rerank_score"] = float(s)
            out.append(cc)
        out.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return out

