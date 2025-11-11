from __future__ import annotations

from typing import Dict, List, Tuple, Any


def rrf_fuse(rankings: List[List[Dict[str, Any]]], *, id_key: str = "segment_id", k: int = 60) -> List[Tuple[Dict[str, Any], float]]:
    """Reciprocal Rank Fusion across multiple ranked lists.

    rankings: list of ranked lists (each list is list of hit dicts)
    id_key: key to identify unique items (default: segment_id)
    k: RRF constant (typically 60)

    Returns list of tuples (representative_hit_dict, fused_score) sorted by fused_score desc.
    Representative hit is taken from the first list where the item appears, falling back to any.
    """
    fused: Dict[Any, float] = {}
    rep: Dict[Any, Dict[str, Any]] = {}
    for lst in rankings:
        for rank, hit in enumerate(lst, start=1):
            key = hit.get(id_key)
            if key is None:
                continue
            score = 1.0 / (k + rank)
            fused[key] = fused.get(key, 0.0) + score
            if key not in rep:
                rep[key] = hit
    items = [
        (rep[key], score)
        for key, score in fused.items()
    ]
    items.sort(key=lambda x: x[1], reverse=True)
    return items

