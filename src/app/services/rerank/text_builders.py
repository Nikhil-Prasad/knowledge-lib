from __future__ import annotations

from typing import Dict, Any


def build_rerank_text(candidate: Dict[str, Any]) -> str:
    """Heuristic to compose a reranker input text from result fields.

    Format: "{title} · {section}\n{snippet_or_text}" with safe fallbacks.
    """
    title = candidate.get("title") or ""
    section = candidate.get("section_path") or ""
    snippet = candidate.get("snippet") or candidate.get("text") or ""
    head = " · ".join([s for s in [str(title).strip(), str(section).strip()] if s])
    if head:
        return f"{head}\n{snippet}"
    return str(snippet)

