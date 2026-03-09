from __future__ import annotations

import re
from typing import Optional


def tokenize(text: str) -> list[str]:
    return [tok for tok in re.split(r"[^a-zA-Z0-9_]+", text.lower()) if len(tok) >= 2]


def retrieve(index_obj: dict, query: str, top_k: int = 5, path_prefix: Optional[str] = None) -> dict:
    query_terms = set(tokenize(query))
    if not query_terms:
        return {"query": query, "top_symbols": [], "top_chunks": []}

    path_prefix = path_prefix or ""

    symbol_results = []
    for s in index_obj.get("symbols", []):
        score = 0.0
        name_terms = set(tokenize(s.get("name", "")))
        score += len(query_terms.intersection(name_terms)) * 3
        score += 1 if any(t in s.get("name", "").lower() for t in query_terms) else 0
        p = s.get("path", "")
        if path_prefix and p.startswith(path_prefix):
            score += 1.5
        if score > 0:
            symbol_results.append({"score": score, **s})

    chunk_results = []
    for c in index_obj.get("chunks", []):
        score = 0.0
        chunk_terms = set(c.get("terms", []))
        score += len(query_terms.intersection(chunk_terms))
        p = c.get("path", "")
        if path_prefix and p.startswith(path_prefix):
            score += 2.0
        if score > 0:
            chunk_results.append(
                {
                    "score": score,
                    "path": p,
                    "chunk_id": c.get("chunk_id"),
                    "start_line": c.get("start_line"),
                    "end_line": c.get("end_line"),
                    "text_preview": c.get("text_preview", ""),
                }
            )

    symbol_results.sort(key=lambda x: x["score"], reverse=True)
    chunk_results.sort(key=lambda x: x["score"], reverse=True)

    return {
        "query": query,
        "top_symbols": symbol_results[:top_k],
        "top_chunks": chunk_results[:top_k],
    }
