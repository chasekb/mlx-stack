from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from ai_dev.core.retrieval import load_semantic_scores


def tokenize(text: str) -> list[str]:
    return [tok for tok in re.split(r"[^a-zA-Z0-9_]+", text.lower()) if len(tok) >= 2]


def retrieve(index_obj: dict, query: str, top_k: int = 5, path_prefix: Optional[str] = None) -> dict:
    query_terms = set(tokenize(query))
    if not query_terms:
        return {"query": query, "semantic": {"matches": 0, "fallback": "empty_query"}, "top_symbols": [], "top_chunks": []}

    path_prefix = path_prefix or ""
    root = Path(__file__).resolve().parents[1]
    semantic_scores = load_semantic_scores(query, root / ".ai-dev" / "embeddings.jsonl", path_prefix=path_prefix)

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
        semantic_info = semantic_scores.get(p, {}) if isinstance(semantic_scores.get(p, {}), dict) else {}
        semantic_score = float(semantic_info.get("score", 0.0) or 0.0)
        score += semantic_score
        if score > 0:
            chunk_results.append(
                {
                    "score": round(score, 4),
                    "score_breakdown": {
                        "semantic": round(semantic_score, 4),
                        "semantic_backend": semantic_info.get("backend", "missing"),
                    },
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
        "semantic": {
            "matches": len(semantic_scores),
            "fallback": "lexical_symbol_only" if not semantic_scores else "hybrid_jsonl",
        },
        "top_symbols": symbol_results[:top_k],
        "top_chunks": chunk_results[:top_k],
    }
