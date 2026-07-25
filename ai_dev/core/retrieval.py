from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path


def tokenize(text: str) -> list[str]:
    return [tok for tok in re.split(r"[^a-zA-Z0-9_]+", text.lower()) if len(tok) >= 2]


def safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def deterministic_embed(text: str, dims: int = 16) -> list[float]:
    digest = hashlib.sha256((text or "").encode("utf-8")).digest()
    vals = [((digest[i % len(digest)] / 255.0) * 2.0 - 1.0) for i in range(max(1, dims))]
    norm = math.sqrt(sum(v * v for v in vals)) or 1.0
    return [v / norm for v in vals]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    n = min(len(left), len(right))
    dot = sum(left[i] * right[i] for i in range(n))
    left_norm = math.sqrt(sum(left[i] * left[i] for i in range(n))) or 1.0
    right_norm = math.sqrt(sum(right[i] * right[i] for i in range(n))) or 1.0
    return max(0.0, dot / (left_norm * right_norm))


def _coerce_vector(vec: object) -> list[float]:
    if not isinstance(vec, list):
        return []
    out: list[float] = []
    for value in vec:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            return []
    return out


def load_semantic_scores(query: str, embeddings_path: Path, path_prefix: str = "") -> dict[str, dict]:
    """Load lightweight semantic path scores from .ai-dev/embeddings.jsonl.

    The stack can run with real embedding vectors or deterministic fallback
    vectors. This helper keeps retrieval production-safe in both cases: it uses
    stored vectors when present, surfaces the vector backend in score breakdowns,
    and returns an empty map when semantic data is unavailable so lexical/symbol
    retrieval remains the fallback path.
    """
    if not embeddings_path.exists() or not embeddings_path.is_file():
        return {}

    scores: dict[str, dict] = {}
    query_vector = deterministic_embed(query)
    for line in embeddings_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        metadata = rec.get("metadata", {}) if isinstance(rec.get("metadata", {}), dict) else {}
        rel_path = str(metadata.get("path") or rec.get("path") or "").strip()
        if not rel_path or (path_prefix and not rel_path.startswith(path_prefix)):
            continue
        vector = _coerce_vector(rec.get("vector"))
        if not vector:
            continue
        query_vec = query_vector if len(query_vector) == len(vector) else deterministic_embed(query, dims=len(vector))
        raw_score = cosine_similarity(query_vec, vector)
        weighted = round(raw_score * 2.5, 4)
        previous = scores.get(rel_path, {})
        if weighted > float(previous.get("score", -1.0)):
            scores[rel_path] = {
                "score": weighted,
                "raw": round(raw_score, 4),
                "backend": str(rec.get("vector_backend", "unknown")),
                "embedding_model": str(rec.get("embedding_model", "")),
                "vector_dim": len(vector),
            }
    return scores


def recency_boost_from_commit_ts(commit_ts: int, now_ts: float) -> float:
    if commit_ts <= 0:
        return 0.0
    age_days = max(0.0, (now_ts - float(commit_ts)) / 86_400.0)
    return max(0.0, round(1.5 * (2.0 / (2.0 + age_days)), 4))


def score_symbol_match(
    symbol: dict,
    query_terms: set[str],
    path_prefix: str,
    changed_files: set[str],
    current_branch: str,
    include_changed_bias: bool,
    now_ts: float,
) -> dict | None:
    p = str(symbol.get("path", ""))
    name = str(symbol.get("name", ""))

    lexical_score = 0.0
    name_terms = set(tokenize(name))
    lexical_score += len(query_terms.intersection(name_terms)) * 3.0
    lexical_score += 1.0 if any(t in name.lower() for t in query_terms) else 0.0

    path_score = 1.5 if path_prefix and p.startswith(path_prefix) else 0.0
    changed_score = 1.0 if include_changed_bias and p in changed_files else 0.0

    branch = str(symbol.get("git_branch", "") or "")
    branch_score = 0.8 if current_branch != "unknown" and branch == current_branch else 0.0

    recency_raw = recency_boost_from_commit_ts(safe_int(symbol.get("git_commit_ts", 0), 0), now_ts)
    recency_score = round(recency_raw * 0.6, 4)

    total = lexical_score + path_score + changed_score + branch_score + recency_score
    if total <= 0:
        return None

    return {
        "score": round(total, 4),
        "score_breakdown": {
            "lexical": round(lexical_score, 4),
            "path_prefix": round(path_score, 4),
            "changed_file": round(changed_score, 4),
            "branch_match": round(branch_score, 4),
            "recency": round(recency_score, 4),
            "recency_raw": round(recency_raw, 4),
        },
    }


def score_chunk_match(
    chunk: dict,
    query_terms: set[str],
    path_prefix: str,
    changed_files: set[str],
    current_branch: str,
    include_changed_bias: bool,
    now_ts: float,
    semantic_scores: dict[str, dict] | None = None,
) -> dict | None:
    p = str(chunk.get("path", ""))
    chunk_terms = set(chunk.get("terms", []))

    lexical_score = float(len(query_terms.intersection(chunk_terms)))
    path_score = 2.0 if path_prefix and p.startswith(path_prefix) else 0.0
    changed_score = 1.5 if include_changed_bias and p in changed_files else 0.0

    branch = str(chunk.get("git_branch", "") or "")
    branch_score = 0.9 if current_branch != "unknown" and branch == current_branch else 0.0

    recency_raw = recency_boost_from_commit_ts(safe_int(chunk.get("git_commit_ts", 0), 0), now_ts)
    recency_score = round(recency_raw * 0.8, 4)

    semantic_scores = semantic_scores or {}
    semantic_info = semantic_scores.get(p, {}) if isinstance(semantic_scores.get(p, {}), dict) else {}
    semantic_score = round(float(semantic_info.get("score", 0.0) or 0.0), 4)

    total = lexical_score + path_score + changed_score + branch_score + recency_score + semantic_score
    if total <= 0:
        return None

    return {
        "score": round(total, 4),
        "score_breakdown": {
            "lexical": round(lexical_score, 4),
            "path_prefix": round(path_score, 4),
            "changed_file": round(changed_score, 4),
            "branch_match": round(branch_score, 4),
            "recency": round(recency_score, 4),
            "recency_raw": round(recency_raw, 4),
            "semantic": semantic_score,
            "semantic_raw": round(float(semantic_info.get("raw", 0.0) or 0.0), 4),
            "semantic_backend": semantic_info.get("backend", "missing"),
            "semantic_vector_dim": int(semantic_info.get("vector_dim", 0) or 0),
        },
    }
