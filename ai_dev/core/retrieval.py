from __future__ import annotations

import re


def tokenize(text: str) -> list[str]:
    return [tok for tok in re.split(r"[^a-zA-Z0-9_]+", text.lower()) if len(tok) >= 2]


def safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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
