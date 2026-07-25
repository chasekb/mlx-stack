from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from ai_dev.core.retrieval import load_semantic_scores


def _rank_index(
    *,
    index_obj: dict,
    query: str,
    top_k: int,
    path_prefix: str,
    no_changed_bias: bool,
    index_path: Path,
    tokenize_fn,
    get_git_branch_name_fn,
    get_git_changed_files_fn,
    score_symbol_match_fn,
    score_chunk_match_fn,
    safe_int_fn,
) -> dict:
    query_terms = set(tokenize_fn(query))
    if not query_terms:
        return {"error": "empty_query"}

    root = Path(index_obj.get("root", "."))
    if not root.exists():
        root = Path(".").resolve()
    current_branch = get_git_branch_name_fn(root)
    now_ts = time.time()
    changed_files = get_git_changed_files_fn(root) if not no_changed_bias else set()
    semantic_scores = load_semantic_scores(query, index_path.parent / "embeddings.jsonl", path_prefix=path_prefix)

    symbol_results = []
    for s in index_obj.get("symbols", []):
        scored = score_symbol_match_fn(
            symbol=s,
            query_terms=query_terms,
            path_prefix=path_prefix,
            changed_files=changed_files,
            current_branch=current_branch,
            include_changed_bias=not no_changed_bias,
            now_ts=now_ts,
        )
        if scored:
            symbol_results.append({**scored, **s})

    chunk_results = []
    for c in index_obj.get("chunks", []):
        scored = score_chunk_match_fn(
            chunk=c,
            query_terms=query_terms,
            path_prefix=path_prefix,
            changed_files=changed_files,
            current_branch=current_branch,
            include_changed_bias=not no_changed_bias,
            now_ts=now_ts,
            semantic_scores=semantic_scores,
        )
        if scored:
            chunk_results.append(
                {
                    **scored,
                    "path": c.get("path", ""),
                    "chunk_id": c.get("chunk_id"),
                    "start_line": c.get("start_line"),
                    "end_line": c.get("end_line"),
                    "text_preview": c.get("text_preview", ""),
                    "git_branch": c.get("git_branch", ""),
                    "git_commit_sha": c.get("git_commit_sha", ""),
                    "git_commit_ts": safe_int_fn(c.get("git_commit_ts", 0), 0),
                }
            )

    symbol_results.sort(key=lambda x: x["score"], reverse=True)
    chunk_results.sort(key=lambda x: x["score"], reverse=True)

    return {
        "query": query,
        "current_branch": current_branch,
        "path_prefix": path_prefix,
        "changed_file_bias_enabled": not no_changed_bias,
        "changed_files_count": len(changed_files),
        "semantic": {
            "source": str(index_path.parent / "embeddings.jsonl"),
            "matches": len(semantic_scores),
            "fallback": "lexical_symbol_only" if not semantic_scores else "hybrid_jsonl",
        },
        "top_symbols": symbol_results[:top_k],
        "top_chunks": chunk_results[:top_k],
    }


def command_retrieve(
    args,
    *,
    index_path: Path,
    tokenize_fn,
    get_git_branch_name_fn,
    get_git_changed_files_fn,
    score_symbol_match_fn,
    score_chunk_match_fn,
    safe_int_fn,
) -> int:
    if not index_path.exists():
        print("Missing .ai-dev/index.json. Run `ai-dev index .` first.", file=sys.stderr)
        return 2

    index_obj = json.loads(index_path.read_text(encoding="utf-8"))
    path_prefix = args.path_prefix or ""
    result = _rank_index(
        index_obj=index_obj,
        query=args.query,
        top_k=args.top_k,
        path_prefix=path_prefix,
        no_changed_bias=args.no_changed_bias,
        index_path=index_path,
        tokenize_fn=tokenize_fn,
        get_git_branch_name_fn=get_git_branch_name_fn,
        get_git_changed_files_fn=get_git_changed_files_fn,
        score_symbol_match_fn=score_symbol_match_fn,
        score_chunk_match_fn=score_chunk_match_fn,
        safe_int_fn=safe_int_fn,
    )
    if result.get("error") == "empty_query":
        print("Query is empty after tokenization.", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Query: {args.query}\n")
        print("Top symbols:")
        for s in result["top_symbols"]:
            print(f"- {s['path']}:{s.get('line', '?')} {s.get('kind', 'symbol')} {s.get('name', '')} (score={s['score']:.2f})")
        print("\nTop chunks:")
        for c in result["top_chunks"]:
            print(f"- {c['path']}:{c['start_line']}-{c['end_line']} (score={c['score']:.2f})")
            preview = c.get("text_preview", "").replace("\n", " ")[:140]
            print(f"  {preview}")
    return 0


def command_memory_explain(
    args,
    *,
    index_path: Path,
    tokenize_fn,
    get_git_branch_name_fn,
    get_git_changed_files_fn,
    score_symbol_match_fn,
    score_chunk_match_fn,
    safe_int_fn,
) -> int:
    if not index_path.exists():
        print("Missing .ai-dev/index.json. Run `ai-dev index .` first.", file=sys.stderr)
        return 2

    index_obj = json.loads(index_path.read_text(encoding="utf-8"))
    path_prefix = args.path_prefix or ""
    payload = _rank_index(
        index_obj=index_obj,
        query=args.query,
        top_k=args.top_k,
        path_prefix=path_prefix,
        no_changed_bias=args.no_changed_bias,
        index_path=index_path,
        tokenize_fn=tokenize_fn,
        get_git_branch_name_fn=get_git_branch_name_fn,
        get_git_changed_files_fn=get_git_changed_files_fn,
        score_symbol_match_fn=score_symbol_match_fn,
        score_chunk_match_fn=score_chunk_match_fn,
        safe_int_fn=safe_int_fn,
    )
    if payload.get("error") == "empty_query":
        print("Query is empty after tokenization.", file=sys.stderr)
        return 2

    payload["weights"] = {
        "symbol": {
            "lexical_match": "+3.0 each name token intersection +1.0 substring",
            "path_prefix": "+1.5",
            "changed_file": "+1.0",
            "branch_match": "+0.8",
            "recency": "recency_raw * 0.6",
        },
        "chunk": {
            "lexical_match": "+1.0 each chunk term intersection",
            "path_prefix": "+2.0",
            "changed_file": "+1.5",
            "branch_match": "+0.9",
            "recency": "recency_raw * 0.8",
            "semantic": "cosine(query_embedding, stored_vector) * 2.5 when .ai-dev/embeddings.jsonl has matching paths",
        },
        "recency_raw": "1.5 * (2 / (2 + age_days))",
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"Query: {args.query}")
    print(f"Current branch: {payload['current_branch']}")
    print(f"Changed file bias: {'enabled' if not args.no_changed_bias else 'disabled'}")
    print(f"Semantic vector matches: {payload['semantic']['matches']} ({payload['semantic']['fallback']})")
    print("\nTop symbols (with scoring breakdown):")
    for s in payload["top_symbols"]:
        br = s.get("score_breakdown", {})
        print(
            f"- {s['path']}:{s.get('line', '?')} {s.get('kind', 'symbol')} {s.get('name', '')} "
            f"score={s['score']:.2f} "
            f"[lex={br.get('lexical', 0):.2f}, prefix={br.get('path_prefix', 0):.2f}, "
            f"changed={br.get('changed_file', 0):.2f}, branch={br.get('branch_match', 0):.2f}, "
            f"recency={br.get('recency', 0):.2f}]"
        )

    print("\nTop chunks (with scoring breakdown):")
    for c in payload["top_chunks"]:
        br = c.get("score_breakdown", {})
        preview = c.get("text_preview", "").replace("\n", " ")[:140]
        print(
            f"- {c['path']}:{c.get('start_line', '?')}-{c.get('end_line', '?')} "
            f"score={c['score']:.2f} "
            f"[lex={br.get('lexical', 0):.2f}, prefix={br.get('path_prefix', 0):.2f}, "
            f"changed={br.get('changed_file', 0):.2f}, branch={br.get('branch_match', 0):.2f}, "
            f"recency={br.get('recency', 0):.2f}, semantic={br.get('semantic', 0):.2f}]"
        )
        print(f"  {preview}")
    return 0
