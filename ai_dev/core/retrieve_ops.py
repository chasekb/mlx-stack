from __future__ import annotations

import json
import sys
import time
from pathlib import Path


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
    query_terms = set(tokenize_fn(args.query))
    if not query_terms:
        print("Query is empty after tokenization.", file=sys.stderr)
        return 2

    root = Path(index_obj.get("root", "."))
    current_branch = get_git_branch_name_fn(root)
    now_ts = time.time()
    changed_files = get_git_changed_files_fn(root) if not args.no_changed_bias else set()
    path_prefix = args.path_prefix or ""

    symbol_results = []
    for s in index_obj.get("symbols", []):
        scored = score_symbol_match_fn(
            symbol=s,
            query_terms=query_terms,
            path_prefix=path_prefix,
            changed_files=changed_files,
            current_branch=current_branch,
            include_changed_bias=not args.no_changed_bias,
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
            include_changed_bias=not args.no_changed_bias,
            now_ts=now_ts,
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

    result = {
        "query": args.query,
        "current_branch": current_branch,
        "top_symbols": symbol_results[: args.top_k],
        "top_chunks": chunk_results[: args.top_k],
    }

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
    query_terms = set(tokenize_fn(args.query))
    if not query_terms:
        print("Query is empty after tokenization.", file=sys.stderr)
        return 2

    root = Path(index_obj.get("root", "."))
    current_branch = get_git_branch_name_fn(root)
    now_ts = time.time()
    changed_files = get_git_changed_files_fn(root) if not args.no_changed_bias else set()
    path_prefix = args.path_prefix or ""

    symbol_results = []
    for s in index_obj.get("symbols", []):
        scored = score_symbol_match_fn(
            symbol=s,
            query_terms=query_terms,
            path_prefix=path_prefix,
            changed_files=changed_files,
            current_branch=current_branch,
            include_changed_bias=not args.no_changed_bias,
            now_ts=now_ts,
        )
        if scored:
            symbol_results.append(
                {
                    **scored,
                    "path": s.get("path", ""),
                    "line": s.get("line"),
                    "kind": s.get("kind", "symbol"),
                    "name": s.get("name", ""),
                    "git_branch": s.get("git_branch", ""),
                    "git_commit_sha": s.get("git_commit_sha", ""),
                    "git_commit_ts": safe_int_fn(s.get("git_commit_ts", 0), 0),
                }
            )

    chunk_results = []
    for c in index_obj.get("chunks", []):
        scored = score_chunk_match_fn(
            chunk=c,
            query_terms=query_terms,
            path_prefix=path_prefix,
            changed_files=changed_files,
            current_branch=current_branch,
            include_changed_bias=not args.no_changed_bias,
            now_ts=now_ts,
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

    payload = {
        "query": args.query,
        "current_branch": current_branch,
        "path_prefix": path_prefix,
        "changed_file_bias_enabled": not args.no_changed_bias,
        "changed_files_count": len(changed_files),
        "weights": {
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
            },
            "recency_raw": "1.5 * (2 / (2 + age_days))",
        },
        "top_symbols": symbol_results[: args.top_k],
        "top_chunks": chunk_results[: args.top_k],
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"Query: {args.query}")
    print(f"Current branch: {current_branch}")
    print(f"Changed file bias: {'enabled' if not args.no_changed_bias else 'disabled'}")
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
            f"recency={br.get('recency', 0):.2f}]"
        )
        print(f"  {preview}")
    return 0
