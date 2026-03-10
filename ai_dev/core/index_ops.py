from __future__ import annotations

import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def install_index_git_hooks(*, git_dir: Path, write_file_fn) -> None:
    hooks_dir = git_dir / "hooks"
    if not hooks_dir.exists():
        print("No .git/hooks directory found. Initialize git first.", file=sys.stderr)
        raise SystemExit(2)

    marker = "# ai-dev-auto-index"
    hook_snippet = (
        f"{marker}\n"
        "if command -v python3 >/dev/null 2>&1; then\n"
        "  python3 -m ai_dev.cli index --once . >/dev/null 2>&1 || true\n"
        "fi\n"
    )

    for hook_name in ("post-checkout", "post-merge"):
        hook_path = hooks_dir / hook_name
        if hook_path.exists():
            existing = hook_path.read_text(encoding="utf-8", errors="ignore")
            if marker in existing:
                continue
            if not existing.endswith("\n"):
                existing += "\n"
            content = existing + "\n" + hook_snippet
        else:
            content = "#!/usr/bin/env bash\nset -euo pipefail\n\n" + hook_snippet

        write_file_fn(hook_path, content, executable=True)


def index_single_file(
    file_path: Path,
    *,
    root: Path,
    top_terms_per_file: int,
    chunk_lines: int,
    git_branch: str,
    tokenize_fn,
    extract_symbols_fn,
    build_chunks_fn,
    get_file_git_metadata_fn,
) -> tuple[dict, list[dict], list[dict]]:
    rel = str(file_path.relative_to(root))
    content = file_path.read_text(encoding="utf-8", errors="ignore")
    tok_counter = Counter(tokenize_fn(content))
    symbols = extract_symbols_fn(file_path, content)
    chunks = build_chunks_fn(content, lines_per_chunk=chunk_lines)
    git_meta = get_file_git_metadata_fn(root=root, rel_path=rel, branch_name=git_branch)

    file_entry = {
        "path": rel,
        "size": file_path.stat().st_size,
        "token_count": sum(tok_counter.values()),
        "symbol_count": len(symbols),
        "chunk_count": len(chunks),
        "top_terms": dict(tok_counter.most_common(top_terms_per_file)),
        **git_meta,
    }

    symbol_rows = [{"path": rel, **git_meta, **s} for s in symbols]
    chunk_rows = [{"path": rel, **git_meta, **c} for c in chunks]
    return file_entry, symbol_rows, chunk_rows


def run_index_pass(
    root: Path,
    args,
    incremental: bool,
    *,
    index_path: Path,
    collect_source_files_fn,
    get_git_branch_name_fn,
    load_index_state_fn,
    index_single_file_fn,
) -> tuple[dict, dict]:
    git_branch = get_git_branch_name_fn(root)
    current_files = collect_source_files_fn(root, max_bytes=args.max_file_size)
    current_meta = {
        rel: {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
        for rel, p in current_files.items()
    }

    prev_index = {}
    if index_path.exists():
        try:
            prev_index = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            prev_index = {}
    if str(root) != str(prev_index.get("root", "")):
        prev_index = {}

    prev_state = load_index_state_fn(root)
    prev_meta = prev_state.get("files", {}) if isinstance(prev_state.get("files", {}), dict) else {}

    changed_paths = sorted(rel for rel in current_files if prev_meta.get(rel) != current_meta.get(rel))
    removed_paths = sorted(set(prev_meta.keys()) - set(current_files.keys()))

    if incremental and prev_index and not changed_paths and not removed_paths:
        stats = {
            "mode": "incremental",
            "changed": 0,
            "removed": 0,
            "reused": len(current_files),
            "indexed": 0,
            "skipped_write": True,
        }
        return prev_index, stats

    prev_files_by_path = {f.get("path"): f for f in prev_index.get("files", []) if f.get("path")}
    prev_symbols_by_path: dict[str, list[dict]] = {}
    for s in prev_index.get("symbols", []):
        p = s.get("path")
        if p:
            prev_symbols_by_path.setdefault(p, []).append(s)
    prev_chunks_by_path: dict[str, list[dict]] = {}
    for c in prev_index.get("chunks", []):
        p = c.get("path")
        if p:
            prev_chunks_by_path.setdefault(p, []).append(c)

    file_entries: list[dict] = []
    all_symbols: list[dict] = []
    all_chunks: list[dict] = []
    vocabulary = Counter()
    total_tokens = 0
    indexed_count = 0
    reused_count = 0

    for rel in sorted(current_files.keys()):
        path = current_files[rel]
        can_reuse = (
            incremental
            and rel in prev_files_by_path
            and rel not in changed_paths
            and rel in prev_symbols_by_path
            and rel in prev_chunks_by_path
        )

        if can_reuse:
            reused_count += 1
            file_entry = prev_files_by_path[rel]
            symbols = prev_symbols_by_path[rel]
            chunks = prev_chunks_by_path[rel]
        else:
            indexed_count += 1
            file_entry, symbols, chunks = index_single_file_fn(
                path,
                root=root,
                top_terms_per_file=args.top_terms_per_file,
                chunk_lines=args.chunk_lines,
                git_branch=git_branch,
            )

        file_entries.append(file_entry)
        all_symbols.extend(symbols)
        all_chunks.extend(chunks)
        total_tokens += int(file_entry.get("token_count", 0))
        vocabulary.update(file_entry.get("top_terms", {}))

    index_obj = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "file_count": len(file_entries),
        "total_tokens": total_tokens,
        "top_terms_global": dict(vocabulary.most_common(args.top_terms_global)),
        "symbols": all_symbols,
        "chunks": all_chunks,
        "files": file_entries,
        "index_mode": "incremental" if incremental else "full",
        "git_branch": git_branch,
    }

    stats = {
        "mode": "incremental" if incremental else "full",
        "changed": len(changed_paths),
        "removed": len(removed_paths),
        "reused": reused_count,
        "indexed": indexed_count,
        "skipped_write": False,
    }
    return index_obj, stats


def command_index(
    args,
    *,
    app_dir: Path,
    index_path: Path,
    write_file_fn,
    save_index_state_fn,
    run_index_pass_fn,
    install_index_git_hooks_fn,
) -> int:
    root = Path(args.path).resolve()
    if not root.exists() or not root.is_dir():
        print(f"Path not found or not a directory: {root}", file=sys.stderr)
        return 2

    app_dir.mkdir(parents=True, exist_ok=True)

    if args.install_git_hooks:
        install_index_git_hooks_fn()
        print("Installed git hooks: post-checkout, post-merge")

    def execute_once(incremental: bool) -> int:
        index_obj, stats = run_index_pass_fn(root=root, args=args, incremental=incremental)
        if stats.get("skipped_write"):
            print("No source changes detected. Index is already up to date.")
            return 0

        write_file_fn(index_path, json.dumps(index_obj, indent=2) + "\n")
        file_meta = {
            f["path"]: {
                "size": int(f.get("size", 0)),
                "mtime_ns": int((root / f["path"]).stat().st_mtime_ns) if (root / f["path"]).exists() else 0,
            }
            for f in index_obj.get("files", [])
        }
        save_index_state_fn(root=root, file_meta=file_meta)

        print(
            f"Indexed {index_obj.get('file_count', 0)} files -> {index_path} "
            f"(mode={stats['mode']}, indexed={stats['indexed']}, reused={stats['reused']}, removed={stats['removed']})"
        )
        return 0

    if args.daemon:
        print(f"Starting index daemon (interval={args.interval}s). Press Ctrl+C to stop.")
        try:
            while True:
                execute_once(incremental=True)
                time.sleep(max(0.5, args.interval))
        except KeyboardInterrupt:
            print("Index daemon stopped.")
            return 0

    if args.once:
        return execute_once(incremental=True)

    return execute_once(incremental=False)
