from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from ai_dev.core import indexing as core_indexing
from ai_dev.core import retrieval as core_retrieval
from ai_dev.templates import (
    AGENT_SERVER,
    EMBED_QUEUE_SERVER,
    EMBED_WORKER,
    LITELLM_CONFIG,
    MLX_DOCKERFILE,
    MLX_ENTRYPOINT,
    PODMAN_COMPOSE_YAML,
    RAG_SERVER,
    SPEC_ROUTER_SERVER,
)

APP_DIR = Path(".ai-dev")
CONFIG_PATH = APP_DIR / "config.json"
INDEX_PATH = APP_DIR / "index.json"
INDEX_STATE_PATH = APP_DIR / "index_state.json"




















DEFAULT_CONFIG = {
    "created_at": "",
    "stack": {
        "mlx_port": 8081,
        "litellm_port": 4000,
        "spec_router_port": 8092,
        "embed_queue_port": 8093,
        "default_model": "mlx-community/Qwen3.5-Coder-7B-Instruct-4bit",
    },
    "models": [
        {
            "name": "local-mlx-fast",
            "backend_model": "openai/local-mlx-fast",
            "api_base": "http://mlx:8081/v1",
            "api_key": "local-dev",
            "hf_model": "Qwen/Qwen3.5-Coder-1.5B-Instruct",
            "mlx_model": "mlx-community/Qwen3.5-Coder-1.5B-Instruct-4bit",
            "quantization": "4bit",
            "tags": ["fast", "default"],
        },
        {
            "name": "local-mlx",
            "backend_model": "openai/local-mlx",
            "api_base": "http://mlx:8081/v1",
            "api_key": "local-dev",
            "hf_model": "Qwen/Qwen3.5-Coder-3B-Instruct",
            "mlx_model": "mlx-community/Qwen3.5-Coder-3B-Instruct-4bit",
            "quantization": "4bit",
            "tags": ["quality", "default"],
        },
        {
            "name": "local-mlx-longctx",
            "backend_model": "openai/local-mlx-longctx",
            "api_base": "http://mlx:8081/v1",
            "api_key": "local-dev",
            "hf_model": "Qwen/Qwen3.5-Coder-7B-Instruct",
            "mlx_model": "mlx-community/Qwen3.5-Coder-7B-Instruct-4bit",
            "quantization": "4bit",
            "tags": ["longctx", "analysis"],
        },
    ],
    "routing": {
        "fast": "local-mlx-fast",
        "quality": "local-mlx",
        "longctx": "local-mlx-longctx",
        "analysis": "local-mlx-longctx",
        "default": "local-mlx",
    },
    "cursor": {
        "base_url": "http://localhost:4000/v1",
        "api_key": "local-dev",
        "model": "local-mlx",
    },
}

TASK_TAG_ALIASES = {
    "default": ["default", "quality"],
    "quality": ["quality", "default"],
    "fast": ["fast", "default"],
    "longctx": ["longctx", "analysis", "default"],
    "analysis": ["analysis", "longctx", "quality", "default"],
}


def run(cmd: list[str], cwd: Path | None = None) -> int:
    proc = subprocess.run(cmd, cwd=cwd)
    return proc.returncode


def write_file(path: Path, content: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if executable:
        current_mode = path.stat().st_mode
        path.chmod(current_mode | 0o111)


def load_config() -> dict:
    if CONFIG_PATH.exists():
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return ensure_config_schema(cfg)
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["created_at"] = datetime.now(timezone.utc).isoformat()
    return cfg


def ensure_config_schema(cfg: dict) -> dict:
    if "models" not in cfg or not isinstance(cfg["models"], list) or not cfg["models"]:
        cfg["models"] = copy.deepcopy(DEFAULT_CONFIG["models"])

    if "cursor" not in cfg or not isinstance(cfg["cursor"], dict):
        cfg["cursor"] = copy.deepcopy(DEFAULT_CONFIG["cursor"])

    if not cfg["cursor"].get("model"):
        cfg["cursor"]["model"] = cfg["models"][0]["name"]

    if not cfg["cursor"].get("base_url"):
        cfg["cursor"]["base_url"] = DEFAULT_CONFIG["cursor"]["base_url"]

    if not cfg["cursor"].get("api_key"):
        cfg["cursor"]["api_key"] = DEFAULT_CONFIG["cursor"]["api_key"]

    if "stack" not in cfg or not isinstance(cfg["stack"], dict):
        cfg["stack"] = copy.deepcopy(DEFAULT_CONFIG["stack"])
    else:
        for k, v in DEFAULT_CONFIG["stack"].items():
            cfg["stack"].setdefault(k, v)

    if "routing" not in cfg or not isinstance(cfg["routing"], dict):
        cfg["routing"] = copy.deepcopy(DEFAULT_CONFIG["routing"])
    else:
        for k, v in DEFAULT_CONFIG["routing"].items():
            cfg["routing"].setdefault(k, v)

    for m in cfg.get("models", []):
        if not m.get("output_path"):
            m["output_path"] = f"models/{m.get('name', 'local-mlx')}"

    return cfg


def generate_litellm_config(cfg: dict) -> str:
    models = cfg.get("models") or DEFAULT_CONFIG["models"]
    lines = ["model_list:"]
    for m in models:
        name = m.get("name", "local-mlx")
        backend_model = m.get("backend_model", "openai/local-mlx")
        api_base = m.get("api_base", "http://mlx:8081/v1")
        api_key = m.get("api_key", "local-dev")
        lines.extend(
            [
                f"  - model_name: {name}",
                "    litellm_params:",
                f"      model: {backend_model}",
                f"      api_base: {api_base}",
                f"      api_key: {api_key}",
            ]
        )

    master_key = cfg.get("cursor", {}).get("api_key", "local-dev")
    lines.extend(["", "general_settings:", f"  master_key: {master_key}"])
    return "\n".join(lines) + "\n"


def command_init(_: argparse.Namespace) -> int:
    APP_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config()
    config["created_at"] = config.get("created_at") or datetime.now(timezone.utc).isoformat()

    write_file(Path("podman-compose.yml"), PODMAN_COMPOSE_YAML)
    write_file(Path("litellm_config.yaml"), generate_litellm_config(config))
    write_file(Path("mlx/entrypoint.sh"), MLX_ENTRYPOINT, executable=True)
    write_file(Path("mlx/Dockerfile"), MLX_DOCKERFILE)
    write_file(Path("rag/server.py"), RAG_SERVER)
    write_file(Path("agent/server.py"), AGENT_SERVER)
    write_file(Path("spec_router/server.py"), SPEC_ROUTER_SERVER)
    write_file(Path("embedding_queue/server.py"), EMBED_QUEUE_SERVER)
    write_file(Path("embedding_worker/worker.py"), EMBED_WORKER)

    write_file(CONFIG_PATH, json.dumps(config, indent=2) + "\n")

    print("Initialized local AI dev stack files.")
    return 0


def _compose_command() -> list[str]:
    compose_file = Path("podman-compose.yml")
    if not compose_file.exists():
        print("Missing podman-compose.yml. Run `ai-dev init` first.", file=sys.stderr)
        raise SystemExit(2)
    return ["podman", "compose", "-f", str(compose_file)]


def command_up(args: argparse.Namespace) -> int:
    cmd = _compose_command() + ["up", "-d"]
    if args.with_optional:
        cmd.extend(["--profile", "optional"])
    return run(cmd)


def command_down(_: argparse.Namespace) -> int:
    cmd = _compose_command() + ["down"]
    return run(cmd)


def command_status(_: argparse.Namespace) -> int:
    cmd = _compose_command() + ["ps"]
    return run(cmd)


def command_pull_models(args: argparse.Namespace) -> int:
    cfg = load_config()
    if args.profile:
        profiles = [m for m in cfg.get("models", []) if m.get("name") == args.profile]
    else:
        profiles = cfg.get("models", [])

    if not profiles:
        print("No matching model profiles found.", file=sys.stderr)
        return 2

    commands: list[tuple[str, list[str]]] = []
    for m in profiles:
        name = m.get("name", "local-mlx")
        hf_model = m.get("hf_model") or args.model
        q = m.get("quantization", f"{args.quantization}bit").replace("bit", "")
        output_path = m.get("output_path", f"models/{name}")
        Path(output_path).mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "-m",
            "mlx_lm.convert",
            "--hf-path",
            hf_model,
            "--quantize",
            q,
            "--output-path",
            output_path,
        ]
        commands.append((name, cmd))

    if args.dry_run:
        print("Dry run (commands to execute):\n")
        for name, cmd in commands:
            print(f"# Profile: {name}")
            print(" ".join(cmd))
            print("")
        return 0

    rc = 0
    for name, cmd in commands:
        print(f"[pull-models] Converting profile: {name}")
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            rc = proc.returncode
            print(
                f"[pull-models] Failed for profile '{name}'. "
                "If mlx-lm is not installed in this Python env, install it first.",
                file=sys.stderr,
            )
            if not args.continue_on_error:
                return rc

    if rc == 0:
        print("[pull-models] Completed all model conversions.")
    else:
        print("[pull-models] Completed with errors.", file=sys.stderr)

    return rc


def iter_source_files(root: Path, max_bytes: int) -> Iterable[Path]:
    return core_indexing.iter_source_files(root, max_bytes=max_bytes)


def collect_source_files(root: Path, max_bytes: int) -> dict[str, Path]:
    return core_indexing.collect_source_files(root, max_bytes=max_bytes)


def tokenize(text: str) -> list[str]:
    return core_retrieval.tokenize(text)


def extract_symbols(file_path: Path, content: str) -> list[dict]:
    return core_indexing.extract_symbols(file_path, content)


def build_chunks(content: str, lines_per_chunk: int = 80) -> list[dict]:
    return core_indexing.build_chunks(content, lines_per_chunk=lines_per_chunk)


def get_git_changed_files(root: Path) -> set[str]:
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return set()
    changed = set()
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        # format: XY path
        path = line[3:].strip()
        if path:
            changed.add(path)
    return changed


def get_git_branch_name(root: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return "unknown"
    return (proc.stdout or "").strip() or "unknown"


def get_file_git_metadata(root: Path, rel_path: str, branch_name: str) -> dict:
    proc = subprocess.run(
        ["git", "log", "-1", "--format=%H|%ct", "--", rel_path],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return {
            "git_branch": branch_name,
            "git_commit_sha": "",
            "git_commit_ts": 0,
        }

    out = (proc.stdout or "").strip()
    if "|" not in out:
        return {
            "git_branch": branch_name,
            "git_commit_sha": "",
            "git_commit_ts": 0,
        }

    sha, ts = out.split("|", 1)
    try:
        ts_int = int(ts)
    except ValueError:
        ts_int = 0

    return {
        "git_branch": branch_name,
        "git_commit_sha": sha,
        "git_commit_ts": ts_int,
    }


def load_index_state(expected_root: Path) -> dict:
    if not INDEX_STATE_PATH.exists():
        return {}
    try:
        state = json.loads(INDEX_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if str(expected_root) != str(state.get("root", "")):
        return {}
    return state


def save_index_state(root: Path, file_meta: dict[str, dict]) -> None:
    payload = {
        "schema_version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "files": file_meta,
    }
    write_file(INDEX_STATE_PATH, json.dumps(payload, indent=2) + "\n")


def install_index_git_hooks() -> None:
    hooks_dir = Path(".git") / "hooks"
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

        write_file(hook_path, content, executable=True)


def _index_single_file(
    file_path: Path,
    root: Path,
    top_terms_per_file: int,
    chunk_lines: int,
    git_branch: str,
) -> tuple[dict, list[dict], list[dict]]:
    rel = str(file_path.relative_to(root))
    content = file_path.read_text(encoding="utf-8", errors="ignore")
    tok_counter = Counter(tokenize(content))
    symbols = extract_symbols(file_path, content)
    chunks = build_chunks(content, lines_per_chunk=chunk_lines)
    git_meta = get_file_git_metadata(root=root, rel_path=rel, branch_name=git_branch)

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


def run_index_pass(root: Path, args: argparse.Namespace, incremental: bool) -> tuple[dict, dict]:
    git_branch = get_git_branch_name(root)
    current_files = collect_source_files(root, max_bytes=args.max_file_size)
    current_meta = {
        rel: {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
        for rel, p in current_files.items()
    }

    prev_index = {}
    if INDEX_PATH.exists():
        try:
            prev_index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        except Exception:
            prev_index = {}
    if str(root) != str(prev_index.get("root", "")):
        prev_index = {}

    prev_state = load_index_state(root)
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
            file_entry, symbols, chunks = _index_single_file(
                path,
                root,
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


def command_index(args: argparse.Namespace) -> int:
    root = Path(args.path).resolve()
    if not root.exists() or not root.is_dir():
        print(f"Path not found or not a directory: {root}", file=sys.stderr)
        return 2

    APP_DIR.mkdir(parents=True, exist_ok=True)

    if args.install_git_hooks:
        install_index_git_hooks()
        print("Installed git hooks: post-checkout, post-merge")

    def execute_once(incremental: bool) -> int:
        index_obj, stats = run_index_pass(root=root, args=args, incremental=incremental)
        if stats.get("skipped_write"):
            print("No source changes detected. Index is already up to date.")
            return 0

        write_file(INDEX_PATH, json.dumps(index_obj, indent=2) + "\n")
        file_meta = {
            f["path"]: {
                "size": int(f.get("size", 0)),
                "mtime_ns": int((root / f["path"]).stat().st_mtime_ns) if (root / f["path"]).exists() else 0,
            }
            for f in index_obj.get("files", [])
        }
        save_index_state(root=root, file_meta=file_meta)

        print(
            f"Indexed {index_obj.get('file_count', 0)} files -> {INDEX_PATH} "
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


def _configure_index_mode_args(p_index: argparse.ArgumentParser) -> None:
    mode_group = p_index.add_mutually_exclusive_group()
    mode_group.add_argument("--once", action="store_true", help="Run one incremental indexing pass")
    mode_group.add_argument("--daemon", action="store_true", help="Continuously run incremental indexing")
    return 0


def _safe_int(value: object, default: int = 0) -> int:
    return core_retrieval.safe_int(value, default=default)


def _recency_boost_from_commit_ts(commit_ts: int, now_ts: float) -> float:
    return core_retrieval.recency_boost_from_commit_ts(commit_ts, now_ts)


def _score_symbol_match(
    symbol: dict,
    query_terms: set[str],
    path_prefix: str,
    changed_files: set[str],
    current_branch: str,
    include_changed_bias: bool,
    now_ts: float,
) -> dict | None:
    return core_retrieval.score_symbol_match(
        symbol=symbol,
        query_terms=query_terms,
        path_prefix=path_prefix,
        changed_files=changed_files,
        current_branch=current_branch,
        include_changed_bias=include_changed_bias,
        now_ts=now_ts,
    )


def _score_chunk_match(
    chunk: dict,
    query_terms: set[str],
    path_prefix: str,
    changed_files: set[str],
    current_branch: str,
    include_changed_bias: bool,
    now_ts: float,
) -> dict | None:
    return core_retrieval.score_chunk_match(
        chunk=chunk,
        query_terms=query_terms,
        path_prefix=path_prefix,
        changed_files=changed_files,
        current_branch=current_branch,
        include_changed_bias=include_changed_bias,
        now_ts=now_ts,
    )


def command_retrieve(args: argparse.Namespace) -> int:
    if not INDEX_PATH.exists():
        print("Missing .ai-dev/index.json. Run `ai-dev index .` first.", file=sys.stderr)
        return 2

    index_obj = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    query_terms = set(tokenize(args.query))
    if not query_terms:
        print("Query is empty after tokenization.", file=sys.stderr)
        return 2

    root = Path(index_obj.get("root", "."))
    current_branch = get_git_branch_name(root)
    now_ts = time.time()
    changed_files = get_git_changed_files(root) if not args.no_changed_bias else set()
    path_prefix = args.path_prefix or ""

    symbol_results = []
    for s in index_obj.get("symbols", []):
        scored = _score_symbol_match(
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
        scored = _score_chunk_match(
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
                    "git_commit_ts": _safe_int(c.get("git_commit_ts", 0), 0),
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


def command_memory_explain(args: argparse.Namespace) -> int:
    if not INDEX_PATH.exists():
        print("Missing .ai-dev/index.json. Run `ai-dev index .` first.", file=sys.stderr)
        return 2

    index_obj = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    query_terms = set(tokenize(args.query))
    if not query_terms:
        print("Query is empty after tokenization.", file=sys.stderr)
        return 2

    root = Path(index_obj.get("root", "."))
    current_branch = get_git_branch_name(root)
    now_ts = time.time()
    changed_files = get_git_changed_files(root) if not args.no_changed_bias else set()
    path_prefix = args.path_prefix or ""

    symbol_results = []
    for s in index_obj.get("symbols", []):
        scored = _score_symbol_match(
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
                    "git_commit_ts": _safe_int(s.get("git_commit_ts", 0), 0),
                }
            )

    chunk_results = []
    for c in index_obj.get("chunks", []):
        scored = _score_chunk_match(
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
                    "git_commit_ts": _safe_int(c.get("git_commit_ts", 0), 0),
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


def command_configure_cursor(args: argparse.Namespace) -> int:
    cfg = load_config()

    selected_model = args.model
    if not selected_model and args.task_tag:
        selected_model = resolve_model_for_tag(cfg.get("models", []), args.task_tag)

    if not selected_model:
        selected_model = cfg["cursor"]["model"]

    cursor_cfg = {
        "name": "Local LiteLLM",
        "provider": "openai",
        "baseUrl": args.base_url or cfg["cursor"]["base_url"],
        "apiKey": args.api_key or cfg["cursor"]["api_key"],
        "model": selected_model,
    }

    APP_DIR.mkdir(parents=True, exist_ok=True)
    output_path = APP_DIR / "cursor-openai.json"
    write_file(output_path, json.dumps(cursor_cfg, indent=2) + "\n")

    print("Use the following OpenAI-compatible model config in Cursor:")
    print(json.dumps(cursor_cfg, indent=2))
    print(f"\nSaved: {output_path}")
    return 0


def resolve_model_for_tag(models: list[dict], tag: str) -> str:
    normalized = (tag or "").strip().lower()
    preferred_tags = TASK_TAG_ALIASES.get(normalized, [normalized, "default"])

    for wanted in preferred_tags:
        for m in models:
            tags = [str(t).lower() for t in m.get("tags", [])]
            if wanted in tags:
                return m.get("name", "local-mlx")

    if models:
        return models[0].get("name", "local-mlx")
    return "local-mlx"


def command_route_model(args: argparse.Namespace) -> int:
    cfg = load_config()
    models = cfg.get("models", [])
    chosen = resolve_model_for_tag(models, args.task_tag)
    if args.json:
        print(json.dumps({"task_tag": args.task_tag, "model": chosen}, indent=2))
    else:
        print(chosen)
    return 0


def command_models(args: argparse.Namespace) -> int:
    cfg = load_config()
    models = cfg.get("models", [])

    if args.json:
        print(json.dumps(models, indent=2))
        return 0

    if not models:
        print("No models configured in .ai-dev/config.json")
        return 0

    print("Configured model profiles:\n")
    for m in models:
        tags = ", ".join(m.get("tags", []))
        print(f"- {m.get('name', 'unnamed')}")
        print(f"  backend: {m.get('backend_model', '')}")
        print(f"  api_base: {m.get('api_base', '')}")
        if tags:
            print(f"  tags: {tags}")
    return 0


def _tokenize_for_spec(text: str) -> list[str]:
    normalized = (text or "").replace("\n", " ").strip()
    return [t for t in normalized.split(" ") if t]


def command_spec_decode(args: argparse.Namespace) -> int:
    payload: dict = {}

    if args.prompt:
        payload = {
            "prompt": args.prompt,
            "draft_model": args.draft_model,
            "target_model": args.target_model,
            "draft_url": args.draft_url,
            "target_url": args.target_url,
            "max_tokens": args.max_tokens,
            "timeout": args.timeout,
        }
    else:
        draft_tokens: list[str]
        target_tokens: list[str]

        if args.draft_tokens:
            draft_tokens = [t for t in args.draft_tokens if t]
        else:
            draft_tokens = _tokenize_for_spec(args.draft_text)

        if args.target_tokens:
            target_tokens = [t for t in args.target_tokens if t]
        else:
            target_tokens = _tokenize_for_spec(args.target_text)

        payload = {
            "draft_tokens": draft_tokens,
            "target_tokens": target_tokens,
        }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        args.url.rstrip("/") + "/spec/decode",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.URLError as e:
        print(f"spec-decode request failed: {e}", file=sys.stderr)
        return 2

    try:
        parsed = json.loads(body)
    except Exception:
        print("spec-decode returned invalid JSON", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(parsed, indent=2))
        return 0

    result = parsed.get("result", {}) if isinstance(parsed, dict) else {}
    if result.get("source") == "model_calls":
        print(f"source: {result.get('source')}")
        print(f"draft_model: {result.get('draft_model', '')}")
        print(f"target_model: {result.get('target_model', '')}")
        print(f"draft_call_ms: {result.get('draft_call_ms', 0)}")
        print(f"target_call_ms: {result.get('target_call_ms', 0)}")
        if result.get("draft_error"):
            print(f"draft_error: {result.get('draft_error')}")
    print(f"accepted_tokens: {result.get('accepted_tokens', 0)}")
    print(f"compared_tokens: {result.get('compared_tokens', 0)}")
    print(f"acceptance_rate: {result.get('acceptance_rate', 0.0)}")
    print("output_tokens:")
    for tok in result.get("output_tokens", []):
        print(f"- {tok}")
    return 0


def _http_json(method: str, url: str, payload: dict | None = None, timeout: float = 10.0) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def command_embed_enqueue(args: argparse.Namespace) -> int:
    metadata = {}
    if args.metadata_json:
        try:
            parsed = json.loads(args.metadata_json)
            metadata = parsed if isinstance(parsed, dict) else {}
        except Exception:
            print("Invalid --metadata-json payload", file=sys.stderr)
            return 2

    payload = {
        "kind": args.kind,
        "payload": {
            "path": args.path,
            "text": args.text,
            "metadata": metadata,
            "enqueued_at": datetime.now(timezone.utc).isoformat(),
        },
        "max_attempts": args.max_attempts,
    }

    try:
        out = _http_json(
            "POST",
            args.url.rstrip("/") + "/jobs/enqueue",
            payload=payload,
            timeout=args.timeout,
        )
    except urllib.error.URLError as e:
        print(f"embed-enqueue request failed: {e}", file=sys.stderr)
        return 2

    print(json.dumps(out, indent=2) if args.json else f"Enqueued job_id={out.get('job_id')} status={out.get('status')}")
    return 0


def command_embed_stats(args: argparse.Namespace) -> int:
    try:
        out = _http_json("GET", args.url.rstrip("/") + "/stats", timeout=args.timeout)
    except urllib.error.URLError as e:
        print(f"embed-stats request failed: {e}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(out, indent=2))
        return 0

    stats = out.get("stats", {}) if isinstance(out, dict) else {}
    print("Embedding queue stats:")
    print(f"- queued: {stats.get('queued', 0)}")
    print(f"- retry: {stats.get('retry', 0)}")
    print(f"- in_progress: {stats.get('in_progress', 0)}")
    print(f"- done: {stats.get('done', 0)}")
    print(f"- dead_letter: {stats.get('dead_letter', 0)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ai-dev", description="Local AI dev stack orchestration CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Generate stack files and default config")
    p_init.set_defaults(func=command_init)

    p_up = sub.add_parser("up", help="Start podman compose stack")
    p_up.add_argument("--with-optional", action="store_true", help="Enable optional profile services")
    p_up.set_defaults(func=command_up)

    p_down = sub.add_parser("down", help="Stop podman compose stack")
    p_down.set_defaults(func=command_down)

    p_status = sub.add_parser("status", help="Show service status")
    p_status.set_defaults(func=command_status)

    p_pull = sub.add_parser("pull-models", help="Pull/convert configured models into local output paths")
    p_pull.add_argument("--model", default="Qwen/Qwen3.5-Coder-7B-Instruct", help="Fallback HuggingFace model id")
    p_pull.add_argument("--quantization", default="4", help="Quantization bits for mlx_lm.convert")
    p_pull.add_argument("--profile", default=None, help="Optional model profile name from .ai-dev/config.json")
    p_pull.add_argument("--dry-run", action="store_true", help="Print conversion commands without executing")
    p_pull.add_argument("--continue-on-error", action="store_true", help="Continue converting remaining profiles on failure")
    p_pull.set_defaults(func=command_pull_models)

    p_index = sub.add_parser("index", help="Build lightweight lexical index")
    p_index.add_argument("path", nargs="?", default=".", help="Directory to index")
    p_index.add_argument("--max-file-size", type=int, default=512_000, help="Max file size in bytes")
    p_index.add_argument("--top-terms-per-file", type=int, default=20)
    p_index.add_argument("--top-terms-global", type=int, default=100)
    p_index.add_argument("--chunk-lines", type=int, default=80, help="Lines per retrieval chunk")
    _configure_index_mode_args(p_index)
    p_index.add_argument("--interval", type=float, default=2.0, help="Daemon polling interval in seconds")
    p_index.add_argument(
        "--install-git-hooks",
        action="store_true",
        help="Install post-checkout and post-merge hooks to trigger incremental indexing",
    )
    p_index.set_defaults(func=command_index)

    p_retrieve = sub.add_parser("retrieve", help="Retrieve repo-aware symbols/chunks for a query")
    p_retrieve.add_argument("query", help="Search query")
    p_retrieve.add_argument("--top-k", type=int, default=5)
    p_retrieve.add_argument("--path-prefix", default=None, help="Prefer paths with this prefix")
    p_retrieve.add_argument("--no-changed-bias", action="store_true", help="Disable bias toward changed git files")
    p_retrieve.add_argument("--json", action="store_true")
    p_retrieve.set_defaults(func=command_retrieve)

    p_cursor = sub.add_parser("configure-cursor", help="Output Cursor OpenAI-compatible config")
    p_cursor.add_argument("--base-url", default=None)
    p_cursor.add_argument("--api-key", default=None)
    p_cursor.add_argument("--model", default=None)
    p_cursor.add_argument(
        "--task-tag",
        choices=sorted(TASK_TAG_ALIASES.keys()),
        default=None,
        help="Select model by routing tag (fast, quality, longctx, analysis, default)",
    )
    p_cursor.set_defaults(func=command_configure_cursor)

    p_models = sub.add_parser("models", help="List configured model profiles")
    p_models.add_argument("--json", action="store_true", help="Print model profiles as JSON")
    p_models.set_defaults(func=command_models)

    p_route = sub.add_parser("route-model", help="Resolve model name for a task tag")
    p_route.add_argument("task_tag", choices=sorted(TASK_TAG_ALIASES.keys()))
    p_route.add_argument("--json", action="store_true")
    p_route.set_defaults(func=command_route_model)

    p_spec = sub.add_parser("spec-decode", help="Run speculative decode loop via local spec-router")
    p_spec.add_argument("--url", default="http://localhost:8092", help="Spec-router base URL")
    p_spec.add_argument("--timeout", type=float, default=10.0)
    p_spec.add_argument("--prompt", default="", help="Prompt text for model-backed speculative decode mode")
    p_spec.add_argument("--draft-model", default="local-mlx-fast", help="Draft model name for prompt mode")
    p_spec.add_argument("--target-model", default="local-mlx", help="Target model name for prompt mode")
    p_spec.add_argument(
        "--draft-url",
        default="http://localhost:4000/v1/completions",
        help="Draft model completion endpoint for prompt mode",
    )
    p_spec.add_argument(
        "--target-url",
        default="http://localhost:4000/v1/completions",
        help="Target model completion endpoint for prompt mode",
    )
    p_spec.add_argument("--max-tokens", type=int, default=128, help="Completion max tokens for prompt mode")
    p_spec.add_argument("--draft-text", default="", help="Draft model text to tokenize on spaces")
    p_spec.add_argument("--target-text", default="", help="Target model text to tokenize on spaces")
    p_spec.add_argument("--draft-tokens", nargs="*", default=None, help="Explicit draft tokens")
    p_spec.add_argument("--target-tokens", nargs="*", default=None, help="Explicit target tokens")
    p_spec.add_argument("--json", action="store_true")
    p_spec.set_defaults(func=command_spec_decode)

    p_embed_enqueue = sub.add_parser("embed-enqueue", help="Enqueue an embedding job for background worker")
    p_embed_enqueue.add_argument("--url", default="http://localhost:8093", help="Embed queue base URL")
    p_embed_enqueue.add_argument("--timeout", type=float, default=10.0)
    p_embed_enqueue.add_argument("--kind", default="file_change", help="Job kind")
    p_embed_enqueue.add_argument("--path", default="", help="File path associated with the event")
    p_embed_enqueue.add_argument("--text", default="", help="Optional text payload to embed")
    p_embed_enqueue.add_argument("--metadata-json", default="", help="Optional JSON object string")
    p_embed_enqueue.add_argument("--max-attempts", type=int, default=3)
    p_embed_enqueue.add_argument("--json", action="store_true")
    p_embed_enqueue.set_defaults(func=command_embed_enqueue)

    p_embed_stats = sub.add_parser("embed-stats", help="Show embed queue job stats")
    p_embed_stats.add_argument("--url", default="http://localhost:8093", help="Embed queue base URL")
    p_embed_stats.add_argument("--timeout", type=float, default=10.0)
    p_embed_stats.add_argument("--json", action="store_true")
    p_embed_stats.set_defaults(func=command_embed_stats)

    p_memory = sub.add_parser("memory", help="Git-aware memory utilities")
    memory_sub = p_memory.add_subparsers(dest="memory_command", required=True)

    p_memory_explain = memory_sub.add_parser("explain", help="Explain retrieval scoring for a query")
    p_memory_explain.add_argument("query", help="Search query")
    p_memory_explain.add_argument("--top-k", type=int, default=5)
    p_memory_explain.add_argument("--path-prefix", default=None, help="Prefer paths with this prefix")
    p_memory_explain.add_argument("--no-changed-bias", action="store_true", help="Disable bias toward changed git files")
    p_memory_explain.add_argument("--json", action="store_true")
    p_memory_explain.set_defaults(func=command_memory_explain)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
