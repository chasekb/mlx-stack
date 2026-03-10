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

from ai_dev.core import git_ops as core_git_ops
from ai_dev.core import indexing as core_indexing
from ai_dev.core import index_state as core_index_state
from ai_dev.core import model_ops as core_model_ops
from ai_dev.core import remote_ops as core_remote_ops
from ai_dev.core import retrieve_ops as core_retrieve_ops
from ai_dev.core import retrieval as core_retrieval
from ai_dev.command_groups import register_all_commands
from ai_dev.templates import (
    AGENT_HTTP_API,
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
    write_file(Path("agent/http_api.py"), AGENT_HTTP_API)
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
    return core_git_ops.get_git_changed_files(root)


def get_git_branch_name(root: Path) -> str:
    return core_git_ops.get_git_branch_name(root)


def get_file_git_metadata(root: Path, rel_path: str, branch_name: str) -> dict:
    return core_git_ops.get_file_git_metadata(root, rel_path, branch_name)


def load_index_state(expected_root: Path) -> dict:
    return core_index_state.load_index_state(path=INDEX_STATE_PATH, expected_root=expected_root)


def save_index_state(root: Path, file_meta: dict[str, dict]) -> None:
    core_index_state.save_index_state(path=INDEX_STATE_PATH, root=root, file_meta=file_meta)


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
    return core_retrieve_ops.command_retrieve(
        args,
        index_path=INDEX_PATH,
        tokenize_fn=tokenize,
        get_git_branch_name_fn=get_git_branch_name,
        get_git_changed_files_fn=get_git_changed_files,
        score_symbol_match_fn=_score_symbol_match,
        score_chunk_match_fn=_score_chunk_match,
        safe_int_fn=_safe_int,
    )


def command_memory_explain(args: argparse.Namespace) -> int:
    return core_retrieve_ops.command_memory_explain(
        args,
        index_path=INDEX_PATH,
        tokenize_fn=tokenize,
        get_git_branch_name_fn=get_git_branch_name,
        get_git_changed_files_fn=get_git_changed_files,
        score_symbol_match_fn=_score_symbol_match,
        score_chunk_match_fn=_score_chunk_match,
        safe_int_fn=_safe_int,
    )


def command_configure_cursor(args: argparse.Namespace) -> int:
    return core_model_ops.command_configure_cursor(
        args,
        load_config_fn=load_config,
        write_file_fn=write_file,
        app_dir=APP_DIR,
        task_tag_aliases=TASK_TAG_ALIASES,
    )


def resolve_model_for_tag(models: list[dict], tag: str) -> str:
    return core_model_ops.resolve_model_for_tag(models, tag, TASK_TAG_ALIASES)


def command_route_model(args: argparse.Namespace) -> int:
    return core_model_ops.command_route_model(args, load_config_fn=load_config, task_tag_aliases=TASK_TAG_ALIASES)


def command_models(args: argparse.Namespace) -> int:
    return core_model_ops.command_models(args, load_config_fn=load_config)


def _tokenize_for_spec(text: str) -> list[str]:
    return core_remote_ops.tokenize_for_spec(text)


def command_spec_decode(args: argparse.Namespace) -> int:
    return core_remote_ops.command_spec_decode(args)


def _http_json(method: str, url: str, payload: dict | None = None, timeout: float = 10.0) -> dict:
    return core_remote_ops.http_json(method=method, url=url, payload=payload, timeout=timeout)


def command_embed_enqueue(args: argparse.Namespace) -> int:
    return core_remote_ops.command_embed_enqueue(args)


def command_embed_stats(args: argparse.Namespace) -> int:
    return core_remote_ops.command_embed_stats(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ai-dev", description="Local AI dev stack orchestration CLI")
    register_all_commands(
        parser,
        handlers={
            "command_init": command_init,
            "command_up": command_up,
            "command_down": command_down,
            "command_status": command_status,
            "command_pull_models": command_pull_models,
            "configure_index_mode_args": _configure_index_mode_args,
            "command_index": command_index,
            "command_retrieve": command_retrieve,
            "command_configure_cursor": command_configure_cursor,
            "command_models": command_models,
            "command_route_model": command_route_model,
            "command_spec_decode": command_spec_decode,
            "command_embed_enqueue": command_embed_enqueue,
            "command_embed_stats": command_embed_stats,
            "command_memory_explain": command_memory_explain,
        },
        task_tag_aliases=TASK_TAG_ALIASES,
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
