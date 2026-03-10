from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from ai_dev.core import git_ops as core_git_ops
from ai_dev.core import index_ops as core_index_ops
from ai_dev.core import indexing as core_indexing
from ai_dev.core import index_state as core_index_state
from ai_dev.core import model_ops as core_model_ops
from ai_dev.core import remote_ops as core_remote_ops
from ai_dev.core import retrieve_ops as core_retrieve_ops
from ai_dev.core import retrieval as core_retrieval
from ai_dev.core import stack_ops as core_stack_ops
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
    return core_stack_ops.generate_litellm_config(cfg, DEFAULT_CONFIG["models"])


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
    return core_stack_ops.compose_command(Path("podman-compose.yml"))


def command_up(args: argparse.Namespace) -> int:
    return core_stack_ops.command_up(args, compose_command_fn=_compose_command, run_fn=run)


def command_down(_: argparse.Namespace) -> int:
    return core_stack_ops.command_down(_, compose_command_fn=_compose_command, run_fn=run)


def command_status(_: argparse.Namespace) -> int:
    return core_stack_ops.command_status(_, compose_command_fn=_compose_command, run_fn=run)


def command_pull_models(args: argparse.Namespace) -> int:
    return core_stack_ops.command_pull_models(
        args,
        load_config_fn=load_config,
        python_executable=sys.executable,
    )


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
    core_index_ops.install_index_git_hooks(git_dir=Path(".git"), write_file_fn=write_file)


def _index_single_file(
    file_path: Path,
    root: Path,
    top_terms_per_file: int,
    chunk_lines: int,
    git_branch: str,
) -> tuple[dict, list[dict], list[dict]]:
    return core_index_ops.index_single_file(
        file_path,
        root=root,
        top_terms_per_file=top_terms_per_file,
        chunk_lines=chunk_lines,
        git_branch=git_branch,
        tokenize_fn=tokenize,
        extract_symbols_fn=extract_symbols,
        build_chunks_fn=build_chunks,
        get_file_git_metadata_fn=get_file_git_metadata,
    )


def run_index_pass(root: Path, args: argparse.Namespace, incremental: bool) -> tuple[dict, dict]:
    return core_index_ops.run_index_pass(
        root,
        args,
        incremental,
        index_path=INDEX_PATH,
        collect_source_files_fn=collect_source_files,
        get_git_branch_name_fn=get_git_branch_name,
        load_index_state_fn=load_index_state,
        index_single_file_fn=_index_single_file,
    )


def command_index(args: argparse.Namespace) -> int:
    return core_index_ops.command_index(
        args,
        app_dir=APP_DIR,
        index_path=INDEX_PATH,
        write_file_fn=write_file,
        save_index_state_fn=save_index_state,
        run_index_pass_fn=run_index_pass,
        install_index_git_hooks_fn=install_index_git_hooks,
    )


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
