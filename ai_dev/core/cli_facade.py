from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

from ai_dev.core import config_ops as core_config_ops
from ai_dev.core import git_ops as core_git_ops
from ai_dev.core import handler_ops as core_handler_ops
from ai_dev.core import init_ops as core_init_ops
from ai_dev.core import index_ops as core_index_ops
from ai_dev.core import indexing as core_indexing
from ai_dev.core import index_state as core_index_state
from ai_dev.core import model_ops as core_model_ops
from ai_dev.core import paths as core_paths
from ai_dev.core import remote_ops as core_remote_ops
from ai_dev.core import retrieve_ops as core_retrieve_ops
from ai_dev.core import retrieval as core_retrieval
from ai_dev.core import runtime_ops as core_runtime_ops
from ai_dev.core import stack_ops as core_stack_ops

APP_DIR = core_paths.APP_DIR
CONFIG_PATH = core_paths.CONFIG_PATH
INDEX_PATH = core_paths.INDEX_PATH
INDEX_STATE_PATH = core_paths.INDEX_STATE_PATH

DEFAULT_CONFIG = core_config_ops.DEFAULT_CONFIG
TASK_TAG_ALIASES = core_config_ops.TASK_TAG_ALIASES


def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> int:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return core_runtime_ops.run_command(cmd, cwd=cwd, env=merged_env)


def write_file(path: Path, content: str, executable: bool = False) -> None:
    core_runtime_ops.write_file(path, content, executable=executable)


def load_config() -> dict:
    return core_config_ops.load_config(
        config_path=CONFIG_PATH,
        default_config=DEFAULT_CONFIG,
        ensure_config_schema_fn=ensure_config_schema,
    )


def ensure_config_schema(cfg: dict) -> dict:
    return core_config_ops.ensure_config_schema(cfg, default_config=DEFAULT_CONFIG)


def generate_litellm_config(cfg: dict) -> str:
    return core_stack_ops.generate_litellm_config(cfg, DEFAULT_CONFIG["models"])


def command_init(args: argparse.Namespace) -> int:
    return core_init_ops.command_init(
        args,
        app_dir=APP_DIR,
        config_path=CONFIG_PATH,
        load_config_fn=load_config,
        write_file_fn=write_file,
        generate_litellm_config_fn=generate_litellm_config,
    )


def _compose_command() -> list[str]:
    return core_stack_ops.compose_command(Path("podman-compose.yml"))


def command_up(args: argparse.Namespace) -> int:
    return core_stack_ops.command_up(
        args,
        compose_command_fn=_compose_command,
        run_fn=run,
        load_config_fn=load_config,
        app_dir=APP_DIR,
        project_root=Path.cwd(),
        python_executable=sys.executable,
    )


def command_down(args: argparse.Namespace) -> int:
    return core_stack_ops.command_down(
        args,
        compose_command_fn=_compose_command,
        run_fn=run,
        app_dir=APP_DIR,
        project_root=Path.cwd(),
        compose_file=Path("podman-compose.yml"),
    )


def command_status(args: argparse.Namespace) -> int:
    return core_stack_ops.command_status(
        args,
        compose_command_fn=_compose_command,
        run_fn=run,
        load_config_fn=load_config,
        app_dir=APP_DIR,
    )


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
    semantic_scores: dict[str, dict] | None = None,
) -> dict | None:
    return core_retrieval.score_chunk_match(
        chunk=chunk,
        query_terms=query_terms,
        path_prefix=path_prefix,
        changed_files=changed_files,
        current_branch=current_branch,
        include_changed_bias=include_changed_bias,
        now_ts=now_ts,
        semantic_scores=semantic_scores,
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


def command_spec_decode(args: argparse.Namespace) -> int:
    return core_remote_ops.command_spec_decode(args)


def command_embed_enqueue(args: argparse.Namespace) -> int:
    return core_remote_ops.command_embed_enqueue(args)


def command_embed_stats(args: argparse.Namespace) -> int:
    return core_remote_ops.command_embed_stats(args)


def build_parser() -> argparse.ArgumentParser:
    return core_handler_ops.build_cli_parser(
        task_tag_aliases=TASK_TAG_ALIASES,
        command_init=command_init,
        command_up=command_up,
        command_down=command_down,
        command_status=command_status,
        command_pull_models=command_pull_models,
        command_index=command_index,
        command_retrieve=command_retrieve,
        command_configure_cursor=command_configure_cursor,
        command_models=command_models,
        command_route_model=command_route_model,
        command_spec_decode=command_spec_decode,
        command_embed_enqueue=command_embed_enqueue,
        command_embed_stats=command_embed_stats,
        command_memory_explain=command_memory_explain,
    )


__all__ = [
    "APP_DIR",
    "CONFIG_PATH",
    "INDEX_PATH",
    "INDEX_STATE_PATH",
    "DEFAULT_CONFIG",
    "TASK_TAG_ALIASES",
    "run",
    "write_file",
    "load_config",
    "ensure_config_schema",
    "generate_litellm_config",
    "command_init",
    "command_up",
    "command_down",
    "command_status",
    "command_pull_models",
    "iter_source_files",
    "collect_source_files",
    "tokenize",
    "extract_symbols",
    "build_chunks",
    "get_git_changed_files",
    "get_git_branch_name",
    "get_file_git_metadata",
    "load_index_state",
    "save_index_state",
    "install_index_git_hooks",
    "run_index_pass",
    "command_index",
    "_safe_int",
    "_recency_boost_from_commit_ts",
    "_score_symbol_match",
    "_score_chunk_match",
    "command_retrieve",
    "command_memory_explain",
    "command_configure_cursor",
    "resolve_model_for_tag",
    "command_route_model",
    "command_models",
    "command_spec_decode",
    "command_embed_enqueue",
    "command_embed_stats",
    "build_parser",
]