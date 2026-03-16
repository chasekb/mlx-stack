from __future__ import annotations

from ai_dev.core.cli_facade import APP_DIR
from ai_dev.core.cli_facade import CONFIG_PATH
from ai_dev.core.cli_facade import DEFAULT_CONFIG
from ai_dev.core.cli_facade import INDEX_PATH
from ai_dev.core.cli_facade import INDEX_STATE_PATH
from ai_dev.core.cli_facade import TASK_TAG_ALIASES
from ai_dev.core.cli_facade import _recency_boost_from_commit_ts
from ai_dev.core.cli_facade import _safe_int
from ai_dev.core.cli_facade import _score_chunk_match
from ai_dev.core.cli_facade import _score_symbol_match
from ai_dev.core.cli_facade import build_chunks
from ai_dev.core.cli_facade import build_parser
from ai_dev.core.cli_facade import collect_source_files
from ai_dev.core.cli_facade import command_configure_cursor
from ai_dev.core.cli_facade import command_down
from ai_dev.core.cli_facade import command_embed_enqueue
from ai_dev.core.cli_facade import command_embed_stats
from ai_dev.core.cli_facade import command_index
from ai_dev.core.cli_facade import command_init
from ai_dev.core.cli_facade import command_memory_explain
from ai_dev.core.cli_facade import command_models
from ai_dev.core.cli_facade import command_pull_models
from ai_dev.core.cli_facade import command_retrieve
from ai_dev.core.cli_facade import command_route_model
from ai_dev.core.cli_facade import command_spec_decode
from ai_dev.core.cli_facade import command_status
from ai_dev.core.cli_facade import command_up
from ai_dev.core.cli_facade import ensure_config_schema
from ai_dev.core.cli_facade import extract_symbols
from ai_dev.core.cli_facade import generate_litellm_config
from ai_dev.core.cli_facade import get_file_git_metadata
from ai_dev.core.cli_facade import get_git_branch_name
from ai_dev.core.cli_facade import get_git_changed_files
from ai_dev.core.cli_facade import install_index_git_hooks
from ai_dev.core.cli_facade import iter_source_files
from ai_dev.core.cli_facade import load_config
from ai_dev.core.cli_facade import load_index_state
from ai_dev.core.cli_facade import resolve_model_for_tag
from ai_dev.core.cli_facade import run
from ai_dev.core.cli_facade import run_index_pass
from ai_dev.core.cli_facade import save_index_state
from ai_dev.core.cli_facade import tokenize
from ai_dev.core.cli_facade import write_file


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
    "main",
]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
