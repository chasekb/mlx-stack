from __future__ import annotations

import argparse
from typing import Callable, TypedDict


CommandHandler = Callable[[argparse.Namespace], int]


class CommandHandlers(TypedDict):
    command_init: CommandHandler
    command_up: CommandHandler
    command_down: CommandHandler
    command_status: CommandHandler
    command_pull_models: CommandHandler
    command_index: CommandHandler
    command_retrieve: CommandHandler
    command_configure_cursor: CommandHandler
    command_models: CommandHandler
    command_route_model: CommandHandler
    command_spec_decode: CommandHandler
    command_embed_enqueue: CommandHandler
    command_embed_stats: CommandHandler
    command_memory_explain: CommandHandler


def register_all_commands(
    parser: argparse.ArgumentParser,
    handlers: CommandHandlers,
    task_tag_aliases: dict[str, list[str]],
) -> None:
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Generate stack files and default config")
    p_init.set_defaults(func=handlers["command_init"])

    p_up = sub.add_parser("up", help="Start podman compose stack")
    p_up.add_argument("--with-optional", action="store_true", help="Enable optional profile services")
    p_up.set_defaults(func=handlers["command_up"])

    p_down = sub.add_parser("down", help="Stop podman compose stack")
    p_down.set_defaults(func=handlers["command_down"])

    p_status = sub.add_parser("status", help="Show service status")
    p_status.set_defaults(func=handlers["command_status"])

    p_pull = sub.add_parser("pull-models", help="Pull/convert configured models into local output paths")
    p_pull.add_argument("--model", default="Qwen/Qwen3.5-Coder-7B-Instruct", help="Fallback HuggingFace model id")
    p_pull.add_argument("--quantization", default="4", help="Quantization bits for mlx_lm.convert")
    p_pull.add_argument("--profile", default=None, help="Optional model profile name from .ai-dev/config.json")
    p_pull.add_argument("--dry-run", action="store_true", help="Print conversion commands without executing")
    p_pull.add_argument("--continue-on-error", action="store_true", help="Continue converting remaining profiles on failure")
    p_pull.set_defaults(func=handlers["command_pull_models"])

    p_index = sub.add_parser("index", help="Build lightweight lexical index")
    p_index.add_argument("path", nargs="?", default=".", help="Directory to index")
    p_index.add_argument("--max-file-size", type=int, default=512_000, help="Max file size in bytes")
    p_index.add_argument("--top-terms-per-file", type=int, default=20)
    p_index.add_argument("--top-terms-global", type=int, default=100)
    p_index.add_argument("--chunk-lines", type=int, default=80, help="Lines per retrieval chunk")
    mode_group = p_index.add_mutually_exclusive_group()
    mode_group.add_argument("--once", action="store_true", help="Run one incremental indexing pass")
    mode_group.add_argument("--daemon", action="store_true", help="Continuously run incremental indexing")
    p_index.add_argument("--interval", type=float, default=2.0, help="Daemon polling interval in seconds")
    p_index.add_argument(
        "--install-git-hooks",
        action="store_true",
        help="Install post-checkout and post-merge hooks to trigger incremental indexing",
    )
    p_index.set_defaults(func=handlers["command_index"])

    p_retrieve = sub.add_parser("retrieve", help="Retrieve repo-aware symbols/chunks for a query")
    p_retrieve.add_argument("query", help="Search query")
    p_retrieve.add_argument("--top-k", type=int, default=5)
    p_retrieve.add_argument("--path-prefix", default=None, help="Prefer paths with this prefix")
    p_retrieve.add_argument("--no-changed-bias", action="store_true", help="Disable bias toward changed git files")
    p_retrieve.add_argument("--json", action="store_true")
    p_retrieve.set_defaults(func=handlers["command_retrieve"])

    p_cursor = sub.add_parser("configure-cursor", help="Output Cursor OpenAI-compatible config")
    p_cursor.add_argument("--base-url", default=None)
    p_cursor.add_argument("--api-key", default=None)
    p_cursor.add_argument("--model", default=None)
    p_cursor.add_argument(
        "--task-tag",
        choices=sorted(task_tag_aliases.keys()),
        default=None,
        help="Select model by routing tag (fast, quality, longctx, analysis, default)",
    )
    p_cursor.set_defaults(func=handlers["command_configure_cursor"])

    p_models = sub.add_parser("models", help="List configured model profiles")
    p_models.add_argument("--json", action="store_true", help="Print model profiles as JSON")
    p_models.set_defaults(func=handlers["command_models"])

    p_route = sub.add_parser("route-model", help="Resolve model name for a task tag")
    p_route.add_argument("task_tag", choices=sorted(task_tag_aliases.keys()))
    p_route.add_argument("--json", action="store_true")
    p_route.set_defaults(func=handlers["command_route_model"])

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
    p_spec.set_defaults(func=handlers["command_spec_decode"])

    p_embed_enqueue = sub.add_parser("embed-enqueue", help="Enqueue an embedding job for background worker")
    p_embed_enqueue.add_argument("--url", default="http://localhost:8093", help="Embed queue base URL")
    p_embed_enqueue.add_argument("--timeout", type=float, default=10.0)
    p_embed_enqueue.add_argument("--kind", default="file_change", help="Job kind")
    p_embed_enqueue.add_argument("--path", default="", help="File path associated with the event")
    p_embed_enqueue.add_argument("--text", default="", help="Optional text payload to embed")
    p_embed_enqueue.add_argument("--metadata-json", default="", help="Optional JSON object string")
    p_embed_enqueue.add_argument("--max-attempts", type=int, default=3)
    p_embed_enqueue.add_argument("--json", action="store_true")
    p_embed_enqueue.set_defaults(func=handlers["command_embed_enqueue"])

    p_embed_stats = sub.add_parser("embed-stats", help="Show embed queue job stats")
    p_embed_stats.add_argument("--url", default="http://localhost:8093", help="Embed queue base URL")
    p_embed_stats.add_argument("--timeout", type=float, default=10.0)
    p_embed_stats.add_argument("--json", action="store_true")
    p_embed_stats.set_defaults(func=handlers["command_embed_stats"])

    p_memory = sub.add_parser("memory", help="Git-aware memory utilities")
    memory_sub = p_memory.add_subparsers(dest="memory_command", required=True)

    p_memory_explain = memory_sub.add_parser("explain", help="Explain retrieval scoring for a query")
    p_memory_explain.add_argument("query", help="Search query")
    p_memory_explain.add_argument("--top-k", type=int, default=5)
    p_memory_explain.add_argument("--path-prefix", default=None, help="Prefer paths with this prefix")
    p_memory_explain.add_argument("--no-changed-bias", action="store_true", help="Disable bias toward changed git files")
    p_memory_explain.add_argument("--json", action="store_true")
    p_memory_explain.set_defaults(func=handlers["command_memory_explain"])
