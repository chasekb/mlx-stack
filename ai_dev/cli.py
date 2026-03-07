from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


APP_DIR = Path(".ai-dev")
CONFIG_PATH = APP_DIR / "config.json"
INDEX_PATH = APP_DIR / "index.json"


PODMAN_COMPOSE_YAML = """version: '3.8'

services:
  mlx:
    build:
      context: .
      dockerfile: mlx/Dockerfile
    container_name: ai-dev-mlx
    ports:
      - "8081:8081"
    command: ["/bin/bash", "/app/mlx/entrypoint.sh"]

  litellm:
    image: ghcr.io/berriai/litellm:main-latest
    container_name: ai-dev-litellm
    ports:
      - "4000:4000"
    volumes:
      - ./litellm_config.yaml:/app/config.yaml:ro
    command: ["--config", "/app/config.yaml", "--port", "4000"]
    depends_on:
      - mlx

  qdrant:
    image: qdrant/qdrant:latest
    container_name: ai-dev-qdrant
    ports:
      - "6333:6333"
    profiles: ["optional"]

  rag:
    image: python:3.11-slim
    container_name: ai-dev-rag
    working_dir: /app
    volumes:
      - ./rag:/app
    command: ["python", "server.py"]
    ports:
      - "8090:8090"
    profiles: ["optional"]

  agent:
    image: python:3.11-slim
    container_name: ai-dev-agent
    working_dir: /app
    volumes:
      - ./agent:/app
    command: ["python", "server.py"]
    ports:
      - "8091:8091"
    profiles: ["optional"]
"""


LITELLM_CONFIG = """model_list:
  - model_name: local-mlx
    litellm_params:
      model: openai/local-mlx
      api_base: http://mlx:8081/v1
      api_key: local-dev

general_settings:
  master_key: local-dev
"""


MLX_ENTRYPOINT = """#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MLX_MODEL_PATH:-mlx-community/Qwen2.5-Coder-7B-Instruct-4bit}"
PORT="${MLX_PORT:-8081}"

python -m mlx_lm.server \
  --model "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "$PORT"
"""


MLX_DOCKERFILE = """FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates bash \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir mlx-lm

WORKDIR /app
COPY mlx /app/mlx

EXPOSE 8081
"""


RAG_SERVER = """from http.server import BaseHTTPRequestHandler, HTTPServer
import json


class Handler(BaseHTTPRequestHandler):
    def _reply(self, payload, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode('utf-8'))

    def do_GET(self):
        if self.path == '/health':
            self._reply({'ok': True, 'service': 'rag'})
            return
        self._reply({'error': 'not found'}, status=404)


if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', 8090), Handler)
    print('RAG service listening on :8090')
    server.serve_forever()
"""


AGENT_SERVER = """from http.server import BaseHTTPRequestHandler, HTTPServer
import json


class Handler(BaseHTTPRequestHandler):
    def _reply(self, payload, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode('utf-8'))

    def do_GET(self):
        if self.path == '/health':
            self._reply({'ok': True, 'service': 'agent'})
            return
        self._reply({'error': 'not found'}, status=404)


if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', 8091), Handler)
    print('Agent service listening on :8091')
    server.serve_forever()
"""


DEFAULT_CONFIG = {
    "created_at": "",
    "stack": {
        "mlx_port": 8081,
        "litellm_port": 4000,
        "default_model": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    },
    "cursor": {
        "base_url": "http://localhost:4000/v1",
        "api_key": "local-dev",
        "model": "local-mlx",
    },
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
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg = DEFAULT_CONFIG.copy()
    cfg["created_at"] = datetime.now(timezone.utc).isoformat()
    return cfg


def command_init(_: argparse.Namespace) -> int:
    APP_DIR.mkdir(parents=True, exist_ok=True)

    write_file(Path("podman-compose.yml"), PODMAN_COMPOSE_YAML)
    write_file(Path("litellm_config.yaml"), LITELLM_CONFIG)
    write_file(Path("mlx/entrypoint.sh"), MLX_ENTRYPOINT, executable=True)
    write_file(Path("mlx/Dockerfile"), MLX_DOCKERFILE)
    write_file(Path("rag/server.py"), RAG_SERVER)
    write_file(Path("agent/server.py"), AGENT_SERVER)

    config = load_config()
    config["created_at"] = config.get("created_at") or datetime.now(timezone.utc).isoformat()
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
    model = args.model
    quant = args.quantization
    commands = [
        "# Convert/pull a HF model into MLX format",
        f"python -m mlx_lm.convert --hf-path '{model}' --quantize {quant}",
        "",
        "# Test local MLX server model loading",
        "python -m mlx_lm.server --model ./mlx_model --host 0.0.0.0 --port 8081",
    ]
    print("\n".join(commands))
    return 0


def iter_source_files(root: Path, max_bytes: int) -> Iterable[Path]:
    skip_dirs = {".git", ".venv", "node_modules", "__pycache__", ".ai-dev"}
    allowed = {
        ".py",
        ".md",
        ".txt",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".sh",
        ".sql",
        ".go",
        ".rs",
        ".java",
        ".c",
        ".cpp",
        ".h",
    }
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in skip_dirs for part in p.parts):
            continue
        if p.suffix.lower() not in allowed:
            continue
        if p.stat().st_size > max_bytes:
            continue
        yield p


def tokenize(text: str) -> list[str]:
    return [tok for tok in re.split(r"[^a-zA-Z0-9_]+", text.lower()) if len(tok) >= 2]


def command_index(args: argparse.Namespace) -> int:
    root = Path(args.path).resolve()
    if not root.exists() or not root.is_dir():
        print(f"Path not found or not a directory: {root}", file=sys.stderr)
        return 2

    APP_DIR.mkdir(parents=True, exist_ok=True)
    file_entries = []
    vocabulary = Counter()
    total_tokens = 0

    for file_path in iter_source_files(root, max_bytes=args.max_file_size):
        rel = str(file_path.relative_to(root))
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        toks = tokenize(content)
        tok_counter = Counter(toks)
        vocabulary.update(tok_counter)
        total_tokens += sum(tok_counter.values())
        file_entries.append(
            {
                "path": rel,
                "size": file_path.stat().st_size,
                "token_count": sum(tok_counter.values()),
                "top_terms": dict(tok_counter.most_common(args.top_terms_per_file)),
            }
        )

    index_obj = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "file_count": len(file_entries),
        "total_tokens": total_tokens,
        "top_terms_global": dict(vocabulary.most_common(args.top_terms_global)),
        "files": file_entries,
    }
    write_file(INDEX_PATH, json.dumps(index_obj, indent=2) + "\n")
    print(f"Indexed {len(file_entries)} files -> {INDEX_PATH}")
    return 0


def command_configure_cursor(args: argparse.Namespace) -> int:
    cfg = load_config()
    cursor_cfg = {
        "name": "Local LiteLLM",
        "provider": "openai",
        "baseUrl": args.base_url or cfg["cursor"]["base_url"],
        "apiKey": args.api_key or cfg["cursor"]["api_key"],
        "model": args.model or cfg["cursor"]["model"],
    }

    APP_DIR.mkdir(parents=True, exist_ok=True)
    output_path = APP_DIR / "cursor-openai.json"
    write_file(output_path, json.dumps(cursor_cfg, indent=2) + "\n")

    print("Use the following OpenAI-compatible model config in Cursor:")
    print(json.dumps(cursor_cfg, indent=2))
    print(f"\nSaved: {output_path}")
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

    p_pull = sub.add_parser("pull-models", help="Print MLX model conversion commands")
    p_pull.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct", help="HuggingFace model id")
    p_pull.add_argument("--quantization", default="4", help="Quantization bits for mlx_lm.convert")
    p_pull.set_defaults(func=command_pull_models)

    p_index = sub.add_parser("index", help="Build lightweight lexical index")
    p_index.add_argument("path", nargs="?", default=".", help="Directory to index")
    p_index.add_argument("--max-file-size", type=int, default=512_000, help="Max file size in bytes")
    p_index.add_argument("--top-terms-per-file", type=int, default=20)
    p_index.add_argument("--top-terms-global", type=int, default=100)
    p_index.set_defaults(func=command_index)

    p_cursor = sub.add_parser("configure-cursor", help="Output Cursor OpenAI-compatible config")
    p_cursor.add_argument("--base-url", default=None)
    p_cursor.add_argument("--api-key", default=None)
    p_cursor.add_argument("--model", default=None)
    p_cursor.set_defaults(func=command_configure_cursor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
