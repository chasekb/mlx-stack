from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


APP_DIR = Path(".ai-dev")
CONFIG_PATH = APP_DIR / "config.json"
INDEX_PATH = APP_DIR / "index.json"
INDEX_STATE_PATH = APP_DIR / "index_state.json"


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

MODEL_PATH="${MLX_MODEL_PATH:-mlx-community/Qwen3.5-Coder-7B-Instruct-4bit}"
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


AGENT_SERVER = """import json
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse
from http.server import BaseHTTPRequestHandler, HTTPServer


ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / ".ai-dev" / "index.json"
RUNS_DIR = ROOT / ".ai-dev" / "runs"
ALLOWED_TOOLS = {
    "retrieve",
    "search_code",
    "read_file",
    "git_diff",
    "run_tests",
    "write_patch",
    "commit_changes",
}

TOOL_SCHEMAS = {
    "retrieve": {
        "description": "Retrieve relevant symbols/chunks from local index",
        "input": {"query": "string", "top_k": "int?", "path_prefix": "string?"},
    },
    "search_code": {
        "description": "Regex search across repository files",
        "input": {"regex": "string", "file_pattern": "string?", "limit": "int?"},
    },
    "read_file": {
        "description": "Read a file from repo",
        "input": {"path": "string", "max_chars": "int?"},
    },
    "git_diff": {
        "description": "Get current git diff summary",
        "input": {},
    },
    "run_tests": {
        "description": "Run tests in dry-run or execute mode",
        "input": {"command": "string?"},
    },
    "write_patch": {
        "description": "Apply patch to repo (blocked in dry-run)",
        "input": {"patch": "string"},
    },
    "commit_changes": {
        "description": "Commit current changes (blocked in dry-run)",
        "input": {"message": "string"},
    },
}


def tokenize(text: str) -> list[str]:
    return [tok for tok in re.split(r"[^a-zA-Z0-9_]+", text.lower()) if len(tok) >= 2]


def retrieve(index_obj: dict, query: str, top_k: int = 5, path_prefix: Optional[str] = None) -> dict:
    query_terms = set(tokenize(query))
    if not query_terms:
        return {"query": query, "top_symbols": [], "top_chunks": []}

    path_prefix = path_prefix or ""

    symbol_results = []
    for s in index_obj.get("symbols", []):
        score = 0.0
        name_terms = set(tokenize(s.get("name", "")))
        score += len(query_terms.intersection(name_terms)) * 3
        score += 1 if any(t in s.get("name", "").lower() for t in query_terms) else 0
        p = s.get("path", "")
        if path_prefix and p.startswith(path_prefix):
            score += 1.5
        if score > 0:
            symbol_results.append({"score": score, **s})

    chunk_results = []
    for c in index_obj.get("chunks", []):
        score = 0.0
        chunk_terms = set(c.get("terms", []))
        score += len(query_terms.intersection(chunk_terms))
        p = c.get("path", "")
        if path_prefix and p.startswith(path_prefix):
            score += 2.0
        if score > 0:
            chunk_results.append(
                {
                    "score": score,
                    "path": p,
                    "chunk_id": c.get("chunk_id"),
                    "start_line": c.get("start_line"),
                    "end_line": c.get("end_line"),
                    "text_preview": c.get("text_preview", ""),
                }
            )

    symbol_results.sort(key=lambda x: x["score"], reverse=True)
    chunk_results.sort(key=lambda x: x["score"], reverse=True)

    return {
        "query": query,
        "top_symbols": symbol_results[:top_k],
        "top_chunks": chunk_results[:top_k],
    }


def ensure_under_root(path: Path) -> bool:
    try:
        path.resolve().relative_to(ROOT.resolve())
        return True
    except Exception:
        return False


def tool_search_code(args: dict) -> dict:
    regex = str(args.get("regex", "")).strip()
    if not regex:
        return {"error": "missing_regex"}
    file_pattern = str(args.get("file_pattern", "*") or "*")
    limit = int(args.get("limit", 50))
    cmd = ["bash", "-lc", f"grep -RInE --include='{file_pattern}' {json.dumps(regex)} {json.dumps(str(ROOT))} | head -n {max(1, min(limit, 200))}"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {"ok": proc.returncode in (0, 1), "output": proc.stdout.strip(), "stderr": proc.stderr.strip()}


def tool_read_file(args: dict) -> dict:
    rel = str(args.get("path", "")).strip()
    if not rel:
        return {"error": "missing_path"}
    target = (ROOT / rel).resolve()
    if not target.exists() or not target.is_file() or not ensure_under_root(target):
        return {"error": "invalid_path"}
    max_chars = int(args.get("max_chars", 12000))
    content = target.read_text(encoding="utf-8", errors="ignore")[: max(1, max_chars)]
    return {"ok": True, "path": rel, "content": content}


def tool_git_diff(_: dict) -> dict:
    proc = subprocess.run(["git", "--no-pager", "diff", "--stat"], cwd=ROOT, capture_output=True, text=True)
    return {"ok": proc.returncode == 0, "output": proc.stdout.strip(), "stderr": proc.stderr.strip()}


def tool_run_tests(args: dict, dry_run: bool) -> dict:
    command = str(args.get("command", "python3 -m pytest -q") or "python3 -m pytest -q")
    if dry_run:
        return {"ok": True, "dry_run": True, "command": command}
    proc = subprocess.run(["bash", "-lc", command], cwd=ROOT, capture_output=True, text=True)
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-8000:],
        "stderr": proc.stderr[-4000:],
    }


def tool_write_patch(args: dict, dry_run: bool) -> dict:
    if dry_run:
        return {"ok": False, "error": "blocked_in_dry_run"}
    return {"ok": False, "error": "not_implemented"}


def tool_commit_changes(args: dict, dry_run: bool) -> dict:
    if dry_run:
        return {"ok": False, "error": "blocked_in_dry_run"}
    msg = str(args.get("message", "Agent commit")).strip()
    if not msg:
        return {"ok": False, "error": "missing_message"}
    proc = subprocess.run(["git", "commit", "-am", msg], cwd=ROOT, capture_output=True, text=True)
    return {"ok": proc.returncode == 0, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}


def execute_tool_call(tool: str, args: dict, dry_run: bool) -> dict:
    if tool not in ALLOWED_TOOLS:
        return {"ok": False, "error": "tool_not_allowed", "tool": tool}
    if tool == "retrieve":
        if not INDEX_PATH.exists():
            return {"ok": False, "error": "missing_index"}
        query = str(args.get("query", "")).strip()
        if not query:
            return {"ok": False, "error": "missing_query"}
        index_obj = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        top_k = int(args.get("top_k", 5))
        path_prefix = args.get("path_prefix")
        return {"ok": True, "result": retrieve(index_obj, query=query, top_k=max(1, min(top_k, 20)), path_prefix=path_prefix)}
    if tool == "search_code":
        return tool_search_code(args)
    if tool == "read_file":
        return tool_read_file(args)
    if tool == "git_diff":
        return tool_git_diff(args)
    if tool == "run_tests":
        return tool_run_tests(args, dry_run=dry_run)
    if tool == "write_patch":
        return tool_write_patch(args, dry_run=dry_run)
    if tool == "commit_changes":
        return tool_commit_changes(args, dry_run=dry_run)
    return {"ok": False, "error": "unhandled_tool"}


def run_agent_task(payload: dict) -> dict:
    task = str(payload.get("task", "")).strip()
    dry_run = bool(payload.get("dry_run", True))
    max_steps = int(payload.get("max_steps", 6))
    max_steps = max(1, min(max_steps, 25))
    plan = payload.get("plan", [])
    run_id = uuid.uuid4().hex[:12]

    trace = {
        "run_id": run_id,
        "task": task,
        "dry_run": dry_run,
        "max_steps": max_steps,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "steps": [],
    }

    if not isinstance(plan, list) or not plan:
        trace["steps"].append({"tool": "noop", "result": {"ok": True, "detail": "No plan steps provided"}})
    else:
        for step in plan[:max_steps]:
            tool = str(step.get("tool", "")).strip()
            args = step.get("args", {}) if isinstance(step.get("args", {}), dict) else {}
            result = execute_tool_call(tool, args, dry_run=dry_run)
            trace["steps"].append({"tool": tool, "args": args, "result": result})

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_path = RUNS_DIR / f"{run_id}.json"
    run_path.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")

    return {
        "ok": True,
        "run_id": run_id,
        "run_path": str(run_path.relative_to(ROOT)),
        "step_count": len(trace["steps"]),
        "dry_run": dry_run,
        "steps": trace["steps"],
    }


class Handler(BaseHTTPRequestHandler):
    def _reply(self, payload, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode('utf-8'))

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == '/tools':
            self._reply({'ok': True, 'service': 'agent', 'tools': TOOL_SCHEMAS})
            return

        if parsed.path.startswith('/runs/'):
            run_id = parsed.path.split('/runs/', 1)[1].strip()
            target = (RUNS_DIR / f"{run_id}.json").resolve()
            if not target.exists() or not target.is_file() or not ensure_under_root(target):
                self._reply({'error': 'run_not_found'}, status=404)
                return
            payload = json.loads(target.read_text(encoding='utf-8'))
            self._reply({'ok': True, 'service': 'agent', 'run': payload})
            return

        if parsed.path == '/retrieve':
            if not INDEX_PATH.exists():
                self._reply({'error': 'missing_index', 'detail': 'Run `ai-dev index .` first.'}, status=400)
                return

            qs = parse_qs(parsed.query)
            query = (qs.get('q', [''])[0] or '').strip()
            if not query:
                self._reply({'error': 'missing_query', 'detail': 'Provide q=<query>'}, status=400)
                return

            try:
                top_k = int((qs.get('top_k', ['5'])[0] or '5'))
            except ValueError:
                top_k = 5
            top_k = max(1, min(top_k, 20))
            path_prefix = (qs.get('path_prefix', [''])[0] or '').strip() or None

            index_obj = json.loads(INDEX_PATH.read_text(encoding='utf-8'))
            payload = retrieve(index_obj, query=query, top_k=top_k, path_prefix=path_prefix)
            self._reply({'ok': True, 'service': 'agent', 'retrieval': payload})
            return

        if parsed.path == '/health':
            self._reply({'ok': True, 'service': 'agent'})
            return

        self._reply({'error': 'not found'}, status=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != '/agent/run':
            self._reply({'error': 'not found'}, status=404)
            return

        try:
            content_length = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            content_length = 0
        body = self.rfile.read(max(0, content_length))

        try:
            payload = json.loads(body.decode('utf-8') if body else '{}')
        except Exception:
            self._reply({'error': 'invalid_json'}, status=400)
            return

        result = run_agent_task(payload if isinstance(payload, dict) else {})
        self._reply({'ok': True, 'service': 'agent', 'result': result})


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

    if "routing" not in cfg or not isinstance(cfg["routing"], dict):
        cfg["routing"] = copy.deepcopy(DEFAULT_CONFIG["routing"])

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


def collect_source_files(root: Path, max_bytes: int) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for p in iter_source_files(root, max_bytes=max_bytes):
        files[str(p.relative_to(root))] = p
    return files


def tokenize(text: str) -> list[str]:
    return [tok for tok in re.split(r"[^a-zA-Z0-9_]+", text.lower()) if len(tok) >= 2]


def extract_symbols(file_path: Path, content: str) -> list[dict]:
    suffix = file_path.suffix.lower()
    symbols: list[dict] = []
    lines = content.splitlines()

    def add(name: str, line_no: int, kind: str) -> None:
        symbols.append({"name": name, "line": line_no, "kind": kind})

    for i, line in enumerate(lines, start=1):
        if suffix == ".py":
            m = re.match(r"^\s*(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", line)
            if m:
                add(m.group(2), i, m.group(1))
        elif suffix in {".js", ".ts", ".jsx", ".tsx"}:
            m = re.match(r"^\s*(export\s+)?(async\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)", line)
            if m:
                add(m.group(3), i, "function")
            m2 = re.match(r"^\s*(export\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)", line)
            if m2:
                add(m2.group(2), i, "class")
        elif suffix == ".go":
            m = re.match(r"^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_][A-Za-z0-9_]*)", line)
            if m:
                add(m.group(1), i, "func")
        elif suffix == ".rs":
            m = re.match(r"^\s*(?:pub\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)", line)
            if m:
                add(m.group(1), i, "fn")
    return symbols


def build_chunks(content: str, lines_per_chunk: int = 80) -> list[dict]:
    lines = content.splitlines()
    chunks = []
    chunk_id = 0
    for start in range(0, len(lines), lines_per_chunk):
        chunk_id += 1
        end = min(start + lines_per_chunk, len(lines))
        text = "\n".join(lines[start:end])
        tok_counter = Counter(tokenize(text))
        chunks.append(
            {
                "chunk_id": chunk_id,
                "start_line": start + 1,
                "end_line": end,
                "token_count": sum(tok_counter.values()),
                "top_terms": dict(tok_counter.most_common(15)),
                "text_preview": text[:300],
                "terms": list(tok_counter.keys()),
            }
        )
    return chunks


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


def _index_single_file(file_path: Path, root: Path, top_terms_per_file: int, chunk_lines: int) -> tuple[dict, list[dict], list[dict]]:
    rel = str(file_path.relative_to(root))
    content = file_path.read_text(encoding="utf-8", errors="ignore")
    tok_counter = Counter(tokenize(content))
    symbols = extract_symbols(file_path, content)
    chunks = build_chunks(content, lines_per_chunk=chunk_lines)

    file_entry = {
        "path": rel,
        "size": file_path.stat().st_size,
        "token_count": sum(tok_counter.values()),
        "symbol_count": len(symbols),
        "chunk_count": len(chunks),
        "top_terms": dict(tok_counter.most_common(top_terms_per_file)),
    }

    symbol_rows = [{"path": rel, **s} for s in symbols]
    chunk_rows = [{"path": rel, **c} for c in chunks]
    return file_entry, symbol_rows, chunk_rows


def run_index_pass(root: Path, args: argparse.Namespace, incremental: bool) -> tuple[dict, dict]:
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
    changed_files = get_git_changed_files(root) if not args.no_changed_bias else set()
    path_prefix = args.path_prefix or ""

    symbol_results = []
    for s in index_obj.get("symbols", []):
        score = 0.0
        name_terms = set(tokenize(s.get("name", "")))
        score += len(query_terms.intersection(name_terms)) * 3
        score += 1 if any(t in s.get("name", "").lower() for t in query_terms) else 0
        p = s.get("path", "")
        if path_prefix and p.startswith(path_prefix):
            score += 1.5
        if p in changed_files:
            score += 1.0
        if score > 0:
            symbol_results.append({"score": score, **s})

    chunk_results = []
    for c in index_obj.get("chunks", []):
        score = 0.0
        chunk_terms = set(c.get("terms", []))
        score += len(query_terms.intersection(chunk_terms))
        p = c.get("path", "")
        if path_prefix and p.startswith(path_prefix):
            score += 2.0
        if p in changed_files:
            score += 1.5
        if score > 0:
            chunk_results.append(
                {
                    "score": score,
                    "path": p,
                    "chunk_id": c.get("chunk_id"),
                    "start_line": c.get("start_line"),
                    "end_line": c.get("end_line"),
                    "text_preview": c.get("text_preview", ""),
                }
            )

    symbol_results.sort(key=lambda x: x["score"], reverse=True)
    chunk_results.sort(key=lambda x: x["score"], reverse=True)

    result = {
        "query": args.query,
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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
