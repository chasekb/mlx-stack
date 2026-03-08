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

  spec-router:
    image: python:3.11-slim
    container_name: ai-dev-spec-router
    working_dir: /app
    volumes:
      - ./spec_router:/app
    command: ["python", "server.py"]
    ports:
      - "8092:8092"
    profiles: ["optional"]

  embed-queue:
    image: python:3.11-slim
    container_name: ai-dev-embed-queue
    working_dir: /app
    volumes:
      - ./embedding_queue:/app
    command: ["python", "server.py"]
    ports:
      - "8093:8093"
    profiles: ["optional"]

  embed-worker:
    image: python:3.11-slim
    container_name: ai-dev-embed-worker
    working_dir: /app
    volumes:
      - ./embedding_worker:/app
      - ./embedding_queue:/queue
    command: ["python", "worker.py", "--queue-url", "http://embed-queue:8093"]
    depends_on:
      - embed-queue
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


SPEC_ROUTER_SERVER = """import json
from http.server import BaseHTTPRequestHandler, HTTPServer


def run_speculative_loop(draft_tokens: list[str], target_tokens: list[str]) -> dict:
    accepted = 0
    compared = min(len(draft_tokens), len(target_tokens))
    out_tokens: list[str] = []

    for i in range(compared):
        d = draft_tokens[i]
        t = target_tokens[i]
        if d == t:
            accepted += 1
            out_tokens.append(d)
        else:
            out_tokens.append(t)

    if len(target_tokens) > compared:
        out_tokens.extend(target_tokens[compared:])

    acceptance_rate = (accepted / compared) if compared else 0.0
    return {
        "accepted_tokens": accepted,
        "compared_tokens": compared,
        "acceptance_rate": round(acceptance_rate, 4),
        "output_tokens": out_tokens,
    }


class Handler(BaseHTTPRequestHandler):
    def _reply(self, payload: dict, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def do_GET(self):
        if self.path == "/health":
            self._reply({"ok": True, "service": "spec-router"})
            return
        self._reply({"error": "not found"}, status=404)

    def do_POST(self):
        if self.path != "/spec/decode":
            self._reply({"error": "not found"}, status=404)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        body = self.rfile.read(max(0, content_length))

        try:
            payload = json.loads(body.decode("utf-8") if body else "{}")
        except Exception:
            self._reply({"error": "invalid_json"}, status=400)
            return

        draft_tokens = payload.get("draft_tokens", [])
        target_tokens = payload.get("target_tokens", [])
        if not isinstance(draft_tokens, list) or not isinstance(target_tokens, list):
            self._reply({"error": "invalid_tokens", "detail": "draft_tokens and target_tokens must be arrays"}, status=400)
            return

        result = run_speculative_loop(
            draft_tokens=[str(t) for t in draft_tokens],
            target_tokens=[str(t) for t in target_tokens],
        )
        self._reply({"ok": True, "service": "spec-router", "result": result})


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 8092), Handler)
    print("Spec router listening on :8092")
    server.serve_forever()
"""


EMBED_QUEUE_SERVER = """from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer


DB_PATH = Path('.ai-dev/embedding_jobs.db')


def _db_connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _db_connect() as conn:
        conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS embedding_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                max_attempts INTEGER NOT NULL DEFAULT 3,
                next_attempt_at REAL NOT NULL DEFAULT 0,
                last_error TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            '''
        )
        conn.commit()


def enqueue_job(kind: str, payload: dict, max_attempts: int = 3) -> dict:
    now = time.time()
    with _db_connect() as conn:
        cur = conn.execute(
            '''
            INSERT INTO embedding_jobs (
                kind, payload_json, status, attempts, max_attempts, next_attempt_at, created_at, updated_at
            ) VALUES (?, ?, 'queued', 0, ?, 0, ?, ?)
            ''',
            (kind, json.dumps(payload), max(1, int(max_attempts)), now, now),
        )
        conn.commit()
        return {'job_id': int(cur.lastrowid), 'status': 'queued'}


def claim_next_job() -> dict | None:
    now = time.time()
    with _db_connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        row = conn.execute(
            '''
            SELECT * FROM embedding_jobs
            WHERE status IN ('queued', 'retry')
              AND next_attempt_at <= ?
            ORDER BY id ASC
            LIMIT 1
            ''',
            (now,),
        ).fetchone()

        if row is None:
            conn.commit()
            return None

        next_attempts = int(row['attempts']) + 1
        conn.execute(
            '''
            UPDATE embedding_jobs
            SET status='in_progress', attempts=?, updated_at=?
            WHERE id=?
            ''',
            (next_attempts, now, int(row['id'])),
        )
        conn.commit()

        return {
            'id': int(row['id']),
            'kind': row['kind'],
            'payload': json.loads(row['payload_json']),
            'attempts': next_attempts,
            'max_attempts': int(row['max_attempts']),
        }


def complete_job(job_id: int) -> None:
    now = time.time()
    with _db_connect() as conn:
        conn.execute(
            "UPDATE embedding_jobs SET status='done', updated_at=? WHERE id=?",
            (now, int(job_id)),
        )
        conn.commit()


def fail_job(job_id: int, error: str) -> dict:
    now = time.time()
    with _db_connect() as conn:
        row = conn.execute(
            'SELECT attempts, max_attempts FROM embedding_jobs WHERE id=?',
            (int(job_id),),
        ).fetchone()
        if row is None:
            return {'error': 'job_not_found'}

        attempts = int(row['attempts'])
        max_attempts = int(row['max_attempts'])
        if attempts >= max_attempts:
            status = 'dead_letter'
            next_attempt_at = 0
        else:
            status = 'retry'
            next_attempt_at = now + min(60.0, float(2 ** attempts))

        conn.execute(
            '''
            UPDATE embedding_jobs
            SET status=?, next_attempt_at=?, last_error=?, updated_at=?
            WHERE id=?
            ''',
            (status, next_attempt_at, str(error)[:2000], now, int(job_id)),
        )
        conn.commit()
        return {'ok': True, 'status': status, 'next_attempt_at': next_attempt_at}


def get_stats() -> dict:
    with _db_connect() as conn:
        counts = {
            row['status']: int(row['count'])
            for row in conn.execute(
                'SELECT status, COUNT(*) AS count FROM embedding_jobs GROUP BY status'
            ).fetchall()
        }
    return {
        'queued': counts.get('queued', 0),
        'retry': counts.get('retry', 0),
        'in_progress': counts.get('in_progress', 0),
        'done': counts.get('done', 0),
        'dead_letter': counts.get('dead_letter', 0),
    }


class Handler(BaseHTTPRequestHandler):
    def _reply(self, payload: dict, status: int = 200) -> None:
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode('utf-8'))

    def _read_json(self) -> dict:
        try:
            n = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            n = 0
        body = self.rfile.read(max(0, n))
        if not body:
            return {}
        try:
            parsed = json.loads(body.decode('utf-8'))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def do_GET(self):
        if self.path == '/health':
            self._reply({'ok': True, 'service': 'embed-queue', 'db_path': str(DB_PATH)})
            return
        if self.path == '/stats':
            self._reply({'ok': True, 'service': 'embed-queue', 'stats': get_stats()})
            return
        self._reply({'error': 'not found'}, status=404)

    def do_POST(self):
        if self.path == '/jobs/enqueue':
            payload = self._read_json()
            kind = str(payload.get('kind', 'file_change') or 'file_change')
            job_payload = payload.get('payload', {}) if isinstance(payload.get('payload', {}), dict) else {}
            max_attempts = int(payload.get('max_attempts', 3) or 3)
            out = enqueue_job(kind=kind, payload=job_payload, max_attempts=max_attempts)
            self._reply({'ok': True, 'service': 'embed-queue', **out})
            return

        if self.path == '/jobs/claim':
            job = claim_next_job()
            self._reply({'ok': True, 'service': 'embed-queue', 'job': job})
            return

        if self.path == '/jobs/complete':
            payload = self._read_json()
            job_id = int(payload.get('job_id', 0) or 0)
            if job_id <= 0:
                self._reply({'error': 'missing_job_id'}, status=400)
                return
            complete_job(job_id)
            self._reply({'ok': True, 'service': 'embed-queue', 'job_id': job_id, 'status': 'done'})
            return

        if self.path == '/jobs/fail':
            payload = self._read_json()
            job_id = int(payload.get('job_id', 0) or 0)
            if job_id <= 0:
                self._reply({'error': 'missing_job_id'}, status=400)
                return
            error = str(payload.get('error', 'unknown_error'))
            out = fail_job(job_id=job_id, error=error)
            if out.get('error'):
                self._reply(out, status=404)
                return
            self._reply({'ok': True, 'service': 'embed-queue', 'job_id': job_id, **out})
            return

        self._reply({'error': 'not found'}, status=404)


if __name__ == '__main__':
    _init_db()
    server = HTTPServer(('0.0.0.0', 8093), Handler)
    print('Embedding queue listening on :8093')
    server.serve_forever()
"""


EMBED_WORKER = """from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def http_json(method: str, url: str, payload: dict | None = None, timeout: float = 10.0) -> dict:
    data = json.dumps(payload or {}).encode('utf-8') if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={'Content-Type': 'application/json'},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode('utf-8')
    return json.loads(raw) if raw else {}


def fake_embed(text: str, dims: int = 16) -> list[float]:
    digest = hashlib.sha256((text or '').encode('utf-8')).digest()
    vals = [((digest[i % len(digest)] / 255.0) * 2.0 - 1.0) for i in range(dims)]
    norm = math.sqrt(sum(v * v for v in vals)) or 1.0
    return [round(v / norm, 6) for v in vals]


def process_job(job: dict, output_path: Path) -> None:
    payload = job.get('payload', {}) if isinstance(job.get('payload', {}), dict) else {}
    source_text = str(payload.get('text', '') or '')
    if not source_text:
        source_text = str(payload.get('path', '') or '')
    if not source_text:
        source_text = json.dumps(payload, sort_keys=True)

    rec = {
        'embedded_at': datetime.now(timezone.utc).isoformat(),
        'job_id': int(job.get('id', 0) or 0),
        'kind': str(job.get('kind', 'unknown')),
        'metadata': payload,
        'vector': fake_embed(source_text),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(rec) + '\\n')


def run_once(queue_url: str, output_path: Path, timeout: float) -> bool:
    claim = http_json('POST', queue_url.rstrip('/') + '/jobs/claim', payload={}, timeout=timeout)
    job = claim.get('job') if isinstance(claim, dict) else None
    if not isinstance(job, dict):
        return False

    job_id = int(job.get('id', 0) or 0)
    if job_id <= 0:
        return False

    try:
        process_job(job=job, output_path=output_path)
    except Exception as e:
        http_json(
            'POST',
            queue_url.rstrip('/') + '/jobs/fail',
            payload={'job_id': job_id, 'error': str(e)},
            timeout=timeout,
        )
        return True

    http_json(
        'POST',
        queue_url.rstrip('/') + '/jobs/complete',
        payload={'job_id': job_id},
        timeout=timeout,
    )
    return True


def main() -> int:
    p = argparse.ArgumentParser(description='Background embedding worker')
    p.add_argument('--queue-url', default='http://localhost:8093')
    p.add_argument('--output-path', default='.ai-dev/embeddings.jsonl')
    p.add_argument('--poll-interval', type=float, default=2.0)
    p.add_argument('--timeout', type=float, default=10.0)
    p.add_argument('--once', action='store_true', help='Process at most one available job and exit')
    args = p.parse_args()

    output_path = Path(args.output_path)

    if args.once:
        try:
            run_once(queue_url=args.queue_url, output_path=output_path, timeout=args.timeout)
            return 0
        except urllib.error.URLError as e:
            print(f'worker failed to reach queue: {e}')
            return 2

    print(f'Embedding worker polling {args.queue_url} every {args.poll_interval}s')
    while True:
        try:
            processed = run_once(queue_url=args.queue_url, output_path=output_path, timeout=args.timeout)
        except urllib.error.URLError as e:
            print(f'worker queue error: {e}')
            processed = False

        if not processed:
            time.sleep(max(0.25, args.poll_interval))


if __name__ == '__main__':
    raise SystemExit(main())
"""


AGENT_SERVER = """import json
import re
import subprocess
import uuid
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse
from http.server import BaseHTTPRequestHandler, HTTPServer


ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / ".ai-dev" / "index.json"
RUNS_DIR = ROOT / ".ai-dev" / "runs"
CACHE_PATH = ROOT / ".ai-dev" / "prompt_cache.json"
METRICS_PATH = ROOT / ".ai-dev" / "metrics.json"
KV_CACHE_PATH = ROOT / ".ai-dev" / "kv_cache.json"
DEFAULT_CACHE_TTL_SECONDS = 600
DEFAULT_KV_MODEL_BUDGET_TOKENS = 8000
DEFAULT_KV_ENTRY_MAX_TOKENS = 2048
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

PATCH_DENY_PREFIXES = (
    ".git/",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json_file(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json_file(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def get_git_branch() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return "unknown"
    return (proc.stdout or "").strip() or "unknown"


def get_index_signature() -> str:
    if not INDEX_PATH.exists():
        return "no-index"
    try:
        index_obj = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except Exception:
        return "index-unreadable"
    generated_at = str(index_obj.get("generated_at", "unknown"))
    schema_version = str(index_obj.get("schema_version", "?"))
    file_count = str(index_obj.get("file_count", "?"))
    return f"sv{schema_version}:{generated_at}:{file_count}"


def compute_cache_namespace() -> str:
    return f"branch={get_git_branch()}|index={get_index_signature()}"


def normalize_task_payload(payload: dict) -> dict:
    return {
        "task": str(payload.get("task", "")).strip(),
        "model": payload.get("model"),
        "dry_run": bool(payload.get("dry_run", True)),
        "max_steps": int(payload.get("max_steps", 6)),
        "plan": payload.get("plan", []),
        "tool_context_hash": payload.get("tool_context_hash"),
        "options": payload.get("options", {}),
    }


def compute_cache_key(payload: dict) -> str:
    canonical = json.dumps(normalize_task_payload(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_cache() -> dict:
    cache = load_json_file(CACHE_PATH, {"schema_version": 1, "updated_at": utc_now_iso(), "entries": {}})
    if not isinstance(cache, dict):
        return {"schema_version": 1, "updated_at": utc_now_iso(), "entries": {}}
    if not isinstance(cache.get("entries"), dict):
        cache["entries"] = {}
    return cache


def save_cache(cache_obj: dict) -> None:
    cache_obj["updated_at"] = utc_now_iso()
    save_json_file(CACHE_PATH, cache_obj)


def get_cache_entry(cache_obj: dict, key: str, namespace: str) -> Optional[dict]:
    entry = cache_obj.get("entries", {}).get(key)
    if not isinstance(entry, dict):
        return None
    if entry.get("namespace") != namespace:
        return None
    expires_at = float(entry.get("expires_at_epoch", 0.0) or 0.0)
    now = time.time()
    if expires_at and now > expires_at:
        cache_obj.get("entries", {}).pop(key, None)
        return None
    return entry


def set_cache_entry(cache_obj: dict, key: str, namespace: str, result: dict, ttl_seconds: int) -> None:
    now = time.time()
    cache_obj.setdefault("entries", {})[key] = {
        "namespace": namespace,
        "created_at": utc_now_iso(),
        "created_at_epoch": now,
        "expires_at_epoch": now + max(1, ttl_seconds),
        "result": result,
    }


def load_metrics() -> dict:
    metrics = load_json_file(
        METRICS_PATH,
        {
            "schema_version": 1,
            "updated_at": utc_now_iso(),
            "cache": {
                "requests": 0,
                "hits": 0,
                "misses": 0,
                "hit_rate": 0.0,
                "saved_calls": 0,
                "compute_ms_total": 0.0,
                "avg_compute_ms": 0.0,
            },
        },
    )
    if not isinstance(metrics, dict):
        return {"schema_version": 1, "updated_at": utc_now_iso(), "cache": {}}
    metrics.setdefault("cache", {})
    return metrics


def record_cache_metrics(hit: bool, compute_ms: float, namespace: str, key: str) -> None:
    metrics = load_metrics()
    cache_metrics = metrics.setdefault("cache", {})
    requests = int(cache_metrics.get("requests", 0)) + 1
    hits = int(cache_metrics.get("hits", 0)) + (1 if hit else 0)
    misses = int(cache_metrics.get("misses", 0)) + (0 if hit else 1)
    saved_calls = int(cache_metrics.get("saved_calls", 0)) + (1 if hit else 0)
    compute_ms_total = float(cache_metrics.get("compute_ms_total", 0.0)) + max(0.0, compute_ms)

    cache_metrics.update(
        {
            "requests": requests,
            "hits": hits,
            "misses": misses,
            "hit_rate": round(hits / requests, 4) if requests else 0.0,
            "saved_calls": saved_calls,
            "compute_ms_total": round(compute_ms_total, 4),
            "avg_compute_ms": round(compute_ms_total / misses, 4) if misses else 0.0,
            "last_namespace": namespace,
            "last_key": key,
            "last_status": "hit" if hit else "miss",
            "last_updated": utc_now_iso(),
        }
    )
    metrics["updated_at"] = utc_now_iso()
    save_json_file(METRICS_PATH, metrics)


def estimate_tokens(text: str) -> int:
    return max(1, len([t for t in re.split(r"\s+", (text or "").strip()) if t]))


def normalize_prefix_text(payload: dict) -> str:
    raw = payload.get("prefix")
    if raw is None:
        raw = payload.get("prompt_prefix")
    if raw is None:
        raw = ""
    return str(raw).strip()


def hash_prefix(prefix: str) -> str:
    return hashlib.sha256((prefix or "").encode("utf-8")).hexdigest()


def load_kv_cache() -> dict:
    obj = load_json_file(KV_CACHE_PATH, {"schema_version": 1, "updated_at": utc_now_iso(), "models": {}})
    if not isinstance(obj, dict):
        obj = {"schema_version": 1, "updated_at": utc_now_iso(), "models": {}}
    if not isinstance(obj.get("models"), dict):
        obj["models"] = {}
    return obj


def save_kv_cache(obj: dict) -> None:
    obj["updated_at"] = utc_now_iso()
    save_json_file(KV_CACHE_PATH, obj)


def _ensure_model_store(kv_obj: dict, model: str, budget_tokens: int) -> dict:
    models = kv_obj.setdefault("models", {})
    store = models.get(model)
    if not isinstance(store, dict):
        store = {
            "budget_tokens": max(256, int(budget_tokens)),
            "entries": {},
            "updated_at": utc_now_iso(),
        }
        models[model] = store
    store["budget_tokens"] = max(256, int(budget_tokens))
    if not isinstance(store.get("entries"), dict):
        store["entries"] = {}
    return store


def _enforce_model_budget(store: dict, keep_key: str) -> list[str]:
    entries = store.get("entries", {}) if isinstance(store.get("entries", {}), dict) else {}
    budget = max(256, int(store.get("budget_tokens", DEFAULT_KV_MODEL_BUDGET_TOKENS)))
    evicted: list[str] = []

    def used_tokens() -> int:
        return sum(int(v.get("token_estimate", 0) or 0) for v in entries.values() if isinstance(v, dict))

    while used_tokens() > budget and entries:
        candidates = [(k, v) for k, v in entries.items() if isinstance(v, dict) and k != keep_key]
        if not candidates:
            break
        candidates.sort(key=lambda kv: float(kv[1].get("updated_at_epoch", 0.0) or 0.0))
        drop_key = candidates[0][0]
        entries.pop(drop_key, None)
        evicted.append(drop_key)

    store["entries"] = entries
    store["used_tokens"] = used_tokens()
    store["updated_at"] = utc_now_iso()
    return evicted


def get_kv_reuse_status(payload: dict) -> dict:
    kv_cfg = payload.get("kv_cache", {}) if isinstance(payload.get("kv_cache", {}), dict) else {}
    enabled = bool(kv_cfg.get("enabled", True))
    if not enabled:
        return {"enabled": False, "status": "disabled"}

    tenant_id = str(kv_cfg.get("tenant_id", "default")).strip() or "default"
    session_id = str(kv_cfg.get("session_id", "")).strip()
    if not session_id:
        return {"enabled": True, "status": "bypass", "reason": "missing_session_id", "tenant_id": tenant_id}

    model = str(payload.get("model") or kv_cfg.get("model") or "default").strip() or "default"
    prefix = normalize_prefix_text(kv_cfg)
    if not prefix:
        return {
            "enabled": True,
            "status": "bypass",
            "reason": "missing_prefix",
            "tenant_id": tenant_id,
            "session_id": session_id,
            "model": model,
        }

    provided_hash = str(kv_cfg.get("prefix_hash", "")).strip()
    computed_hash = hash_prefix(prefix)
    if provided_hash and provided_hash != computed_hash:
        return {
            "enabled": True,
            "status": "rejected",
            "reason": "prefix_hash_mismatch",
            "tenant_id": tenant_id,
            "session_id": session_id,
            "model": model,
        }

    max_entry_tokens = max(1, int(kv_cfg.get("entry_max_tokens", DEFAULT_KV_ENTRY_MAX_TOKENS)))
    token_estimate = min(estimate_tokens(prefix), max_entry_tokens)
    model_budget_tokens = max(256, int(kv_cfg.get("model_budget_tokens", DEFAULT_KV_MODEL_BUDGET_TOKENS)))

    kv_obj = load_kv_cache()
    model_store = _ensure_model_store(kv_obj, model=model, budget_tokens=model_budget_tokens)
    entries = model_store.get("entries", {})
    session_key = f"{tenant_id}|{session_id}|{model}"
    prev = entries.get(session_key) if isinstance(entries, dict) else None

    reused = False
    reuse_reason = "cold_start"
    reused_tokens = 0
    if isinstance(prev, dict):
        prev_prefix = str(prev.get("prefix", ""))
        if prefix.startswith(prev_prefix):
            reused = True
            reuse_reason = "prefix_extension"
            reused_tokens = min(int(prev.get("token_estimate", 0) or 0), token_estimate)
        else:
            reuse_reason = "prefix_boundary_mismatch"

    now_epoch = time.time()
    entries[session_key] = {
        "tenant_id": tenant_id,
        "session_id": session_id,
        "model": model,
        "prefix": prefix,
        "prefix_hash": computed_hash,
        "prefix_chars": len(prefix),
        "token_estimate": token_estimate,
        "updated_at": utc_now_iso(),
        "updated_at_epoch": now_epoch,
    }
    model_store["entries"] = entries
    evicted = _enforce_model_budget(model_store, keep_key=session_key)
    save_kv_cache(kv_obj)

    return {
        "enabled": True,
        "status": "hit" if reused else "miss",
        "reason": reuse_reason,
        "tenant_id": tenant_id,
        "session_id": session_id,
        "model": model,
        "session_key": session_key,
        "prefix_hash": computed_hash,
        "token_estimate": token_estimate,
        "reused_tokens": reused_tokens,
        "model_budget_tokens": int(model_store.get("budget_tokens", model_budget_tokens)),
        "model_used_tokens": int(model_store.get("used_tokens", 0)),
        "evicted_entries": evicted,
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


def normalize_patch_path(path_text: str) -> Optional[str]:
    rel = str(path_text or "").strip().strip('"').strip("'")
    if not rel or rel == "/dev/null":
        return None
    rel = rel.replace("\\", "/")
    if rel.startswith("a/") or rel.startswith("b/"):
        rel = rel[2:]
    while rel.startswith("./"):
        rel = rel[2:]
    if not rel:
        return None
    return rel


def extract_patch_paths(patch_text: str) -> list[str]:
    seen: dict[str, bool] = {}
    lines = patch_text.splitlines()

    for line in lines:
        rel = None
        if line.startswith("diff --git "):
            m = re.match(r"^diff --git a/(.+?) b/(.+?)$", line)
            if m:
                rel = normalize_patch_path(m.group(2))
        elif line.startswith("+++ "):
            rel = normalize_patch_path(line[4:])
        elif line.startswith("*** Add File:") or line.startswith("*** Update File:") or line.startswith("*** Delete File:"):
            m = re.match(r"^\*\*\* (?:Add|Update|Delete) File: (.+?)(?:\s+->.+)?$", line)
            if m:
                rel = normalize_patch_path(m.group(1))

        if rel:
            seen[rel] = True

    return sorted(seen.keys())


def path_allowed_for_patch(rel_path: str) -> bool:
    if not rel_path:
        return False
    p = Path(rel_path)
    if p.is_absolute() or ".." in p.parts:
        return False
    normalized = rel_path.replace("\\", "/")
    for deny_prefix in PATCH_DENY_PREFIXES:
        if normalized.startswith(deny_prefix):
            return False
    target = (ROOT / p).resolve()
    return ensure_under_root(target)


def snapshot_paths(rel_paths: list[str]) -> dict:
    snapshot = {}
    for rel in rel_paths:
        target = (ROOT / rel).resolve()
        if target.exists() and target.is_dir():
            raise ValueError(f"target_is_directory:{rel}")
        if target.exists() and target.is_file():
            snapshot[rel] = {"exists": True, "content": target.read_text(encoding="utf-8", errors="ignore")}
        else:
            snapshot[rel] = {"exists": False, "content": ""}
    return snapshot


def restore_snapshot(snapshot: dict) -> None:
    for rel, prior in snapshot.items():
        target = (ROOT / rel).resolve()
        if not ensure_under_root(target):
            continue
        if bool(prior.get("exists", False)):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(prior.get("content", "")), encoding="utf-8")
        else:
            if target.exists() and target.is_file():
                target.unlink()


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

    patch = str(args.get("patch", "") or "")
    if not patch.strip():
        return {"ok": False, "error": "missing_patch"}
    if len(patch) > 500_000:
        return {"ok": False, "error": "patch_too_large", "max_chars": 500000}

    rel_paths = extract_patch_paths(patch)
    if not rel_paths:
        return {"ok": False, "error": "no_target_files_detected"}

    denied = [p for p in rel_paths if not path_allowed_for_patch(p)]
    if denied:
        return {"ok": False, "error": "patch_target_denied", "denied_paths": denied}

    try:
        before_state = snapshot_paths(rel_paths)
    except ValueError as exc:
        return {"ok": False, "error": "invalid_patch_target", "detail": str(exc)}

    preflight = subprocess.run(
        ["git", "apply", "--check", "--whitespace=nowarn", "-"],
        cwd=ROOT,
        input=patch,
        text=True,
        capture_output=True,
    )
    if preflight.returncode != 0:
        return {
            "ok": False,
            "error": "preflight_failed",
            "stderr": (preflight.stderr or "").strip(),
            "stdout": (preflight.stdout or "").strip(),
        }

    apply_proc = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", "-"],
        cwd=ROOT,
        input=patch,
        text=True,
        capture_output=True,
    )
    if apply_proc.returncode != 0:
        return {
            "ok": False,
            "error": "apply_failed",
            "stderr": (apply_proc.stderr or "").strip(),
            "stdout": (apply_proc.stdout or "").strip(),
        }

    failed_verification = []
    for rel in rel_paths:
        target = (ROOT / rel).resolve()
        if not ensure_under_root(target):
            failed_verification.append(rel)

    if failed_verification:
        restore_snapshot(before_state)
        return {
            "ok": False,
            "error": "post_apply_verification_failed",
            "invalid_paths": failed_verification,
            "rolled_back": True,
        }

    return {"ok": True, "applied_files": rel_paths, "file_count": len(rel_paths)}


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

        if parsed.path == '/metrics':
            metrics = load_metrics()
            kv_obj = load_kv_cache()
            kv_summary = {}
            for model_name, store in kv_obj.get("models", {}).items():
                if not isinstance(store, dict):
                    continue
                entries = store.get("entries", {}) if isinstance(store.get("entries", {}), dict) else {}
                kv_summary[model_name] = {
                    "entries": len(entries),
                    "used_tokens": int(store.get("used_tokens", 0)),
                    "budget_tokens": int(store.get("budget_tokens", DEFAULT_KV_MODEL_BUDGET_TOKENS)),
                }
            self._reply({'ok': True, 'service': 'agent', 'metrics': metrics, 'kv_cache': {'models': kv_summary}})
            return

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

        payload = payload if isinstance(payload, dict) else {}
        cache_cfg = payload.get("cache", {}) if isinstance(payload.get("cache", {}), dict) else {}
        cache_enabled = bool(cache_cfg.get("enabled", True))
        cache_refresh = bool(cache_cfg.get("refresh", False))
        ttl_seconds = int(cache_cfg.get("ttl_seconds", DEFAULT_CACHE_TTL_SECONDS) or DEFAULT_CACHE_TTL_SECONDS)
        ttl_seconds = max(1, min(ttl_seconds, 86_400))

        namespace = compute_cache_namespace()
        key = compute_cache_key(payload)
        cache_hit = False
        kv_status = get_kv_reuse_status(payload)
        started = time.perf_counter()
        result = None

        if cache_enabled and not cache_refresh:
            cache_obj = load_cache()
            entry = get_cache_entry(cache_obj, key=key, namespace=namespace)
            if entry and isinstance(entry.get("result"), dict):
                result = entry["result"]
                cache_hit = True
                save_cache(cache_obj)

        if result is None:
            result = run_agent_task(payload)
            if cache_enabled:
                cache_obj = load_cache()
                set_cache_entry(cache_obj, key=key, namespace=namespace, result=result, ttl_seconds=ttl_seconds)
                save_cache(cache_obj)

        compute_ms = (time.perf_counter() - started) * 1000.0
        record_cache_metrics(hit=cache_hit, compute_ms=compute_ms, namespace=namespace, key=key)

        self._reply(
            {
                'ok': True,
                'service': 'agent',
                'result': result,
                'cache': {
                    'enabled': cache_enabled,
                    'refresh': cache_refresh,
                    'hit': cache_hit,
                    'ttl_seconds': ttl_seconds,
                    'namespace': namespace,
                    'key': key,
                    'compute_ms': round(compute_ms, 2),
                },
                'kv_cache': kv_status,
            }
        )


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
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _recency_boost_from_commit_ts(commit_ts: int, now_ts: float) -> float:
    if commit_ts <= 0:
        return 0.0
    age_days = max(0.0, (now_ts - float(commit_ts)) / 86_400.0)
    return max(0.0, round(1.5 * (2.0 / (2.0 + age_days)), 4))


def _score_symbol_match(
    symbol: dict,
    query_terms: set[str],
    path_prefix: str,
    changed_files: set[str],
    current_branch: str,
    include_changed_bias: bool,
    now_ts: float,
) -> dict | None:
    p = str(symbol.get("path", ""))
    name = str(symbol.get("name", ""))

    lexical_score = 0.0
    name_terms = set(tokenize(name))
    lexical_score += len(query_terms.intersection(name_terms)) * 3.0
    lexical_score += 1.0 if any(t in name.lower() for t in query_terms) else 0.0

    path_score = 1.5 if path_prefix and p.startswith(path_prefix) else 0.0
    changed_score = 1.0 if include_changed_bias and p in changed_files else 0.0

    branch = str(symbol.get("git_branch", "") or "")
    branch_score = 0.8 if current_branch != "unknown" and branch == current_branch else 0.0

    recency_raw = _recency_boost_from_commit_ts(_safe_int(symbol.get("git_commit_ts", 0), 0), now_ts)
    recency_score = round(recency_raw * 0.6, 4)

    total = lexical_score + path_score + changed_score + branch_score + recency_score
    if total <= 0:
        return None

    return {
        "score": round(total, 4),
        "score_breakdown": {
            "lexical": round(lexical_score, 4),
            "path_prefix": round(path_score, 4),
            "changed_file": round(changed_score, 4),
            "branch_match": round(branch_score, 4),
            "recency": round(recency_score, 4),
            "recency_raw": round(recency_raw, 4),
        },
    }


def _score_chunk_match(
    chunk: dict,
    query_terms: set[str],
    path_prefix: str,
    changed_files: set[str],
    current_branch: str,
    include_changed_bias: bool,
    now_ts: float,
) -> dict | None:
    p = str(chunk.get("path", ""))
    chunk_terms = set(chunk.get("terms", []))

    lexical_score = float(len(query_terms.intersection(chunk_terms)))
    path_score = 2.0 if path_prefix and p.startswith(path_prefix) else 0.0
    changed_score = 1.5 if include_changed_bias and p in changed_files else 0.0

    branch = str(chunk.get("git_branch", "") or "")
    branch_score = 0.9 if current_branch != "unknown" and branch == current_branch else 0.0

    recency_raw = _recency_boost_from_commit_ts(_safe_int(chunk.get("git_commit_ts", 0), 0), now_ts)
    recency_score = round(recency_raw * 0.8, 4)

    total = lexical_score + path_score + changed_score + branch_score + recency_score
    if total <= 0:
        return None

    return {
        "score": round(total, 4),
        "score_breakdown": {
            "lexical": round(lexical_score, 4),
            "path_prefix": round(path_score, 4),
            "changed_file": round(changed_score, 4),
            "branch_match": round(branch_score, 4),
            "recency": round(recency_score, 4),
            "recency_raw": round(recency_raw, 4),
        },
    }


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
