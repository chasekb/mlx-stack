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


SPEC_ROUTER_SERVER = """from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer


def tokenize_text(text: str) -> list[str]:
    return [tok for tok in re.split(r"\s+", (text or "").strip()) if tok]


def http_json(method: str, url: str, payload: dict | None = None, timeout: float = 20.0) -> dict:
    data = json.dumps(payload or {}).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body) if body else {}


def extract_completion_text(resp: dict) -> str:
    if not isinstance(resp, dict):
        return ""
    choices = resp.get("choices", []) if isinstance(resp.get("choices", []), list) else []
    if not choices:
        return ""
    c0 = choices[0] if isinstance(choices[0], dict) else {}
    text = c0.get("text")
    if isinstance(text, str) and text.strip():
        return text
    msg = c0.get("message") if isinstance(c0.get("message"), dict) else {}
    content = msg.get("content")
    if isinstance(content, str):
        return content
    return ""


def request_model_tokens(
    *,
    api_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> tuple[list[str], float]:
    started = time.perf_counter()
    resp = http_json(
        "POST",
        api_url.rstrip("/"),
        payload={
            "model": model,
            "prompt": prompt,
            "max_tokens": max(1, int(max_tokens)),
            "temperature": 0,
        },
        timeout=timeout,
    )
    text = extract_completion_text(resp)
    took_ms = (time.perf_counter() - started) * 1000.0
    return tokenize_text(text), took_ms


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


def run_speculative_decode(payload: dict) -> dict:
    draft_tokens = payload.get("draft_tokens", [])
    target_tokens = payload.get("target_tokens", [])

    if isinstance(draft_tokens, list) and isinstance(target_tokens, list) and (draft_tokens or target_tokens):
        result = run_speculative_loop(
            draft_tokens=[str(t) for t in draft_tokens],
            target_tokens=[str(t) for t in target_tokens],
        )
        result["source"] = "provided_tokens"
        return result

    prompt = str(payload.get("prompt", "") or "").strip()
    if not prompt:
        raise ValueError("missing_prompt_or_tokens")

    draft_model = str(payload.get("draft_model") or os.environ.get("SPEC_DRAFT_MODEL", "local-mlx-fast"))
    target_model = str(payload.get("target_model") or os.environ.get("SPEC_TARGET_MODEL", "local-mlx"))
    draft_url = str(
        payload.get("draft_url") or os.environ.get("SPEC_DRAFT_URL", "http://localhost:4000/v1/completions")
    )
    target_url = str(
        payload.get("target_url") or os.environ.get("SPEC_TARGET_URL", "http://localhost:4000/v1/completions")
    )
    max_tokens = int(payload.get("max_tokens", 128) or 128)
    timeout = float(payload.get("timeout", 20.0) or 20.0)

    draft_tokens_out: list[str] = []
    target_tokens_out: list[str] = []
    draft_ms = 0.0
    target_ms = 0.0
    draft_error = ""

    try:
        draft_tokens_out, draft_ms = request_model_tokens(
            api_url=draft_url,
            model=draft_model,
            prompt=prompt,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    except Exception as e:
        draft_error = str(e)

    target_tokens_out, target_ms = request_model_tokens(
        api_url=target_url,
        model=target_model,
        prompt=prompt,
        max_tokens=max_tokens,
        timeout=timeout,
    )

    result = run_speculative_loop(draft_tokens=draft_tokens_out, target_tokens=target_tokens_out)
    result.update(
        {
            "source": "model_calls",
            "draft_model": draft_model,
            "target_model": target_model,
            "draft_token_count": len(draft_tokens_out),
            "target_token_count": len(target_tokens_out),
            "draft_call_ms": round(draft_ms, 2),
            "target_call_ms": round(target_ms, 2),
        }
    )
    if draft_error:
        result["draft_error"] = draft_error
    return result


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

        try:
            result = run_speculative_decode(payload)
        except ValueError as e:
            self._reply({"error": str(e)}, status=400)
            return
        except urllib.error.URLError as e:
            self._reply({"error": "model_backend_unreachable", "detail": str(e)}, status=502)
            return
        except Exception as e:
            self._reply({"error": "decode_failed", "detail": str(e)}, status=500)
            return
        self._reply({"ok": True, "service": "spec-router", "result": result})


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 8092), Handler)
    print("Spec router listening on :8092")
    server.serve_forever()
"""


EMBED_QUEUE_SERVER = 'from __future__ import annotations\n\nimport json\nimport os\nimport sqlite3\nimport time\nfrom pathlib import Path\nfrom http.server import BaseHTTPRequestHandler, HTTPServer\n\n\nDB_PATH = Path(\'.ai-dev/embedding_jobs.db\')\nEVENT_LOG_PATH = Path(\'.ai-dev/events/embed-queue.jsonl\')\n\n\ndef emit_event(event_type: str, **fields: object) -> None:\n    EVENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)\n    rec = {\n        \'ts\': time.time(),\n        \'service\': \'embed-queue\',\n        \'event\': event_type,\n        **fields,\n    }\n    with EVENT_LOG_PATH.open(\'a\', encoding=\'utf-8\') as f:\n        f.write(json.dumps(rec) + \'\\n\')\n\n\ndef parse_dead_letter_threshold() -> int:\n    raw = os.environ.get(\'EMBED_QUEUE_ALERT_DEAD_LETTER\', \'5\')\n    try:\n        return max(0, int(raw))\n    except ValueError:\n        return 5\n\n\ndef compute_alerts(stats: dict, dead_letter_threshold: int) -> list[dict]:\n    alerts: list[dict] = []\n    dead_letter = int(stats.get(\'dead_letter\', 0) or 0)\n    if dead_letter >= max(0, int(dead_letter_threshold)):\n        alerts.append(\n            {\n                \'name\': \'dead_letter_threshold_exceeded\',\n                \'severity\': \'warning\',\n                \'value\': dead_letter,\n                \'threshold\': int(dead_letter_threshold),\n                \'message\': f\'dead_letter jobs ({dead_letter}) >= threshold ({dead_letter_threshold})\',\n            }\n        )\n    return alerts\n\n\ndef _db_connect() -> sqlite3.Connection:\n    DB_PATH.parent.mkdir(parents=True, exist_ok=True)\n    conn = sqlite3.connect(DB_PATH)\n    conn.row_factory = sqlite3.Row\n    return conn\n\n\ndef _init_db() -> None:\n    with _db_connect() as conn:\n        conn.execute(\n            \'\'\'\n            CREATE TABLE IF NOT EXISTS embedding_jobs (\n                id INTEGER PRIMARY KEY AUTOINCREMENT,\n                kind TEXT NOT NULL,\n                payload_json TEXT NOT NULL,\n                status TEXT NOT NULL,\n                attempts INTEGER NOT NULL DEFAULT 0,\n                max_attempts INTEGER NOT NULL DEFAULT 3,\n                next_attempt_at REAL NOT NULL DEFAULT 0,\n                last_error TEXT,\n                created_at REAL NOT NULL,\n                updated_at REAL NOT NULL\n            )\n            \'\'\'\n        )\n        conn.commit()\n\n\ndef enqueue_job(kind: str, payload: dict, max_attempts: int = 3) -> dict:\n    now = time.time()\n    with _db_connect() as conn:\n        cur = conn.execute(\n            \'\'\'\n            INSERT INTO embedding_jobs (\n                kind, payload_json, status, attempts, max_attempts, next_attempt_at, created_at, updated_at\n            ) VALUES (?, ?, \'queued\', 0, ?, 0, ?, ?)\n            \'\'\',\n            (kind, json.dumps(payload), max(1, int(max_attempts)), now, now),\n        )\n        conn.commit()\n        out = {\'job_id\': int(cur.lastrowid), \'status\': \'queued\'}\n        emit_event(\'job_enqueued\', job_id=out[\'job_id\'], kind=kind, max_attempts=max_attempts)\n        return out\n\n\ndef claim_next_job() -> dict | None:\n    now = time.time()\n    with _db_connect() as conn:\n        conn.execute(\'BEGIN IMMEDIATE\')\n        row = conn.execute(\n            \'\'\'\n            SELECT * FROM embedding_jobs\n            WHERE status IN (\'queued\', \'retry\')\n              AND next_attempt_at <= ?\n            ORDER BY id ASC\n            LIMIT 1\n            \'\'\',\n            (now,),\n        ).fetchone()\n\n        if row is None:\n            conn.commit()\n            return None\n\n        next_attempts = int(row[\'attempts\']) + 1\n        conn.execute(\n            \'\'\'\n            UPDATE embedding_jobs\n            SET status=\'in_progress\', attempts=?, updated_at=?\n            WHERE id=?\n            \'\'\',\n            (next_attempts, now, int(row[\'id\'])),\n        )\n        conn.commit()\n\n        out = {\n            \'id\': int(row[\'id\']),\n            \'kind\': row[\'kind\'],\n            \'payload\': json.loads(row[\'payload_json\']),\n            \'attempts\': next_attempts,\n            \'max_attempts\': int(row[\'max_attempts\']),\n        }\n        emit_event(\'job_claimed\', job_id=out[\'id\'], attempts=next_attempts, kind=out[\'kind\'])\n        return out\n\n\ndef complete_job(job_id: int) -> None:\n    now = time.time()\n    with _db_connect() as conn:\n        conn.execute(\n            "UPDATE embedding_jobs SET status=\'done\', updated_at=? WHERE id=?",\n            (now, int(job_id)),\n        )\n        conn.commit()\n    emit_event(\'job_completed\', job_id=int(job_id))\n\n\ndef fail_job(job_id: int, error: str) -> dict:\n    now = time.time()\n    with _db_connect() as conn:\n        row = conn.execute(\n            \'SELECT attempts, max_attempts FROM embedding_jobs WHERE id=?\',\n            (int(job_id),),\n        ).fetchone()\n        if row is None:\n            return {\'error\': \'job_not_found\'}\n\n        attempts = int(row[\'attempts\'])\n        max_attempts = int(row[\'max_attempts\'])\n        if attempts >= max_attempts:\n            status = \'dead_letter\'\n            next_attempt_at = 0\n        else:\n            status = \'retry\'\n            next_attempt_at = now + min(60.0, float(2 ** attempts))\n\n        conn.execute(\n            \'\'\'\n            UPDATE embedding_jobs\n            SET status=?, next_attempt_at=?, last_error=?, updated_at=?\n            WHERE id=?\n            \'\'\',\n            (status, next_attempt_at, str(error)[:2000], now, int(job_id)),\n        )\n        conn.commit()\n        out = {\'ok\': True, \'status\': status, \'next_attempt_at\': next_attempt_at}\n        emit_event(\n            \'job_failed\',\n            job_id=int(job_id),\n            status=status,\n            attempts=attempts,\n            max_attempts=max_attempts,\n            error=str(error)[:300],\n        )\n        return out\n\n\ndef get_stats() -> dict:\n    with _db_connect() as conn:\n        counts = {\n            row[\'status\']: int(row[\'count\'])\n            for row in conn.execute(\n                \'SELECT status, COUNT(*) AS count FROM embedding_jobs GROUP BY status\'\n            ).fetchall()\n        }\n    return {\n        \'queued\': counts.get(\'queued\', 0),\n        \'retry\': counts.get(\'retry\', 0),\n        \'in_progress\': counts.get(\'in_progress\', 0),\n        \'done\': counts.get(\'done\', 0),\n        \'dead_letter\': counts.get(\'dead_letter\', 0),\n    }\n\n\nclass Handler(BaseHTTPRequestHandler):\n    def _reply(self, payload: dict, status: int = 200) -> None:\n        self.send_response(status)\n        self.send_header(\'Content-Type\', \'application/json\')\n        self.end_headers()\n        self.wfile.write(json.dumps(payload).encode(\'utf-8\'))\n\n    def _read_json(self) -> dict:\n        try:\n            n = int(self.headers.get(\'Content-Length\', \'0\'))\n        except ValueError:\n            n = 0\n        body = self.rfile.read(max(0, n))\n        if not body:\n            return {}\n        try:\n            parsed = json.loads(body.decode(\'utf-8\'))\n            return parsed if isinstance(parsed, dict) else {}\n        except Exception:\n            return {}\n\n    def do_GET(self):\n        if self.path == \'/health\':\n            self._reply({\'ok\': True, \'service\': \'embed-queue\', \'db_path\': str(DB_PATH)})\n            return\n        if self.path == \'/stats\':\n            stats = get_stats()\n            threshold = parse_dead_letter_threshold()\n            alerts = compute_alerts(stats, dead_letter_threshold=threshold)\n            if alerts:\n                emit_event(\'alerts_emitted\', alerts=alerts)\n            self._reply(\n                {\n                    \'ok\': True,\n                    \'service\': \'embed-queue\',\n                    \'stats\': stats,\n                    \'alerts\': alerts,\n                    \'alert_thresholds\': {\'dead_letter\': threshold},\n                }\n            )\n            return\n        self._reply({\'error\': \'not found\'}, status=404)\n\n    def do_POST(self):\n        if self.path == \'/jobs/enqueue\':\n            payload = self._read_json()\n            kind = str(payload.get(\'kind\', \'file_change\') or \'file_change\')\n            job_payload = payload.get(\'payload\', {}) if isinstance(payload.get(\'payload\', {}), dict) else {}\n            max_attempts = int(payload.get(\'max_attempts\', 3) or 3)\n            out = enqueue_job(kind=kind, payload=job_payload, max_attempts=max_attempts)\n            self._reply({\'ok\': True, \'service\': \'embed-queue\', **out})\n            return\n\n        if self.path == \'/jobs/claim\':\n            job = claim_next_job()\n            self._reply({\'ok\': True, \'service\': \'embed-queue\', \'job\': job})\n            return\n\n        if self.path == \'/jobs/complete\':\n            payload = self._read_json()\n            job_id = int(payload.get(\'job_id\', 0) or 0)\n            if job_id <= 0:\n                self._reply({\'error\': \'missing_job_id\'}, status=400)\n                return\n            complete_job(job_id)\n            self._reply({\'ok\': True, \'service\': \'embed-queue\', \'job_id\': job_id, \'status\': \'done\'})\n            return\n\n        if self.path == \'/jobs/fail\':\n            payload = self._read_json()\n            job_id = int(payload.get(\'job_id\', 0) or 0)\n            if job_id <= 0:\n                self._reply({\'error\': \'missing_job_id\'}, status=400)\n                return\n            error = str(payload.get(\'error\', \'unknown_error\'))\n            out = fail_job(job_id=job_id, error=error)\n            if out.get(\'error\'):\n                self._reply(out, status=404)\n                return\n            self._reply({\'ok\': True, \'service\': \'embed-queue\', \'job_id\': job_id, **out})\n            return\n\n        self._reply({\'error\': \'not found\'}, status=404)\n\n\nif __name__ == \'__main__\':\n    _init_db()\n    server = HTTPServer((\'0.0.0.0\', 8093), Handler)\n    print(\'Embedding queue listening on :8093\')\n    server.serve_forever()\n'


EMBED_WORKER = "from __future__ import annotations\n\nimport argparse\nimport hashlib\nimport json\nimport math\nimport os\nimport shutil\nimport time\nimport urllib.error\nimport urllib.request\nfrom datetime import datetime, timezone\nfrom pathlib import Path\n\n\nEMBEDDING_SCHEMA_VERSION = 2\nEVENT_LOG_PATH = Path('.ai-dev/events/embed-worker.jsonl')\n\n\ndef emit_event(event_type: str, **fields: object) -> None:\n    EVENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)\n    rec = {\n        'ts': time.time(),\n        'service': 'embed-worker',\n        'event': event_type,\n        **fields,\n    }\n    with EVENT_LOG_PATH.open('a', encoding='utf-8') as f:\n        f.write(json.dumps(rec) + '\\n')\n\n\ndef http_json(method: str, url: str, payload: dict | None = None, timeout: float = 10.0) -> dict:\n    data = json.dumps(payload or {}).encode('utf-8') if payload is not None else None\n    req = urllib.request.Request(\n        url,\n        data=data,\n        headers={'Content-Type': 'application/json'},\n        method=method,\n    )\n    with urllib.request.urlopen(req, timeout=timeout) as resp:\n        raw = resp.read().decode('utf-8')\n    return json.loads(raw) if raw else {}\n\n\ndef fake_embed(text: str, dims: int = 16) -> list[float]:\n    digest = hashlib.sha256((text or '').encode('utf-8')).digest()\n    vals = [((digest[i % len(digest)] / 255.0) * 2.0 - 1.0) for i in range(dims)]\n    norm = math.sqrt(sum(v * v for v in vals)) or 1.0\n    return [round(v / norm, 6) for v in vals]\n\n\ndef _coerce_vector(vec: object) -> list[float]:\n    if not isinstance(vec, list):\n        return []\n    out: list[float] = []\n    for x in vec:\n        try:\n            out.append(float(x))\n        except (TypeError, ValueError):\n            return []\n    return out\n\n\ndef _extract_source_text(payload: dict) -> str:\n    source_text = str(payload.get('text', '') or '')\n    if not source_text:\n        source_text = str(payload.get('path', '') or '')\n    if not source_text:\n        source_text = json.dumps(payload, sort_keys=True)\n    return source_text\n\n\ndef _embed_via_http(text: str, embed_url: str, embed_model: str, timeout: float) -> list[float]:\n    body = {\n        'model': embed_model,\n        'input': text,\n    }\n    resp = http_json('POST', embed_url, payload=body, timeout=timeout)\n    data = resp.get('data', []) if isinstance(resp, dict) else []\n    if not isinstance(data, list) or not data:\n        raise ValueError('missing_embedding_data')\n    first = data[0] if isinstance(data[0], dict) else {}\n    vec = _coerce_vector(first.get('embedding'))\n    if not vec:\n        raise ValueError('invalid_embedding_vector')\n    return vec\n\n\ndef _load_schema(schema_path: Path) -> dict:\n    if not schema_path.exists():\n        return {}\n    try:\n        parsed = json.loads(schema_path.read_text(encoding='utf-8'))\n        return parsed if isinstance(parsed, dict) else {}\n    except Exception:\n        return {}\n\n\ndef _save_schema(schema_path: Path, schema: dict) -> None:\n    schema_path.parent.mkdir(parents=True, exist_ok=True)\n    schema_path.write_text(json.dumps(schema, indent=2) + '\\n', encoding='utf-8')\n\n\ndef _append_migration_event(migration_log_path: Path, event: dict) -> None:\n    migration_log_path.parent.mkdir(parents=True, exist_ok=True)\n    with migration_log_path.open('a', encoding='utf-8') as f:\n        f.write(json.dumps(event) + '\\n')\n\n\ndef _rotate_output_for_migration(output_path: Path, old_schema: dict, migration_log_path: Path) -> None:\n    if not output_path.exists() or not output_path.is_file():\n        return\n    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')\n    rotated = output_path.with_suffix(output_path.suffix + f'.migrated-{ts}.bak')\n    shutil.move(str(output_path), str(rotated))\n    _append_migration_event(\n        migration_log_path,\n        {\n            'event': 'embedding_schema_migration',\n            'migrated_at': datetime.now(timezone.utc).isoformat(),\n            'old_schema': old_schema,\n            'rotated_output_path': str(rotated),\n        },\n    )\n\n\ndef ensure_embedding_schema(\n    schema_path: Path,\n    output_path: Path,\n    migration_log_path: Path,\n    *,\n    embedding_model: str,\n    vector_dim: int,\n    backend: str,\n    allow_migrate: bool,\n) -> dict:\n    now_iso = datetime.now(timezone.utc).isoformat()\n    expected = {\n        'schema_version': EMBEDDING_SCHEMA_VERSION,\n        'embedding_model': embedding_model,\n        'vector_dim': int(vector_dim),\n        'backend': backend,\n    }\n    current = _load_schema(schema_path)\n    if not current:\n        out = {**expected, 'created_at': now_iso, 'updated_at': now_iso}\n        _save_schema(schema_path, out)\n        return out\n\n    compatible = (\n        int(current.get('schema_version', 0) or 0) == expected['schema_version']\n        and str(current.get('embedding_model', '')) == expected['embedding_model']\n        and int(current.get('vector_dim', 0) or 0) == expected['vector_dim']\n        and str(current.get('backend', '')) == expected['backend']\n    )\n    if compatible:\n        current['updated_at'] = now_iso\n        _save_schema(schema_path, current)\n        return current\n\n    if not allow_migrate:\n        raise RuntimeError(\n            'embedding_schema_mismatch: pass --allow-schema-migrate to rotate old embeddings and continue'\n        )\n\n    _rotate_output_for_migration(output_path=output_path, old_schema=current, migration_log_path=migration_log_path)\n    out = {**expected, 'created_at': now_iso, 'updated_at': now_iso}\n    _save_schema(schema_path, out)\n    _append_migration_event(\n        migration_log_path,\n        {\n            'event': 'embedding_schema_initialized',\n            'initialized_at': now_iso,\n            'schema': out,\n        },\n    )\n    return out\n\n\ndef qdrant_upsert(\n    *,\n    qdrant_url: str,\n    collection: str,\n    point_id: int,\n    vector: list[float],\n    payload: dict,\n    timeout: float,\n) -> dict:\n    base = qdrant_url.rstrip('/')\n    coll_url = f'{base}/collections/{collection}'\n    try:\n        http_json('GET', coll_url, timeout=timeout)\n    except Exception:\n        http_json(\n            'PUT',\n            coll_url,\n            payload={'vectors': {'size': len(vector), 'distance': 'Cosine'}},\n            timeout=timeout,\n        )\n\n    return http_json(\n        'PUT',\n        f'{coll_url}/points?wait=true',\n        payload={\n            'points': [\n                {\n                    'id': int(point_id),\n                    'vector': vector,\n                    'payload': payload,\n                }\n            ]\n        },\n        timeout=timeout,\n    )\n\n\ndef process_job(\n    job: dict,\n    output_path: Path,\n    schema_path: Path,\n    migration_log_path: Path,\n    *,\n    embed_url: str,\n    embed_model: str,\n    timeout: float,\n    qdrant_url: str,\n    qdrant_collection: str,\n    qdrant_enabled: bool,\n    allow_schema_migrate: bool,\n    force_fake_embed: bool,\n) -> None:\n    emit_event('job_processing_started', job_id=int(job.get('id', 0) or 0), kind=str(job.get('kind', 'unknown')))\n    payload = job.get('payload', {}) if isinstance(job.get('payload', {}), dict) else {}\n    source_text = _extract_source_text(payload)\n\n    vector_backend = 'local_http'\n    if force_fake_embed:\n        vector = fake_embed(source_text)\n        vector_backend = 'deterministic_fallback'\n    else:\n        try:\n            vector = _embed_via_http(source_text, embed_url=embed_url, embed_model=embed_model, timeout=timeout)\n        except Exception:\n            vector = fake_embed(source_text)\n            vector_backend = 'deterministic_fallback'\n\n    schema = ensure_embedding_schema(\n        schema_path=schema_path,\n        output_path=output_path,\n        migration_log_path=migration_log_path,\n        embedding_model=embed_model,\n        vector_dim=len(vector),\n        backend=vector_backend,\n        allow_migrate=allow_schema_migrate,\n    )\n\n    qdrant_status = {'enabled': qdrant_enabled, 'upserted': False}\n    if qdrant_enabled:\n        try:\n            qdrant_upsert(\n                qdrant_url=qdrant_url,\n                collection=qdrant_collection,\n                point_id=int(job.get('id', 0) or 0),\n                vector=vector,\n                payload={\n                    'kind': str(job.get('kind', 'unknown')),\n                    'metadata': payload,\n                    'schema_version': EMBEDDING_SCHEMA_VERSION,\n                    'embedding_model': embed_model,\n                    'vector_backend': vector_backend,\n                },\n                timeout=timeout,\n            )\n            qdrant_status = {'enabled': True, 'upserted': True}\n            emit_event('qdrant_upsert_succeeded', job_id=int(job.get('id', 0) or 0), collection=qdrant_collection)\n        except Exception as e:\n            qdrant_status = {'enabled': True, 'upserted': False, 'error': str(e)[:500]}\n            emit_event('qdrant_upsert_failed', job_id=int(job.get('id', 0) or 0), error=str(e)[:300])\n\n    rec = {\n        'embedded_at': datetime.now(timezone.utc).isoformat(),\n        'schema_version': EMBEDDING_SCHEMA_VERSION,\n        'job_id': int(job.get('id', 0) or 0),\n        'kind': str(job.get('kind', 'unknown')),\n        'embedding_model': embed_model,\n        'vector_backend': vector_backend,\n        'vector_dim': len(vector),\n        'schema': {\n            'embedding_model': schema.get('embedding_model', ''),\n            'vector_dim': int(schema.get('vector_dim', 0) or 0),\n            'backend': schema.get('backend', ''),\n        },\n        'qdrant': qdrant_status,\n        'metadata': payload,\n        'vector': vector,\n    }\n\n    output_path.parent.mkdir(parents=True, exist_ok=True)\n    with output_path.open('a', encoding='utf-8') as f:\n        f.write(json.dumps(rec) + '\\n')\n    emit_event(\n        'job_processing_completed',\n        job_id=int(job.get('id', 0) or 0),\n        vector_backend=vector_backend,\n        vector_dim=len(vector),\n        qdrant_upserted=bool(qdrant_status.get('upserted', False)),\n    )\n\n\ndef run_once(queue_url: str, output_path: Path, timeout: float) -> bool:\n    claim = http_json('POST', queue_url.rstrip('/') + '/jobs/claim', payload={}, timeout=timeout)\n    job = claim.get('job') if isinstance(claim, dict) else None\n    if not isinstance(job, dict):\n        return False\n\n    job_id = int(job.get('id', 0) or 0)\n    if job_id <= 0:\n        return False\n\n    try:\n        process_job(\n            job=job,\n            output_path=output_path,\n            schema_path=Path(os.environ.get('EMBED_SCHEMA_PATH', '.ai-dev/embedding_schema.json')),\n            migration_log_path=Path(os.environ.get('EMBED_MIGRATION_LOG_PATH', '.ai-dev/embedding_migrations.jsonl')),\n            embed_url=os.environ.get('EMBED_URL', 'http://localhost:4000/v1/embeddings'),\n            embed_model=os.environ.get('EMBED_MODEL', 'local-embed'),\n            timeout=timeout,\n            qdrant_url=os.environ.get('QDRANT_URL', 'http://localhost:6333'),\n            qdrant_collection=os.environ.get('QDRANT_COLLECTION', 'ai_dev_embeddings'),\n            qdrant_enabled=os.environ.get('QDRANT_ENABLED', '1') not in ('0', 'false', 'False'),\n            allow_schema_migrate=os.environ.get('ALLOW_SCHEMA_MIGRATE', '0') in ('1', 'true', 'True'),\n            force_fake_embed=os.environ.get('FORCE_FAKE_EMBED', '0') in ('1', 'true', 'True'),\n        )\n    except Exception as e:\n        emit_event('job_processing_failed', job_id=job_id, error=str(e)[:300])\n        http_json(\n            'POST',\n            queue_url.rstrip('/') + '/jobs/fail',\n            payload={'job_id': job_id, 'error': str(e)},\n            timeout=timeout,\n        )\n        return True\n\n    http_json(\n        'POST',\n        queue_url.rstrip('/') + '/jobs/complete',\n        payload={'job_id': job_id},\n        timeout=timeout,\n    )\n    emit_event('job_marked_done', job_id=job_id)\n    return True\n\n\ndef main() -> int:\n    p = argparse.ArgumentParser(description='Background embedding worker')\n    p.add_argument('--queue-url', default='http://localhost:8093')\n    p.add_argument('--output-path', default='.ai-dev/embeddings.jsonl')\n    p.add_argument('--poll-interval', type=float, default=2.0)\n    p.add_argument('--timeout', type=float, default=10.0)\n    p.add_argument('--embed-url', default='http://localhost:4000/v1/embeddings')\n    p.add_argument('--embed-model', default='local-embed')\n    p.add_argument('--schema-path', default='.ai-dev/embedding_schema.json')\n    p.add_argument('--migration-log-path', default='.ai-dev/embedding_migrations.jsonl')\n    p.add_argument('--qdrant-url', default='http://localhost:6333')\n    p.add_argument('--qdrant-collection', default='ai_dev_embeddings')\n    p.add_argument('--disable-qdrant', action='store_true')\n    p.add_argument('--allow-schema-migrate', action='store_true')\n    p.add_argument('--force-fake-embed', action='store_true')\n    p.add_argument('--once', action='store_true', help='Process at most one available job and exit')\n    args = p.parse_args()\n\n    os.environ['EMBED_URL'] = str(args.embed_url)\n    os.environ['EMBED_MODEL'] = str(args.embed_model)\n    os.environ['EMBED_SCHEMA_PATH'] = str(args.schema_path)\n    os.environ['EMBED_MIGRATION_LOG_PATH'] = str(args.migration_log_path)\n    os.environ['QDRANT_URL'] = str(args.qdrant_url)\n    os.environ['QDRANT_COLLECTION'] = str(args.qdrant_collection)\n    os.environ['QDRANT_ENABLED'] = '0' if args.disable_qdrant else '1'\n    os.environ['ALLOW_SCHEMA_MIGRATE'] = '1' if args.allow_schema_migrate else '0'\n    os.environ['FORCE_FAKE_EMBED'] = '1' if args.force_fake_embed else '0'\n\n    output_path = Path(args.output_path)\n\n    if args.once:\n        try:\n            run_once(queue_url=args.queue_url, output_path=output_path, timeout=args.timeout)\n            return 0\n        except urllib.error.URLError as e:\n            print(f'worker failed to reach queue: {e}')\n            return 2\n\n    print(f'Embedding worker polling {args.queue_url} every {args.poll_interval}s')\n    while True:\n        try:\n            processed = run_once(queue_url=args.queue_url, output_path=output_path, timeout=args.timeout)\n        except urllib.error.URLError as e:\n            print(f'worker queue error: {e}')\n            processed = False\n\n        if not processed:\n            time.sleep(max(0.25, args.poll_interval))\n\n\nif __name__ == '__main__':\n    raise SystemExit(main())\n"


AGENT_SERVER = 'import json\nimport re\nimport subprocess\nimport uuid\nimport hashlib\nimport time\nfrom datetime import datetime, timezone\nfrom pathlib import Path\nfrom typing import Optional\nfrom urllib.parse import parse_qs, urlparse\nfrom http.server import BaseHTTPRequestHandler, HTTPServer\n\n\nROOT = Path(__file__).resolve().parents[1]\nINDEX_PATH = ROOT / ".ai-dev" / "index.json"\nRUNS_DIR = ROOT / ".ai-dev" / "runs"\nCACHE_PATH = ROOT / ".ai-dev" / "prompt_cache.json"\nMETRICS_PATH = ROOT / ".ai-dev" / "metrics.json"\nKV_CACHE_PATH = ROOT / ".ai-dev" / "kv_cache.json"\nEVENT_LOG_PATH = ROOT / ".ai-dev" / "events" / "agent.jsonl"\nDEFAULT_CACHE_TTL_SECONDS = 600\nDEFAULT_KV_MODEL_BUDGET_TOKENS = 8000\nDEFAULT_KV_ENTRY_MAX_TOKENS = 2048\nALLOWED_TOOLS = {\n    "retrieve",\n    "search_code",\n    "read_file",\n    "git_diff",\n    "run_tests",\n    "write_patch",\n    "commit_changes",\n}\n\nTOOL_SCHEMAS = {\n    "retrieve": {\n        "description": "Retrieve relevant symbols/chunks from local index",\n        "input": {"query": "string", "top_k": "int?", "path_prefix": "string?"},\n    },\n    "search_code": {\n        "description": "Regex search across repository files",\n        "input": {"regex": "string", "file_pattern": "string?", "limit": "int?"},\n    },\n    "read_file": {\n        "description": "Read a file from repo",\n        "input": {"path": "string", "max_chars": "int?"},\n    },\n    "git_diff": {\n        "description": "Get current git diff summary",\n        "input": {},\n    },\n    "run_tests": {\n        "description": "Run tests in dry-run or execute mode",\n        "input": {"command": "string?"},\n    },\n    "write_patch": {\n        "description": "Apply patch to repo (blocked in dry-run)",\n        "input": {"patch": "string"},\n    },\n    "commit_changes": {\n        "description": "Commit current changes (blocked in dry-run)",\n        "input": {"message": "string"},\n    },\n}\n\nPATCH_DENY_PREFIXES = (\n    ".git/",\n)\n\n\ndef emit_event(event_type: str, **fields: object) -> None:\n    EVENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)\n    rec = {\n        "ts": time.time(),\n        "service": "agent",\n        "event": event_type,\n        **fields,\n    }\n    with EVENT_LOG_PATH.open("a", encoding="utf-8") as f:\n        f.write(json.dumps(rec) + "\\n")\n\n\ndef parse_alert_thresholds() -> dict:\n    raw_errors = "5"\n    raw_hit_rate = "0.2"\n    try:\n        import os\n\n        raw_errors = os.environ.get("AGENT_ALERT_TOOL_ERRORS", "5")\n        raw_hit_rate = os.environ.get("AGENT_ALERT_CACHE_HIT_RATE_MIN", "0.2")\n    except Exception:\n        pass\n    try:\n        max_tool_errors = max(0, int(raw_errors))\n    except ValueError:\n        max_tool_errors = 5\n    try:\n        min_cache_hit_rate = max(0.0, min(1.0, float(raw_hit_rate)))\n    except ValueError:\n        min_cache_hit_rate = 0.2\n    return {\n        "max_tool_errors": max_tool_errors,\n        "min_cache_hit_rate": min_cache_hit_rate,\n    }\n\n\ndef compute_alerts(metrics: dict, thresholds: dict) -> list[dict]:\n    alerts: list[dict] = []\n    tools = metrics.get("tools", {}) if isinstance(metrics.get("tools", {}), dict) else {}\n    total_errors = sum(int(v.get("errors", 0) or 0) for v in tools.values() if isinstance(v, dict))\n    max_tool_errors = int(thresholds.get("max_tool_errors", 5) or 5)\n    if total_errors >= max_tool_errors:\n        alerts.append(\n            {\n                "name": "tool_errors_threshold_exceeded",\n                "severity": "warning",\n                "value": total_errors,\n                "threshold": max_tool_errors,\n                "message": f"tool errors ({total_errors}) >= threshold ({max_tool_errors})",\n            }\n        )\n\n    cache = metrics.get("cache", {}) if isinstance(metrics.get("cache", {}), dict) else {}\n    requests = int(cache.get("requests", 0) or 0)\n    hit_rate = float(cache.get("hit_rate", 0.0) or 0.0)\n    min_cache_hit_rate = float(thresholds.get("min_cache_hit_rate", 0.2) or 0.2)\n    if requests >= 10 and hit_rate < min_cache_hit_rate:\n        alerts.append(\n            {\n                "name": "cache_hit_rate_below_minimum",\n                "severity": "warning",\n                "value": round(hit_rate, 4),\n                "threshold": round(min_cache_hit_rate, 4),\n                "message": f"cache hit_rate ({hit_rate:.4f}) < threshold ({min_cache_hit_rate:.4f})",\n            }\n        )\n    return alerts\n\n\ndef utc_now_iso() -> str:\n    return datetime.now(timezone.utc).isoformat()\n\n\ndef load_json_file(path: Path, default):\n    if not path.exists():\n        return default\n    try:\n        return json.loads(path.read_text(encoding="utf-8"))\n    except Exception:\n        return default\n\n\ndef save_json_file(path: Path, payload: dict) -> None:\n    path.parent.mkdir(parents=True, exist_ok=True)\n    path.write_text(json.dumps(payload, indent=2) + "\\n", encoding="utf-8")\n\n\ndef get_git_branch() -> str:\n    proc = subprocess.run(\n        ["git", "rev-parse", "--abbrev-ref", "HEAD"],\n        cwd=ROOT,\n        capture_output=True,\n        text=True,\n    )\n    if proc.returncode != 0:\n        return "unknown"\n    return (proc.stdout or "").strip() or "unknown"\n\n\ndef get_index_signature() -> str:\n    if not INDEX_PATH.exists():\n        return "no-index"\n    try:\n        index_obj = json.loads(INDEX_PATH.read_text(encoding="utf-8"))\n    except Exception:\n        return "index-unreadable"\n    generated_at = str(index_obj.get("generated_at", "unknown"))\n    schema_version = str(index_obj.get("schema_version", "?"))\n    file_count = str(index_obj.get("file_count", "?"))\n    return f"sv{schema_version}:{generated_at}:{file_count}"\n\n\ndef compute_cache_namespace() -> str:\n    return f"branch={get_git_branch()}|index={get_index_signature()}"\n\n\ndef normalize_task_payload(payload: dict) -> dict:\n    return {\n        "task": str(payload.get("task", "")).strip(),\n        "model": payload.get("model"),\n        "dry_run": bool(payload.get("dry_run", True)),\n        "max_steps": int(payload.get("max_steps", 6)),\n        "plan": payload.get("plan", []),\n        "tool_context_hash": payload.get("tool_context_hash"),\n        "options": payload.get("options", {}),\n    }\n\n\ndef compute_cache_key(payload: dict) -> str:\n    canonical = json.dumps(normalize_task_payload(payload), sort_keys=True, separators=(",", ":"))\n    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()\n\n\ndef load_cache() -> dict:\n    cache = load_json_file(CACHE_PATH, {"schema_version": 1, "updated_at": utc_now_iso(), "entries": {}})\n    if not isinstance(cache, dict):\n        return {"schema_version": 1, "updated_at": utc_now_iso(), "entries": {}}\n    if not isinstance(cache.get("entries"), dict):\n        cache["entries"] = {}\n    return cache\n\n\ndef save_cache(cache_obj: dict) -> None:\n    cache_obj["updated_at"] = utc_now_iso()\n    save_json_file(CACHE_PATH, cache_obj)\n\n\ndef get_cache_entry(cache_obj: dict, key: str, namespace: str) -> Optional[dict]:\n    entry = cache_obj.get("entries", {}).get(key)\n    if not isinstance(entry, dict):\n        return None\n    if entry.get("namespace") != namespace:\n        return None\n    expires_at = float(entry.get("expires_at_epoch", 0.0) or 0.0)\n    now = time.time()\n    if expires_at and now > expires_at:\n        cache_obj.get("entries", {}).pop(key, None)\n        return None\n    return entry\n\n\ndef set_cache_entry(cache_obj: dict, key: str, namespace: str, result: dict, ttl_seconds: int) -> None:\n    now = time.time()\n    cache_obj.setdefault("entries", {})[key] = {\n        "namespace": namespace,\n        "created_at": utc_now_iso(),\n        "created_at_epoch": now,\n        "expires_at_epoch": now + max(1, ttl_seconds),\n        "result": result,\n    }\n\n\ndef load_metrics() -> dict:\n    metrics = load_json_file(\n        METRICS_PATH,\n        {\n            "schema_version": 1,\n            "updated_at": utc_now_iso(),\n            "cache": {\n                "requests": 0,\n                "hits": 0,\n                "misses": 0,\n                "hit_rate": 0.0,\n                "saved_calls": 0,\n                "compute_ms_total": 0.0,\n                "avg_compute_ms": 0.0,\n            },\n        },\n    )\n    if not isinstance(metrics, dict):\n        return {"schema_version": 1, "updated_at": utc_now_iso(), "cache": {}}\n    metrics.setdefault("cache", {})\n    return metrics\n\n\ndef record_cache_metrics(hit: bool, compute_ms: float, namespace: str, key: str) -> None:\n    metrics = load_metrics()\n    cache_metrics = metrics.setdefault("cache", {})\n    requests = int(cache_metrics.get("requests", 0)) + 1\n    hits = int(cache_metrics.get("hits", 0)) + (1 if hit else 0)\n    misses = int(cache_metrics.get("misses", 0)) + (0 if hit else 1)\n    saved_calls = int(cache_metrics.get("saved_calls", 0)) + (1 if hit else 0)\n    compute_ms_total = float(cache_metrics.get("compute_ms_total", 0.0)) + max(0.0, compute_ms)\n\n    cache_metrics.update(\n        {\n            "requests": requests,\n            "hits": hits,\n            "misses": misses,\n            "hit_rate": round(hits / requests, 4) if requests else 0.0,\n            "saved_calls": saved_calls,\n            "compute_ms_total": round(compute_ms_total, 4),\n            "avg_compute_ms": round(compute_ms_total / misses, 4) if misses else 0.0,\n            "last_namespace": namespace,\n            "last_key": key,\n            "last_status": "hit" if hit else "miss",\n            "last_updated": utc_now_iso(),\n        }\n    )\n    metrics["updated_at"] = utc_now_iso()\n    save_json_file(METRICS_PATH, metrics)\n\n\ndef record_tool_metrics(tool: str, ok: bool, duration_ms: float, error: str = "") -> None:\n    metrics = load_metrics()\n    tools = metrics.setdefault("tools", {})\n    row = tools.setdefault(\n        tool,\n        {\n            "calls": 0,\n            "ok": 0,\n            "errors": 0,\n            "duration_ms_total": 0.0,\n            "avg_duration_ms": 0.0,\n            "last_error": "",\n            "last_updated": utc_now_iso(),\n        },\n    )\n\n    row["calls"] = int(row.get("calls", 0)) + 1\n    row["ok"] = int(row.get("ok", 0)) + (1 if ok else 0)\n    row["errors"] = int(row.get("errors", 0)) + (0 if ok else 1)\n    row["duration_ms_total"] = round(float(row.get("duration_ms_total", 0.0)) + max(0.0, duration_ms), 4)\n    row["avg_duration_ms"] = round(row["duration_ms_total"] / max(1, row["calls"]), 4)\n    row["last_updated"] = utc_now_iso()\n    if not ok and error:\n        row["last_error"] = str(error)[:2000]\n\n    metrics["updated_at"] = utc_now_iso()\n    save_json_file(METRICS_PATH, metrics)\n\n\ndef estimate_tokens(text: str) -> int:\n    return max(1, len([t for t in re.split(r"\\s+", (text or "").strip()) if t]))\n\n\ndef normalize_prefix_text(payload: dict) -> str:\n    raw = payload.get("prefix")\n    if raw is None:\n        raw = payload.get("prompt_prefix")\n    if raw is None:\n        raw = ""\n    return str(raw).strip()\n\n\ndef hash_prefix(prefix: str) -> str:\n    return hashlib.sha256((prefix or "").encode("utf-8")).hexdigest()\n\n\ndef load_kv_cache() -> dict:\n    obj = load_json_file(KV_CACHE_PATH, {"schema_version": 1, "updated_at": utc_now_iso(), "models": {}})\n    if not isinstance(obj, dict):\n        obj = {"schema_version": 1, "updated_at": utc_now_iso(), "models": {}}\n    if not isinstance(obj.get("models"), dict):\n        obj["models"] = {}\n    return obj\n\n\ndef save_kv_cache(obj: dict) -> None:\n    obj["updated_at"] = utc_now_iso()\n    save_json_file(KV_CACHE_PATH, obj)\n\n\ndef _ensure_model_store(kv_obj: dict, model: str, budget_tokens: int) -> dict:\n    models = kv_obj.setdefault("models", {})\n    store = models.get(model)\n    if not isinstance(store, dict):\n        store = {\n            "budget_tokens": max(256, int(budget_tokens)),\n            "entries": {},\n            "updated_at": utc_now_iso(),\n        }\n        models[model] = store\n    store["budget_tokens"] = max(256, int(budget_tokens))\n    if not isinstance(store.get("entries"), dict):\n        store["entries"] = {}\n    return store\n\n\ndef _enforce_model_budget(store: dict, keep_key: str) -> list[str]:\n    entries = store.get("entries", {}) if isinstance(store.get("entries", {}), dict) else {}\n    budget = max(256, int(store.get("budget_tokens", DEFAULT_KV_MODEL_BUDGET_TOKENS)))\n    evicted: list[str] = []\n\n    def used_tokens() -> int:\n        return sum(int(v.get("token_estimate", 0) or 0) for v in entries.values() if isinstance(v, dict))\n\n    while used_tokens() > budget and entries:\n        candidates = [(k, v) for k, v in entries.items() if isinstance(v, dict) and k != keep_key]\n        if not candidates:\n            break\n        candidates.sort(key=lambda kv: float(kv[1].get("updated_at_epoch", 0.0) or 0.0))\n        drop_key = candidates[0][0]\n        entries.pop(drop_key, None)\n        evicted.append(drop_key)\n\n    store["entries"] = entries\n    store["used_tokens"] = used_tokens()\n    store["updated_at"] = utc_now_iso()\n    return evicted\n\n\ndef get_kv_reuse_status(payload: dict) -> dict:\n    kv_cfg = payload.get("kv_cache", {}) if isinstance(payload.get("kv_cache", {}), dict) else {}\n    enabled = bool(kv_cfg.get("enabled", True))\n    if not enabled:\n        return {"enabled": False, "status": "disabled"}\n\n    tenant_id = str(kv_cfg.get("tenant_id", "default")).strip() or "default"\n    session_id = str(kv_cfg.get("session_id", "")).strip()\n    if not session_id:\n        return {"enabled": True, "status": "bypass", "reason": "missing_session_id", "tenant_id": tenant_id}\n\n    model = str(payload.get("model") or kv_cfg.get("model") or "default").strip() or "default"\n    prefix = normalize_prefix_text(kv_cfg)\n    if not prefix:\n        return {\n            "enabled": True,\n            "status": "bypass",\n            "reason": "missing_prefix",\n            "tenant_id": tenant_id,\n            "session_id": session_id,\n            "model": model,\n        }\n\n    provided_hash = str(kv_cfg.get("prefix_hash", "")).strip()\n    computed_hash = hash_prefix(prefix)\n    if provided_hash and provided_hash != computed_hash:\n        return {\n            "enabled": True,\n            "status": "rejected",\n            "reason": "prefix_hash_mismatch",\n            "tenant_id": tenant_id,\n            "session_id": session_id,\n            "model": model,\n        }\n\n    max_entry_tokens = max(1, int(kv_cfg.get("entry_max_tokens", DEFAULT_KV_ENTRY_MAX_TOKENS)))\n    token_estimate = min(estimate_tokens(prefix), max_entry_tokens)\n    model_budget_tokens = max(256, int(kv_cfg.get("model_budget_tokens", DEFAULT_KV_MODEL_BUDGET_TOKENS)))\n\n    kv_obj = load_kv_cache()\n    model_store = _ensure_model_store(kv_obj, model=model, budget_tokens=model_budget_tokens)\n    entries = model_store.get("entries", {})\n    session_key = f"{tenant_id}|{session_id}|{model}"\n    prev = entries.get(session_key) if isinstance(entries, dict) else None\n\n    reused = False\n    reuse_reason = "cold_start"\n    reused_tokens = 0\n    if isinstance(prev, dict):\n        prev_prefix = str(prev.get("prefix", ""))\n        if prefix.startswith(prev_prefix):\n            reused = True\n            reuse_reason = "prefix_extension"\n            reused_tokens = min(int(prev.get("token_estimate", 0) or 0), token_estimate)\n        else:\n            reuse_reason = "prefix_boundary_mismatch"\n\n    now_epoch = time.time()\n    entries[session_key] = {\n        "tenant_id": tenant_id,\n        "session_id": session_id,\n        "model": model,\n        "prefix": prefix,\n        "prefix_hash": computed_hash,\n        "prefix_chars": len(prefix),\n        "token_estimate": token_estimate,\n        "updated_at": utc_now_iso(),\n        "updated_at_epoch": now_epoch,\n    }\n    model_store["entries"] = entries\n    evicted = _enforce_model_budget(model_store, keep_key=session_key)\n    save_kv_cache(kv_obj)\n\n    return {\n        "enabled": True,\n        "status": "hit" if reused else "miss",\n        "reason": reuse_reason,\n        "tenant_id": tenant_id,\n        "session_id": session_id,\n        "model": model,\n        "session_key": session_key,\n        "prefix_hash": computed_hash,\n        "token_estimate": token_estimate,\n        "reused_tokens": reused_tokens,\n        "model_budget_tokens": int(model_store.get("budget_tokens", model_budget_tokens)),\n        "model_used_tokens": int(model_store.get("used_tokens", 0)),\n        "evicted_entries": evicted,\n    }\n\n\ndef tokenize(text: str) -> list[str]:\n    return [tok for tok in re.split(r"[^a-zA-Z0-9_]+", text.lower()) if len(tok) >= 2]\n\n\ndef retrieve(index_obj: dict, query: str, top_k: int = 5, path_prefix: Optional[str] = None) -> dict:\n    query_terms = set(tokenize(query))\n    if not query_terms:\n        return {"query": query, "top_symbols": [], "top_chunks": []}\n\n    path_prefix = path_prefix or ""\n\n    symbol_results = []\n    for s in index_obj.get("symbols", []):\n        score = 0.0\n        name_terms = set(tokenize(s.get("name", "")))\n        score += len(query_terms.intersection(name_terms)) * 3\n        score += 1 if any(t in s.get("name", "").lower() for t in query_terms) else 0\n        p = s.get("path", "")\n        if path_prefix and p.startswith(path_prefix):\n            score += 1.5\n        if score > 0:\n            symbol_results.append({"score": score, **s})\n\n    chunk_results = []\n    for c in index_obj.get("chunks", []):\n        score = 0.0\n        chunk_terms = set(c.get("terms", []))\n        score += len(query_terms.intersection(chunk_terms))\n        p = c.get("path", "")\n        if path_prefix and p.startswith(path_prefix):\n            score += 2.0\n        if score > 0:\n            chunk_results.append(\n                {\n                    "score": score,\n                    "path": p,\n                    "chunk_id": c.get("chunk_id"),\n                    "start_line": c.get("start_line"),\n                    "end_line": c.get("end_line"),\n                    "text_preview": c.get("text_preview", ""),\n                }\n            )\n\n    symbol_results.sort(key=lambda x: x["score"], reverse=True)\n    chunk_results.sort(key=lambda x: x["score"], reverse=True)\n\n    return {\n        "query": query,\n        "top_symbols": symbol_results[:top_k],\n        "top_chunks": chunk_results[:top_k],\n    }\n\n\ndef ensure_under_root(path: Path) -> bool:\n    try:\n        path.resolve().relative_to(ROOT.resolve())\n        return True\n    except Exception:\n        return False\n\n\ndef normalize_patch_path(path_text: str) -> Optional[str]:\n    rel = str(path_text or "").strip().strip(\'"\').strip("\'")\n    if not rel or rel == "/dev/null":\n        return None\n    rel = rel.replace("\\\\", "/")\n    if rel.startswith("a/") or rel.startswith("b/"):\n        rel = rel[2:]\n    while rel.startswith("./"):\n        rel = rel[2:]\n    if not rel:\n        return None\n    return rel\n\n\ndef extract_patch_paths(patch_text: str) -> list[str]:\n    seen: dict[str, bool] = {}\n    lines = patch_text.splitlines()\n\n    for line in lines:\n        rel = None\n        if line.startswith("diff --git "):\n            m = re.match(r"^diff --git a/(.+?) b/(.+?)$", line)\n            if m:\n                rel = normalize_patch_path(m.group(2))\n        elif line.startswith("+++ "):\n            rel = normalize_patch_path(line[4:])\n        elif line.startswith("*** Add File:") or line.startswith("*** Update File:") or line.startswith("*** Delete File:"):\n            m = re.match(r"^\\*\\*\\* (?:Add|Update|Delete) File: (.+?)(?:\\s+->.+)?$", line)\n            if m:\n                rel = normalize_patch_path(m.group(1))\n\n        if rel:\n            seen[rel] = True\n\n    return sorted(seen.keys())\n\n\ndef path_allowed_for_patch(rel_path: str) -> bool:\n    if not rel_path:\n        return False\n    p = Path(rel_path)\n    if p.is_absolute() or ".." in p.parts:\n        return False\n    normalized = rel_path.replace("\\\\", "/")\n    for deny_prefix in PATCH_DENY_PREFIXES:\n        if normalized.startswith(deny_prefix):\n            return False\n    target = (ROOT / p).resolve()\n    return ensure_under_root(target)\n\n\ndef snapshot_paths(rel_paths: list[str]) -> dict:\n    snapshot = {}\n    for rel in rel_paths:\n        target = (ROOT / rel).resolve()\n        if target.exists() and target.is_dir():\n            raise ValueError(f"target_is_directory:{rel}")\n        if target.exists() and target.is_file():\n            snapshot[rel] = {"exists": True, "content": target.read_text(encoding="utf-8", errors="ignore")}\n        else:\n            snapshot[rel] = {"exists": False, "content": ""}\n    return snapshot\n\n\ndef restore_snapshot(snapshot: dict) -> None:\n    for rel, prior in snapshot.items():\n        target = (ROOT / rel).resolve()\n        if not ensure_under_root(target):\n            continue\n        if bool(prior.get("exists", False)):\n            target.parent.mkdir(parents=True, exist_ok=True)\n            target.write_text(str(prior.get("content", "")), encoding="utf-8")\n        else:\n            if target.exists() and target.is_file():\n                target.unlink()\n\n\ndef tool_search_code(args: dict) -> dict:\n    regex = str(args.get("regex", "")).strip()\n    if not regex:\n        return {"error": "missing_regex"}\n    file_pattern = str(args.get("file_pattern", "*") or "*")\n    limit = int(args.get("limit", 50))\n    cmd = ["bash", "-lc", f"grep -RInE --include=\'{file_pattern}\' {json.dumps(regex)} {json.dumps(str(ROOT))} | head -n {max(1, min(limit, 200))}"]\n    proc = subprocess.run(cmd, capture_output=True, text=True)\n    return {"ok": proc.returncode in (0, 1), "output": proc.stdout.strip(), "stderr": proc.stderr.strip()}\n\n\ndef tool_read_file(args: dict) -> dict:\n    rel = str(args.get("path", "")).strip()\n    if not rel:\n        return {"error": "missing_path"}\n    target = (ROOT / rel).resolve()\n    if not target.exists() or not target.is_file() or not ensure_under_root(target):\n        return {"error": "invalid_path"}\n    max_chars = int(args.get("max_chars", 12000))\n    content = target.read_text(encoding="utf-8", errors="ignore")[: max(1, max_chars)]\n    return {"ok": True, "path": rel, "content": content}\n\n\ndef tool_git_diff(_: dict) -> dict:\n    proc = subprocess.run(["git", "--no-pager", "diff", "--stat"], cwd=ROOT, capture_output=True, text=True)\n    return {"ok": proc.returncode == 0, "output": proc.stdout.strip(), "stderr": proc.stderr.strip()}\n\n\ndef tool_run_tests(args: dict, dry_run: bool) -> dict:\n    command = str(args.get("command", "python3 -m pytest -q") or "python3 -m pytest -q")\n    if dry_run:\n        return {"ok": True, "dry_run": True, "command": command}\n    proc = subprocess.run(["bash", "-lc", command], cwd=ROOT, capture_output=True, text=True)\n    return {\n        "ok": proc.returncode == 0,\n        "returncode": proc.returncode,\n        "stdout": proc.stdout[-8000:],\n        "stderr": proc.stderr[-4000:],\n    }\n\n\ndef tool_write_patch(args: dict, dry_run: bool) -> dict:\n    if dry_run:\n        return {"ok": False, "error": "blocked_in_dry_run"}\n\n    patch = str(args.get("patch", "") or "")\n    if not patch.strip():\n        return {"ok": False, "error": "missing_patch"}\n    if len(patch) > 500_000:\n        return {"ok": False, "error": "patch_too_large", "max_chars": 500000}\n\n    rel_paths = extract_patch_paths(patch)\n    if not rel_paths:\n        return {"ok": False, "error": "no_target_files_detected"}\n\n    denied = [p for p in rel_paths if not path_allowed_for_patch(p)]\n    if denied:\n        return {"ok": False, "error": "patch_target_denied", "denied_paths": denied}\n\n    try:\n        before_state = snapshot_paths(rel_paths)\n    except ValueError as exc:\n        return {"ok": False, "error": "invalid_patch_target", "detail": str(exc)}\n\n    preflight = subprocess.run(\n        ["git", "apply", "--check", "--whitespace=nowarn", "-"],\n        cwd=ROOT,\n        input=patch,\n        text=True,\n        capture_output=True,\n    )\n    if preflight.returncode != 0:\n        return {\n            "ok": False,\n            "error": "preflight_failed",\n            "stderr": (preflight.stderr or "").strip(),\n            "stdout": (preflight.stdout or "").strip(),\n        }\n\n    apply_proc = subprocess.run(\n        ["git", "apply", "--whitespace=nowarn", "-"],\n        cwd=ROOT,\n        input=patch,\n        text=True,\n        capture_output=True,\n    )\n    if apply_proc.returncode != 0:\n        return {\n            "ok": False,\n            "error": "apply_failed",\n            "stderr": (apply_proc.stderr or "").strip(),\n            "stdout": (apply_proc.stdout or "").strip(),\n        }\n\n    failed_verification = []\n    for rel in rel_paths:\n        target = (ROOT / rel).resolve()\n        if not ensure_under_root(target):\n            failed_verification.append(rel)\n\n    if failed_verification:\n        restore_snapshot(before_state)\n        return {\n            "ok": False,\n            "error": "post_apply_verification_failed",\n            "invalid_paths": failed_verification,\n            "rolled_back": True,\n        }\n\n    return {"ok": True, "applied_files": rel_paths, "file_count": len(rel_paths)}\n\n\ndef tool_commit_changes(args: dict, dry_run: bool) -> dict:\n    if dry_run:\n        return {"ok": False, "error": "blocked_in_dry_run"}\n    msg = str(args.get("message", "Agent commit")).strip()\n    if not msg:\n        return {"ok": False, "error": "missing_message"}\n    proc = subprocess.run(["git", "commit", "-am", msg], cwd=ROOT, capture_output=True, text=True)\n    return {"ok": proc.returncode == 0, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}\n\n\ndef execute_tool_call(tool: str, args: dict, dry_run: bool) -> dict:\n    if tool not in ALLOWED_TOOLS:\n        return {"ok": False, "error": "tool_not_allowed", "tool": tool}\n    if tool == "retrieve":\n        if not INDEX_PATH.exists():\n            return {"ok": False, "error": "missing_index"}\n        query = str(args.get("query", "")).strip()\n        if not query:\n            return {"ok": False, "error": "missing_query"}\n        index_obj = json.loads(INDEX_PATH.read_text(encoding="utf-8"))\n        top_k = int(args.get("top_k", 5))\n        path_prefix = args.get("path_prefix")\n        return {"ok": True, "result": retrieve(index_obj, query=query, top_k=max(1, min(top_k, 20)), path_prefix=path_prefix)}\n    if tool == "search_code":\n        return tool_search_code(args)\n    if tool == "read_file":\n        return tool_read_file(args)\n    if tool == "git_diff":\n        return tool_git_diff(args)\n    if tool == "run_tests":\n        return tool_run_tests(args, dry_run=dry_run)\n    if tool == "write_patch":\n        return tool_write_patch(args, dry_run=dry_run)\n    if tool == "commit_changes":\n        return tool_commit_changes(args, dry_run=dry_run)\n    return {"ok": False, "error": "unhandled_tool"}\n\n\ndef run_agent_task(payload: dict) -> dict:\n    task = str(payload.get("task", "")).strip()\n    dry_run = bool(payload.get("dry_run", True))\n    max_steps = int(payload.get("max_steps", 6))\n    max_steps = max(1, min(max_steps, 25))\n    plan = payload.get("plan", [])\n    run_id = uuid.uuid4().hex[:12]\n\n    trace = {\n        "run_id": run_id,\n        "task": task,\n        "dry_run": dry_run,\n        "max_steps": max_steps,\n        "created_at": datetime.now(timezone.utc).isoformat(),\n        "steps": [],\n    }\n    emit_event("run_started", run_id=run_id, dry_run=dry_run, max_steps=max_steps, task=task[:300])\n\n    if not isinstance(plan, list) or not plan:\n        trace["steps"].append({"tool": "noop", "result": {"ok": True, "detail": "No plan steps provided"}})\n        record_tool_metrics(tool="noop", ok=True, duration_ms=0.0)\n    else:\n        for step in plan[:max_steps]:\n            tool = str(step.get("tool", "")).strip()\n            args = step.get("args", {}) if isinstance(step.get("args", {}), dict) else {}\n            t0 = time.perf_counter()\n            result = execute_tool_call(tool, args, dry_run=dry_run)\n            tool_duration_ms = (time.perf_counter() - t0) * 1000.0\n            ok = bool(result.get("ok", False)) if isinstance(result, dict) else False\n            error = str(result.get("error", "")) if isinstance(result, dict) else "unknown_error"\n            record_tool_metrics(tool=tool or "unknown", ok=ok, duration_ms=tool_duration_ms, error=error)\n            trace["steps"].append(\n                {\n                    "tool": tool,\n                    "args": args,\n                    "duration_ms": round(tool_duration_ms, 3),\n                    "result": result,\n                }\n            )\n\n    RUNS_DIR.mkdir(parents=True, exist_ok=True)\n    run_path = RUNS_DIR / f"{run_id}.json"\n    run_path.write_text(json.dumps(trace, indent=2) + "\\n", encoding="utf-8")\n    emit_event("run_completed", run_id=run_id, step_count=len(trace["steps"]), run_path=str(run_path.relative_to(ROOT)))\n\n    return {\n        "ok": True,\n        "run_id": run_id,\n        "run_path": str(run_path.relative_to(ROOT)),\n        "step_count": len(trace["steps"]),\n        "dry_run": dry_run,\n        "steps": trace["steps"],\n    }\n\n\nclass Handler(BaseHTTPRequestHandler):\n    def _reply(self, payload, status=200):\n        self.send_response(status)\n        self.send_header(\'Content-Type\', \'application/json\')\n        self.end_headers()\n        self.wfile.write(json.dumps(payload).encode(\'utf-8\'))\n\n    def do_GET(self):\n        parsed = urlparse(self.path)\n\n        if parsed.path == \'/metrics\':\n            metrics = load_metrics()\n            kv_obj = load_kv_cache()\n            kv_summary = {}\n            for model_name, store in kv_obj.get("models", {}).items():\n                if not isinstance(store, dict):\n                    continue\n                entries = store.get("entries", {}) if isinstance(store.get("entries", {}), dict) else {}\n                kv_summary[model_name] = {\n                    "entries": len(entries),\n                    "used_tokens": int(store.get("used_tokens", 0)),\n                    "budget_tokens": int(store.get("budget_tokens", DEFAULT_KV_MODEL_BUDGET_TOKENS)),\n                }\n            thresholds = parse_alert_thresholds()\n            alerts = compute_alerts(metrics, thresholds=thresholds)\n            if alerts:\n                emit_event(\'alerts_emitted\', alerts=alerts)\n            self._reply(\n                {\n                    \'ok\': True,\n                    \'service\': \'agent\',\n                    \'metrics\': metrics,\n                    \'kv_cache\': {\'models\': kv_summary},\n                    \'alerts\': alerts,\n                    \'alert_thresholds\': thresholds,\n                }\n            )\n            return\n\n        if parsed.path == \'/tools\':\n            self._reply({\'ok\': True, \'service\': \'agent\', \'tools\': TOOL_SCHEMAS})\n            return\n\n        if parsed.path.startswith(\'/runs/\'):\n            run_id = parsed.path.split(\'/runs/\', 1)[1].strip()\n            target = (RUNS_DIR / f"{run_id}.json").resolve()\n            if not target.exists() or not target.is_file() or not ensure_under_root(target):\n                self._reply({\'error\': \'run_not_found\'}, status=404)\n                return\n            payload = json.loads(target.read_text(encoding=\'utf-8\'))\n            self._reply({\'ok\': True, \'service\': \'agent\', \'run\': payload})\n            return\n\n        if parsed.path == \'/retrieve\':\n            if not INDEX_PATH.exists():\n                self._reply({\'error\': \'missing_index\', \'detail\': \'Run `ai-dev index .` first.\'}, status=400)\n                return\n\n            qs = parse_qs(parsed.query)\n            query = (qs.get(\'q\', [\'\'])[0] or \'\').strip()\n            if not query:\n                self._reply({\'error\': \'missing_query\', \'detail\': \'Provide q=<query>\'}, status=400)\n                return\n\n            try:\n                top_k = int((qs.get(\'top_k\', [\'5\'])[0] or \'5\'))\n            except ValueError:\n                top_k = 5\n            top_k = max(1, min(top_k, 20))\n            path_prefix = (qs.get(\'path_prefix\', [\'\'])[0] or \'\').strip() or None\n\n            index_obj = json.loads(INDEX_PATH.read_text(encoding=\'utf-8\'))\n            payload = retrieve(index_obj, query=query, top_k=top_k, path_prefix=path_prefix)\n            self._reply({\'ok\': True, \'service\': \'agent\', \'retrieval\': payload})\n            return\n\n        if parsed.path == \'/health\':\n            self._reply({\'ok\': True, \'service\': \'agent\'})\n            return\n\n        self._reply({\'error\': \'not found\'}, status=404)\n\n    def do_POST(self):\n        parsed = urlparse(self.path)\n        if parsed.path != \'/agent/run\':\n            self._reply({\'error\': \'not found\'}, status=404)\n            return\n\n        try:\n            content_length = int(self.headers.get(\'Content-Length\', \'0\'))\n        except ValueError:\n            content_length = 0\n        body = self.rfile.read(max(0, content_length))\n\n        try:\n            payload = json.loads(body.decode(\'utf-8\') if body else \'{}\')\n        except Exception:\n            self._reply({\'error\': \'invalid_json\'}, status=400)\n            return\n\n        payload = payload if isinstance(payload, dict) else {}\n        cache_cfg = payload.get("cache", {}) if isinstance(payload.get("cache", {}), dict) else {}\n        cache_enabled = bool(cache_cfg.get("enabled", True))\n        cache_refresh = bool(cache_cfg.get("refresh", False))\n        ttl_seconds = int(cache_cfg.get("ttl_seconds", DEFAULT_CACHE_TTL_SECONDS) or DEFAULT_CACHE_TTL_SECONDS)\n        ttl_seconds = max(1, min(ttl_seconds, 86_400))\n\n        namespace = compute_cache_namespace()\n        key = compute_cache_key(payload)\n        cache_hit = False\n        kv_status = get_kv_reuse_status(payload)\n        started = time.perf_counter()\n        result = None\n\n        if cache_enabled and not cache_refresh:\n            cache_obj = load_cache()\n            entry = get_cache_entry(cache_obj, key=key, namespace=namespace)\n            if entry and isinstance(entry.get("result"), dict):\n                result = entry["result"]\n                cache_hit = True\n                save_cache(cache_obj)\n\n        if result is None:\n            result = run_agent_task(payload)\n            if cache_enabled:\n                cache_obj = load_cache()\n                set_cache_entry(cache_obj, key=key, namespace=namespace, result=result, ttl_seconds=ttl_seconds)\n                save_cache(cache_obj)\n\n        compute_ms = (time.perf_counter() - started) * 1000.0\n        record_cache_metrics(hit=cache_hit, compute_ms=compute_ms, namespace=namespace, key=key)\n\n        self._reply(\n            {\n                \'ok\': True,\n                \'service\': \'agent\',\n                \'result\': result,\n                \'cache\': {\n                    \'enabled\': cache_enabled,\n                    \'refresh\': cache_refresh,\n                    \'hit\': cache_hit,\n                    \'ttl_seconds\': ttl_seconds,\n                    \'namespace\': namespace,\n                    \'key\': key,\n                    \'compute_ms\': round(compute_ms, 2),\n                },\n                \'kv_cache\': kv_status,\n            }\n        )\n\n\nif __name__ == \'__main__\':\n    server = HTTPServer((\'0.0.0.0\', 8091), Handler)\n    print(\'Agent service listening on :8091\')\n    server.serve_forever()\n'


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
