import json
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
