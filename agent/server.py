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

from agent import cache_kv


ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / ".ai-dev" / "index.json"
RUNS_DIR = ROOT / ".ai-dev" / "runs"
CACHE_PATH = cache_kv.CACHE_PATH
METRICS_PATH = ROOT / ".ai-dev" / "metrics.json"
KV_CACHE_PATH = cache_kv.KV_CACHE_PATH
EVENT_LOG_PATH = ROOT / ".ai-dev" / "events" / "agent.jsonl"
DEFAULT_CACHE_TTL_SECONDS = 600
DEFAULT_KV_MODEL_BUDGET_TOKENS = cache_kv.DEFAULT_KV_MODEL_BUDGET_TOKENS
DEFAULT_KV_ENTRY_MAX_TOKENS = cache_kv.DEFAULT_KV_ENTRY_MAX_TOKENS
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


def emit_event(event_type: str, **fields: object) -> None:
    EVENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "ts": time.time(),
        "service": "agent",
        "event": event_type,
        **fields,
    }
    with EVENT_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def parse_alert_thresholds() -> dict:
    raw_errors = "5"
    raw_hit_rate = "0.2"
    try:
        import os

        raw_errors = os.environ.get("AGENT_ALERT_TOOL_ERRORS", "5")
        raw_hit_rate = os.environ.get("AGENT_ALERT_CACHE_HIT_RATE_MIN", "0.2")
    except Exception:
        pass
    try:
        max_tool_errors = max(0, int(raw_errors))
    except ValueError:
        max_tool_errors = 5
    try:
        min_cache_hit_rate = max(0.0, min(1.0, float(raw_hit_rate)))
    except ValueError:
        min_cache_hit_rate = 0.2
    return {
        "max_tool_errors": max_tool_errors,
        "min_cache_hit_rate": min_cache_hit_rate,
    }


def compute_alerts(metrics: dict, thresholds: dict) -> list[dict]:
    alerts: list[dict] = []
    tools = metrics.get("tools", {}) if isinstance(metrics.get("tools", {}), dict) else {}
    total_errors = sum(int(v.get("errors", 0) or 0) for v in tools.values() if isinstance(v, dict))
    max_tool_errors = int(thresholds.get("max_tool_errors", 5) or 5)
    if total_errors >= max_tool_errors:
        alerts.append(
            {
                "name": "tool_errors_threshold_exceeded",
                "severity": "warning",
                "value": total_errors,
                "threshold": max_tool_errors,
                "message": f"tool errors ({total_errors}) >= threshold ({max_tool_errors})",
            }
        )

    cache = metrics.get("cache", {}) if isinstance(metrics.get("cache", {}), dict) else {}
    requests = int(cache.get("requests", 0) or 0)
    hit_rate = float(cache.get("hit_rate", 0.0) or 0.0)
    min_cache_hit_rate = float(thresholds.get("min_cache_hit_rate", 0.2) or 0.2)
    if requests >= 10 and hit_rate < min_cache_hit_rate:
        alerts.append(
            {
                "name": "cache_hit_rate_below_minimum",
                "severity": "warning",
                "value": round(hit_rate, 4),
                "threshold": round(min_cache_hit_rate, 4),
                "message": f"cache hit_rate ({hit_rate:.4f}) < threshold ({min_cache_hit_rate:.4f})",
            }
        )
    return alerts


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json_file(path: Path, default):
    return cache_kv.load_json_file(path, default)


def save_json_file(path: Path, payload: dict) -> None:
    cache_kv.save_json_file(path, payload)


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
    return cache_kv.load_cache()


def save_cache(cache_obj: dict) -> None:
    cache_kv.save_cache(cache_obj)


def get_cache_entry(cache_obj: dict, key: str, namespace: str) -> Optional[dict]:
    return cache_kv.get_cache_entry(cache_obj, key=key, namespace=namespace)


def set_cache_entry(cache_obj: dict, key: str, namespace: str, result: dict, ttl_seconds: int) -> None:
    cache_kv.set_cache_entry(cache_obj, key=key, namespace=namespace, result=result, ttl_seconds=ttl_seconds)


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


def record_tool_metrics(tool: str, ok: bool, duration_ms: float, error: str = "") -> None:
    metrics = load_metrics()
    tools = metrics.setdefault("tools", {})
    row = tools.setdefault(
        tool,
        {
            "calls": 0,
            "ok": 0,
            "errors": 0,
            "duration_ms_total": 0.0,
            "avg_duration_ms": 0.0,
            "last_error": "",
            "last_updated": utc_now_iso(),
        },
    )

    row["calls"] = int(row.get("calls", 0)) + 1
    row["ok"] = int(row.get("ok", 0)) + (1 if ok else 0)
    row["errors"] = int(row.get("errors", 0)) + (0 if ok else 1)
    row["duration_ms_total"] = round(float(row.get("duration_ms_total", 0.0)) + max(0.0, duration_ms), 4)
    row["avg_duration_ms"] = round(row["duration_ms_total"] / max(1, row["calls"]), 4)
    row["last_updated"] = utc_now_iso()
    if not ok and error:
        row["last_error"] = str(error)[:2000]

    metrics["updated_at"] = utc_now_iso()
    save_json_file(METRICS_PATH, metrics)


def estimate_tokens(text: str) -> int:
    return cache_kv.estimate_tokens(text)


def normalize_prefix_text(payload: dict) -> str:
    return cache_kv.normalize_prefix_text(payload)


def hash_prefix(prefix: str) -> str:
    return cache_kv.hash_prefix(prefix)


def load_kv_cache() -> dict:
    return cache_kv.load_kv_cache()


def save_kv_cache(obj: dict) -> None:
    cache_kv.save_kv_cache(obj)


def _ensure_model_store(kv_obj: dict, model: str, budget_tokens: int) -> dict:
    return cache_kv._ensure_model_store(kv_obj, model=model, budget_tokens=budget_tokens)


def _enforce_model_budget(store: dict, keep_key: str) -> list[str]:
    return cache_kv._enforce_model_budget(store, keep_key=keep_key)


def get_kv_reuse_status(payload: dict) -> dict:
    return cache_kv.get_kv_reuse_status(payload)


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
    emit_event("run_started", run_id=run_id, dry_run=dry_run, max_steps=max_steps, task=task[:300])

    if not isinstance(plan, list) or not plan:
        trace["steps"].append({"tool": "noop", "result": {"ok": True, "detail": "No plan steps provided"}})
        record_tool_metrics(tool="noop", ok=True, duration_ms=0.0)
    else:
        for step in plan[:max_steps]:
            tool = str(step.get("tool", "")).strip()
            args = step.get("args", {}) if isinstance(step.get("args", {}), dict) else {}
            t0 = time.perf_counter()
            result = execute_tool_call(tool, args, dry_run=dry_run)
            tool_duration_ms = (time.perf_counter() - t0) * 1000.0
            ok = bool(result.get("ok", False)) if isinstance(result, dict) else False
            error = str(result.get("error", "")) if isinstance(result, dict) else "unknown_error"
            record_tool_metrics(tool=tool or "unknown", ok=ok, duration_ms=tool_duration_ms, error=error)
            trace["steps"].append(
                {
                    "tool": tool,
                    "args": args,
                    "duration_ms": round(tool_duration_ms, 3),
                    "result": result,
                }
            )

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_path = RUNS_DIR / f"{run_id}.json"
    run_path.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    emit_event("run_completed", run_id=run_id, step_count=len(trace["steps"]), run_path=str(run_path.relative_to(ROOT)))

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
            thresholds = parse_alert_thresholds()
            alerts = compute_alerts(metrics, thresholds=thresholds)
            if alerts:
                emit_event('alerts_emitted', alerts=alerts)
            self._reply(
                {
                    'ok': True,
                    'service': 'agent',
                    'metrics': metrics,
                    'kv_cache': {'models': kv_summary},
                    'alerts': alerts,
                    'alert_thresholds': thresholds,
                }
            )
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
