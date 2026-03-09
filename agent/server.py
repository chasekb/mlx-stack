import json
import subprocess
import uuid
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse
from http.server import BaseHTTPRequestHandler, HTTPServer

from agent import cache_kv
from agent import observability
from agent import retrieval
from agent import runtime_context
from agent import schemas
from agent import task_runner
from agent import tooling


ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = tooling.INDEX_PATH
RUNS_DIR = ROOT / ".ai-dev" / "runs"
CACHE_PATH = cache_kv.CACHE_PATH
METRICS_PATH = ROOT / ".ai-dev" / "metrics.json"
KV_CACHE_PATH = cache_kv.KV_CACHE_PATH
EVENT_LOG_PATH = ROOT / ".ai-dev" / "events" / "agent.jsonl"
DEFAULT_CACHE_TTL_SECONDS = 600
DEFAULT_KV_MODEL_BUDGET_TOKENS = cache_kv.DEFAULT_KV_MODEL_BUDGET_TOKENS
DEFAULT_KV_ENTRY_MAX_TOKENS = cache_kv.DEFAULT_KV_ENTRY_MAX_TOKENS
ALLOWED_TOOLS = schemas.ALLOWED_TOOLS

TOOL_SCHEMAS = schemas.TOOL_SCHEMAS

PATCH_DENY_PREFIXES = tooling.PATCH_DENY_PREFIXES


def emit_event(event_type: str, **fields: object) -> None:
    observability.emit_event(event_type, event_log_path=EVENT_LOG_PATH, **fields)


def parse_alert_thresholds() -> dict:
    return observability.parse_alert_thresholds()


def compute_alerts(metrics: dict, thresholds: dict) -> list[dict]:
    return observability.compute_alerts(metrics, thresholds=thresholds)


def utc_now_iso() -> str:
    return observability.utc_now_iso()


def load_json_file(path: Path, default):
    return cache_kv.load_json_file(path, default)


def save_json_file(path: Path, payload: dict) -> None:
    cache_kv.save_json_file(path, payload)


def get_git_branch() -> str:
    return runtime_context.get_git_branch(root=ROOT)


def get_index_signature() -> str:
    return runtime_context.get_index_signature(index_path=INDEX_PATH)


def compute_cache_namespace() -> str:
    return runtime_context.compute_cache_namespace(root=ROOT, index_path=INDEX_PATH)


def normalize_task_payload(payload: dict) -> dict:
    return runtime_context.normalize_task_payload(payload)


def compute_cache_key(payload: dict) -> str:
    return runtime_context.compute_cache_key(payload)


def load_cache() -> dict:
    return cache_kv.load_cache()


def save_cache(cache_obj: dict) -> None:
    cache_kv.save_cache(cache_obj)


def get_cache_entry(cache_obj: dict, key: str, namespace: str) -> Optional[dict]:
    return cache_kv.get_cache_entry(cache_obj, key=key, namespace=namespace)


def set_cache_entry(cache_obj: dict, key: str, namespace: str, result: dict, ttl_seconds: int) -> None:
    cache_kv.set_cache_entry(cache_obj, key=key, namespace=namespace, result=result, ttl_seconds=ttl_seconds)


def load_metrics() -> dict:
    return observability.load_metrics(metrics_path=METRICS_PATH)


def record_cache_metrics(hit: bool, compute_ms: float, namespace: str, key: str) -> None:
    observability.record_cache_metrics(hit, compute_ms=compute_ms, namespace=namespace, key=key, metrics_path=METRICS_PATH)


def record_tool_metrics(tool: str, ok: bool, duration_ms: float, error: str = "") -> None:
    observability.record_tool_metrics(
        tool,
        ok=ok,
        duration_ms=duration_ms,
        error=error,
        metrics_path=METRICS_PATH,
    )


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
    return retrieval.tokenize(text)


def retrieve(index_obj: dict, query: str, top_k: int = 5, path_prefix: Optional[str] = None) -> dict:
    return retrieval.retrieve(index_obj=index_obj, query=query, top_k=top_k, path_prefix=path_prefix)


def ensure_under_root(path: Path) -> bool:
    return tooling.ensure_under_root(path)


def normalize_patch_path(path_text: str) -> Optional[str]:
    return tooling.normalize_patch_path(path_text)


def extract_patch_paths(patch_text: str) -> list[str]:
    return tooling.extract_patch_paths(patch_text)


def path_allowed_for_patch(rel_path: str) -> bool:
    return tooling.path_allowed_for_patch(rel_path)


def snapshot_paths(rel_paths: list[str]) -> dict:
    return tooling.snapshot_paths(rel_paths)


def restore_snapshot(snapshot: dict) -> None:
    tooling.restore_snapshot(snapshot)


def tool_search_code(args: dict) -> dict:
    return tooling.tool_search_code(args)


def tool_read_file(args: dict) -> dict:
    return tooling.tool_read_file(args)


def tool_git_diff(_: dict) -> dict:
    return tooling.tool_git_diff(_)


def tool_run_tests(args: dict, dry_run: bool) -> dict:
    return tooling.tool_run_tests(args, dry_run=dry_run)


def tool_write_patch(args: dict, dry_run: bool) -> dict:
    return tooling.tool_write_patch(args, dry_run=dry_run)


def tool_commit_changes(args: dict, dry_run: bool) -> dict:
    return tooling.tool_commit_changes(args, dry_run=dry_run)


def execute_tool_call(tool: str, args: dict, dry_run: bool) -> dict:
    return tooling.execute_tool_call(
        tool=tool,
        args=args,
        dry_run=dry_run,
        allowed_tools=ALLOWED_TOOLS,
        retrieve_fn=retrieve,
    )


def run_agent_task(payload: dict) -> dict:
    return task_runner.run_agent_task(
        payload,
        execute_tool_call_fn=execute_tool_call,
        record_tool_metrics_fn=record_tool_metrics,
        emit_event_fn=emit_event,
        runs_dir=RUNS_DIR,
        root=ROOT,
        perf_counter_fn=time.perf_counter,
    )


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
