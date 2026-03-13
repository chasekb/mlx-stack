import json
import subprocess
import uuid
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from http.server import HTTPServer

from agent import cache_kv
from agent.contracts import AgentHttpContext
from agent import http_api
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


HANDLER_CONTEXT: AgentHttpContext = {
    "load_metrics": load_metrics,
    "load_kv_cache": load_kv_cache,
    "parse_alert_thresholds": parse_alert_thresholds,
    "compute_alerts": compute_alerts,
    "emit_event": emit_event,
    "tool_schemas": TOOL_SCHEMAS,
    "runs_dir": RUNS_DIR,
    "ensure_under_root": ensure_under_root,
    "index_path": INDEX_PATH,
    "retrieve": retrieve,
    "default_kv_model_budget_tokens": DEFAULT_KV_MODEL_BUDGET_TOKENS,
    "default_cache_ttl_seconds": DEFAULT_CACHE_TTL_SECONDS,
    "compute_cache_namespace": compute_cache_namespace,
    "compute_cache_key": compute_cache_key,
    "get_kv_reuse_status": get_kv_reuse_status,
    "load_cache": load_cache,
    "save_cache": save_cache,
    "get_cache_entry": get_cache_entry,
    "set_cache_entry": set_cache_entry,
    "run_agent_task": run_agent_task,
    "record_cache_metrics": record_cache_metrics,
}


Handler = http_api.build_handler(HANDLER_CONTEXT)


if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', 8091), Handler)
    print('Agent service listening on :8091')
    server.serve_forever()
