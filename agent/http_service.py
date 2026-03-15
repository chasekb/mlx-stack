from __future__ import annotations

import json
import time
from typing import Callable

from agent.contracts import AgentHttpContext


def build_metrics_response(context: AgentHttpContext) -> dict:
    metrics = context["load_metrics"]()
    kv_obj = context["load_kv_cache"]()
    kv_summary = {}
    for model_name, store in kv_obj.get("models", {}).items():
        if not isinstance(store, dict):
            continue
        entries = store.get("entries", {}) if isinstance(store.get("entries", {}), dict) else {}
        kv_summary[model_name] = {
            "entries": len(entries),
            "used_tokens": int(store.get("used_tokens", 0)),
            "budget_tokens": int(store.get("budget_tokens", context["default_kv_model_budget_tokens"])),
        }

    thresholds = context["parse_alert_thresholds"]()
    alerts = context["compute_alerts"](metrics, thresholds=thresholds)
    if alerts:
        context["emit_event"]("alerts_emitted", alerts=alerts)

    return {
        "ok": True,
        "service": "agent",
        "metrics": metrics,
        "kv_cache": {"models": kv_summary},
        "alerts": alerts,
        "alert_thresholds": thresholds,
    }


def build_run_response(run_id: str, context: AgentHttpContext) -> tuple[dict, int]:
    target = (context["runs_dir"] / f"{run_id}.json").resolve()
    if not target.exists() or not target.is_file() or not context["ensure_under_root"](target):
        return {"error": "run_not_found"}, 404
    payload = json.loads(target.read_text(encoding="utf-8"))
    return {"ok": True, "service": "agent", "run": payload}, 200


def build_retrieve_response(
    query: str,
    top_k: int,
    path_prefix: str | None,
    context: AgentHttpContext,
) -> tuple[dict, int]:
    if not context["index_path"].exists():
        return {"error": "missing_index", "detail": "Run `ai-dev index .` first."}, 400

    query = (query or "").strip()
    if not query:
        return {"error": "missing_query", "detail": "Provide q=<query>"}, 400

    top_k = max(1, min(top_k, 20))
    index_obj = json.loads(context["index_path"].read_text(encoding="utf-8"))
    retrieval_payload = context["retrieve"](index_obj, query=query, top_k=top_k, path_prefix=path_prefix)
    return {"ok": True, "service": "agent", "retrieval": retrieval_payload}, 200


def build_agent_run_response(
    payload: dict,
    context: AgentHttpContext,
    perf_counter_fn: Callable[[], float] = time.perf_counter,
) -> dict:
    payload = payload if isinstance(payload, dict) else {}
    cache_cfg = payload.get("cache", {}) if isinstance(payload.get("cache", {}), dict) else {}
    cache_enabled = bool(cache_cfg.get("enabled", True))
    cache_refresh = bool(cache_cfg.get("refresh", False))
    ttl_seconds = int(
        cache_cfg.get("ttl_seconds", context["default_cache_ttl_seconds"])
        or context["default_cache_ttl_seconds"]
    )
    ttl_seconds = max(1, min(ttl_seconds, 86_400))

    namespace = context["compute_cache_namespace"]()
    key = context["compute_cache_key"](payload)
    cache_hit = False
    kv_status = context["get_kv_reuse_status"](payload)
    started = perf_counter_fn()
    result = None

    if cache_enabled and not cache_refresh:
        cache_obj = context["load_cache"]()
        entry = context["get_cache_entry"](cache_obj, key=key, namespace=namespace)
        if entry and isinstance(entry.get("result"), dict):
            result = entry["result"]
            cache_hit = True
            context["save_cache"](cache_obj)

    if result is None:
        result = context["run_agent_task"](payload)
        if cache_enabled:
            cache_obj = context["load_cache"]()
            context["set_cache_entry"](
                cache_obj,
                key=key,
                namespace=namespace,
                result=result,
                ttl_seconds=ttl_seconds,
            )
            context["save_cache"](cache_obj)

    compute_ms = (perf_counter_fn() - started) * 1000.0
    context["record_cache_metrics"](
        hit=cache_hit,
        compute_ms=compute_ms,
        namespace=namespace,
        key=key,
    )

    return {
        "ok": True,
        "service": "agent",
        "result": result,
        "cache": {
            "enabled": cache_enabled,
            "refresh": cache_refresh,
            "hit": cache_hit,
            "ttl_seconds": ttl_seconds,
            "namespace": namespace,
            "key": key,
            "compute_ms": round(compute_ms, 2),
        },
        "kv_cache": kv_status,
    }
