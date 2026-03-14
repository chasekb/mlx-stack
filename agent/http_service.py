from __future__ import annotations

import time
from typing import Callable

from agent.contracts import AgentHttpContext


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
