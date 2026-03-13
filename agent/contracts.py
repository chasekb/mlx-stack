from __future__ import annotations

from typing import Callable, Mapping, TypedDict


class AgentHttpContext(TypedDict):
    load_metrics: Callable[[], dict]
    load_kv_cache: Callable[[], dict]
    parse_alert_thresholds: Callable[[], dict]
    compute_alerts: Callable[[dict, dict], list[dict]]
    emit_event: Callable[..., None]
    tool_schemas: list[dict]
    runs_dir: object
    ensure_under_root: Callable[[object], bool]
    index_path: object
    retrieve: Callable[..., dict]
    default_kv_model_budget_tokens: int
    default_cache_ttl_seconds: int
    compute_cache_namespace: Callable[[], str]
    compute_cache_key: Callable[[dict], str]
    get_kv_reuse_status: Callable[[dict], dict]
    load_cache: Callable[[], dict]
    save_cache: Callable[[dict], None]
    get_cache_entry: Callable[..., dict | None]
    set_cache_entry: Callable[..., None]
    run_agent_task: Callable[[dict], dict]
    record_cache_metrics: Callable[..., None]


REQUIRED_AGENT_HTTP_CONTEXT_KEYS = (
    "load_metrics",
    "load_kv_cache",
    "parse_alert_thresholds",
    "compute_alerts",
    "emit_event",
    "tool_schemas",
    "runs_dir",
    "ensure_under_root",
    "index_path",
    "retrieve",
    "default_kv_model_budget_tokens",
    "default_cache_ttl_seconds",
    "compute_cache_namespace",
    "compute_cache_key",
    "get_kv_reuse_status",
    "load_cache",
    "save_cache",
    "get_cache_entry",
    "set_cache_entry",
    "run_agent_task",
    "record_cache_metrics",
)


CALLABLE_AGENT_HTTP_CONTEXT_KEYS = {
    "load_metrics",
    "load_kv_cache",
    "parse_alert_thresholds",
    "compute_alerts",
    "emit_event",
    "ensure_under_root",
    "retrieve",
    "compute_cache_namespace",
    "compute_cache_key",
    "get_kv_reuse_status",
    "load_cache",
    "save_cache",
    "get_cache_entry",
    "set_cache_entry",
    "run_agent_task",
    "record_cache_metrics",
}


def validate_agent_http_context(context: Mapping[str, object]) -> None:
    missing = [key for key in REQUIRED_AGENT_HTTP_CONTEXT_KEYS if key not in context]
    non_callable = [
        key for key in CALLABLE_AGENT_HTTP_CONTEXT_KEYS if key in context and not callable(context[key])
    ]
    if not missing and not non_callable:
        return

    problems: list[str] = []
    if missing:
        problems.append(f"missing: {', '.join(sorted(missing))}")
    if non_callable:
        problems.append(f"non-callable: {', '.join(sorted(non_callable))}")
    raise ValueError("Invalid agent HTTP context: " + "; ".join(problems))
