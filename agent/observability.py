from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from agent import cache_kv


ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / ".ai-dev" / "metrics.json"
EVENT_LOG_PATH = ROOT / ".ai-dev" / "events" / "agent.jsonl"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def emit_event(event_type: str, *, event_log_path: Path = EVENT_LOG_PATH, **fields: object) -> None:
    event_log_path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "ts": time.time(),
        "service": "agent",
        "event": event_type,
        **fields,
    }
    with event_log_path.open("a", encoding="utf-8") as f:
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


def load_metrics(metrics_path: Path = METRICS_PATH) -> dict:
    metrics = cache_kv.load_json_file(
        metrics_path,
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


def record_cache_metrics(hit: bool, compute_ms: float, namespace: str, key: str, metrics_path: Path = METRICS_PATH) -> None:
    metrics = load_metrics(metrics_path=metrics_path)
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
    cache_kv.save_json_file(metrics_path, metrics)


def record_tool_metrics(tool: str, ok: bool, duration_ms: float, error: str = "", metrics_path: Path = METRICS_PATH) -> None:
    metrics = load_metrics(metrics_path=metrics_path)
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
    cache_kv.save_json_file(metrics_path, metrics)
