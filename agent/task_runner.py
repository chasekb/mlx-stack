from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


def run_agent_task(
    payload: dict,
    *,
    execute_tool_call_fn: Callable[[str, dict, bool], dict],
    record_tool_metrics_fn: Callable[[str, bool, float, str], None],
    emit_event_fn: Callable[..., None],
    runs_dir: Path,
    root: Path,
    perf_counter_fn: Callable[[], float],
) -> dict:
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
    emit_event_fn("run_started", run_id=run_id, dry_run=dry_run, max_steps=max_steps, task=task[:300])

    if not isinstance(plan, list) or not plan:
        trace["steps"].append({"tool": "noop", "result": {"ok": True, "detail": "No plan steps provided"}})
        record_tool_metrics_fn(tool="noop", ok=True, duration_ms=0.0, error="")
    else:
        for step in plan[:max_steps]:
            tool = str(step.get("tool", "")).strip()
            args = step.get("args", {}) if isinstance(step.get("args", {}), dict) else {}
            t0 = perf_counter_fn()
            result = execute_tool_call_fn(tool, args, dry_run)
            tool_duration_ms = (perf_counter_fn() - t0) * 1000.0
            ok = bool(result.get("ok", False)) if isinstance(result, dict) else False
            error = str(result.get("error", "")) if isinstance(result, dict) else "unknown_error"
            record_tool_metrics_fn(tool=tool or "unknown", ok=ok, duration_ms=tool_duration_ms, error=error)
            trace["steps"].append(
                {
                    "tool": tool,
                    "args": args,
                    "duration_ms": round(tool_duration_ms, 3),
                    "result": result,
                }
            )

    runs_dir.mkdir(parents=True, exist_ok=True)
    run_path = runs_dir / f"{run_id}.json"
    run_path.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    emit_event_fn("run_completed", run_id=run_id, step_count=len(trace["steps"]), run_path=str(run_path.relative_to(root)))

    return {
        "ok": True,
        "run_id": run_id,
        "run_path": str(run_path.relative_to(root)),
        "step_count": len(trace["steps"]),
        "dry_run": dry_run,
        "steps": trace["steps"],
    }
