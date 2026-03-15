from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent.http_service import (
    build_agent_run_response,
    build_metrics_response,
    build_retrieve_response,
    build_run_response,
)


def _build_context() -> dict[str, object]:
    return {
        "load_metrics": lambda: {},
        "load_kv_cache": lambda: {"models": {}},
        "parse_alert_thresholds": lambda: {},
        "compute_alerts": lambda metrics, thresholds=None: [],
        "emit_event": lambda *_args, **_kwargs: None,
        "tool_schemas": [],
        "runs_dir": None,
        "ensure_under_root": lambda _path: True,
        "index_path": None,
        "retrieve": lambda *_args, **_kwargs: {},
        "default_kv_model_budget_tokens": 0,
        "default_cache_ttl_seconds": 600,
        "compute_cache_namespace": lambda: "ns",
        "compute_cache_key": lambda _payload: "key",
        "get_kv_reuse_status": lambda _payload: {"status": "bypass"},
        "load_cache": lambda: {},
        "save_cache": lambda _cache_obj: None,
        "get_cache_entry": lambda *_args, **_kwargs: None,
        "set_cache_entry": lambda *_args, **_kwargs: None,
        "run_agent_task": lambda _payload: {"ok": True, "source": "task"},
        "record_cache_metrics": lambda **_kwargs: None,
    }


class _PerfCounter:
    def __init__(self, values: list[float]) -> None:
        self._values = values
        self._idx = 0

    def __call__(self) -> float:
        out = self._values[self._idx]
        self._idx += 1
        return out


class TestAgentHttpService(unittest.TestCase):
    def test_build_metrics_response_emits_alert_event_and_summarizes_kv_models(self) -> None:
        events: list[tuple[str, dict]] = []
        context = _build_context()
        context["load_metrics"] = lambda: {"cache": {"hits": 2, "misses": 1}}
        context["load_kv_cache"] = lambda: {
            "models": {
                "coder": {
                    "entries": {"k1": {"tokens": 10}, "k2": {"tokens": 20}},
                    "used_tokens": 30,
                },
                "skip_me": "not-a-dict",
            }
        }
        context["parse_alert_thresholds"] = lambda: {"tool_errors": 1}
        context["compute_alerts"] = lambda metrics, thresholds=None: [
            {"kind": "tool_errors", "value": 2, "threshold": 1}
        ]
        context["emit_event"] = lambda name, **payload: events.append((name, payload))
        context["default_kv_model_budget_tokens"] = 999

        out = build_metrics_response(context)  # type: ignore[arg-type]

        self.assertTrue(out.get("ok"))
        self.assertEqual(out.get("service"), "agent")
        models = out.get("kv_cache", {}).get("models", {})
        self.assertEqual(models.get("coder", {}).get("entries"), 2)
        self.assertEqual(models.get("coder", {}).get("used_tokens"), 30)
        self.assertEqual(models.get("coder", {}).get("budget_tokens"), 999)
        self.assertNotIn("skip_me", models)
        self.assertEqual(len(out.get("alerts", [])), 1)
        self.assertEqual(events, [("alerts_emitted", {"alerts": out["alerts"]})])

    def test_build_run_response_success_and_not_found(self) -> None:
        context = _build_context()
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            target = runs_dir / "run-123.json"
            target.write_text(json.dumps({"task": "ok"}), encoding="utf-8")

            context["runs_dir"] = runs_dir
            context["ensure_under_root"] = lambda _path: True

            payload, status = build_run_response("run-123", context)  # type: ignore[arg-type]
            self.assertEqual(status, 200)
            self.assertTrue(payload.get("ok"))
            self.assertEqual(payload.get("run", {}).get("task"), "ok")

            missing_payload, missing_status = build_run_response("missing", context)  # type: ignore[arg-type]
            self.assertEqual(missing_status, 404)
            self.assertEqual(missing_payload, {"error": "run_not_found"})

            context["ensure_under_root"] = lambda _path: False
            denied_payload, denied_status = build_run_response("run-123", context)  # type: ignore[arg-type]
            self.assertEqual(denied_status, 404)
            self.assertEqual(denied_payload, {"error": "run_not_found"})

    def test_build_retrieve_response_error_and_success_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "index.json"
            context = _build_context()
            context["index_path"] = index_path

            missing_payload, missing_status = build_retrieve_response(
                query="query",
                top_k=5,
                path_prefix=None,
                context=context,  # type: ignore[arg-type]
            )
            self.assertEqual(missing_status, 400)
            self.assertEqual(missing_payload.get("error"), "missing_index")

            index_path.write_text(json.dumps({"chunks": [], "symbols": []}), encoding="utf-8")

            no_query_payload, no_query_status = build_retrieve_response(
                query="   ",
                top_k=5,
                path_prefix=None,
                context=context,  # type: ignore[arg-type]
            )
            self.assertEqual(no_query_status, 400)
            self.assertEqual(no_query_payload.get("error"), "missing_query")

            retrieve_calls: list[dict[str, object]] = []

            def _retrieve(index_obj, **kwargs):
                retrieve_calls.append({"index_obj": index_obj, **kwargs})
                return {"symbols": [], "chunks": []}

            context["retrieve"] = _retrieve

            ok_payload, ok_status = build_retrieve_response(
                query="needle",
                top_k=999,
                path_prefix="agent/",
                context=context,  # type: ignore[arg-type]
            )
            self.assertEqual(ok_status, 200)
            self.assertTrue(ok_payload.get("ok"))
            self.assertEqual(len(retrieve_calls), 1)
            self.assertEqual(retrieve_calls[0].get("query"), "needle")
            self.assertEqual(retrieve_calls[0].get("top_k"), 20)
            self.assertEqual(retrieve_calls[0].get("path_prefix"), "agent/")

            _, _ = build_retrieve_response(
                query="needle",
                top_k=-2,
                path_prefix=None,
                context=context,  # type: ignore[arg-type]
            )
            self.assertEqual(retrieve_calls[-1].get("top_k"), 1)

    def test_build_agent_run_response_with_cache_hit(self) -> None:
        saved = {"called": False}
        context = _build_context()
        context["get_cache_entry"] = lambda *_args, **_kwargs: {"result": {"ok": True, "source": "cache"}}
        context["save_cache"] = lambda _cache_obj: saved.__setitem__("called", True)
        context["run_agent_task"] = lambda _payload: {"ok": True, "source": "task_unexpected"}

        out = build_agent_run_response(
            payload={"cache": {"enabled": True}},
            context=context,  # type: ignore[arg-type]
            perf_counter_fn=_PerfCounter([1.0, 1.02]),
        )

        self.assertTrue(out.get("ok"))
        self.assertEqual(out.get("service"), "agent")
        self.assertEqual(out.get("result", {}).get("source"), "cache")
        self.assertTrue(out.get("cache", {}).get("hit"))
        self.assertTrue(saved["called"])

    def test_build_agent_run_response_with_cache_miss_sets_entry(self) -> None:
        set_calls: list[dict] = []
        context = _build_context()
        context["set_cache_entry"] = lambda _cache_obj, **kwargs: set_calls.append(kwargs)

        out = build_agent_run_response(
            payload={"cache": {"enabled": True, "ttl_seconds": 120}},
            context=context,  # type: ignore[arg-type]
            perf_counter_fn=_PerfCounter([2.0, 2.05]),
        )

        self.assertTrue(out.get("ok"))
        self.assertEqual(out.get("result", {}).get("source"), "task")
        self.assertFalse(out.get("cache", {}).get("hit"))
        self.assertEqual(out.get("cache", {}).get("ttl_seconds"), 120)
        self.assertEqual(len(set_calls), 1)
        self.assertEqual(set_calls[0].get("ttl_seconds"), 120)

    def test_build_agent_run_response_clamps_ttl(self) -> None:
        context = _build_context()

        low = build_agent_run_response(
            payload={"cache": {"enabled": True, "ttl_seconds": -5}},
            context=context,  # type: ignore[arg-type]
            perf_counter_fn=_PerfCounter([3.0, 3.01]),
        )
        zero = build_agent_run_response(
            payload={"cache": {"enabled": True, "ttl_seconds": 0}},
            context=context,  # type: ignore[arg-type]
            perf_counter_fn=_PerfCounter([3.1, 3.11]),
        )
        high = build_agent_run_response(
            payload={"cache": {"enabled": True, "ttl_seconds": 999999}},
            context=context,  # type: ignore[arg-type]
            perf_counter_fn=_PerfCounter([4.0, 4.01]),
        )

        self.assertEqual(low.get("cache", {}).get("ttl_seconds"), 1)
        self.assertEqual(zero.get("cache", {}).get("ttl_seconds"), 600)
        self.assertEqual(high.get("cache", {}).get("ttl_seconds"), 86_400)


if __name__ == "__main__":
    unittest.main()
