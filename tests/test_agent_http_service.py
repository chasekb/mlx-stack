from __future__ import annotations

import unittest

from agent.http_service import build_agent_run_response


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
