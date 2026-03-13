from __future__ import annotations

import unittest
from pathlib import Path

from agent import http_api


def _build_context() -> dict[str, object]:
    return {
        "load_metrics": lambda: {},
        "load_kv_cache": lambda: {"models": {}},
        "parse_alert_thresholds": lambda: {},
        "compute_alerts": lambda metrics, thresholds=None: [],
        "emit_event": lambda *_args, **_kwargs: None,
        "tool_schemas": [],
        "runs_dir": Path("."),
        "ensure_under_root": lambda _path: True,
        "index_path": Path(".ai-dev/index.json"),
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
        "run_agent_task": lambda _payload: {"ok": True},
        "record_cache_metrics": lambda **_kwargs: None,
    }


class TestAgentHttpApiContextValidation(unittest.TestCase):
    def test_build_handler_accepts_valid_context(self) -> None:
        handler_cls = http_api.build_handler(_build_context())
        self.assertTrue(hasattr(handler_cls, "do_GET"))
        self.assertTrue(hasattr(handler_cls, "do_POST"))

    def test_build_handler_rejects_missing_required_context_key(self) -> None:
        context = _build_context()
        del context["load_metrics"]

        with self.assertRaises(ValueError) as ctx:
            http_api.build_handler(context)

        self.assertIn("missing: load_metrics", str(ctx.exception))

    def test_build_handler_rejects_non_callable_context_key(self) -> None:
        context = _build_context()
        context["retrieve"] = "not-callable"

        with self.assertRaises(ValueError) as ctx:
            http_api.build_handler(context)

        self.assertIn("non-callable: retrieve", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
