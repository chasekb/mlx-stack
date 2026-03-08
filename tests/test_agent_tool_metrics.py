from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import agent.server as agent_server


class TestAgentToolMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self._previous_metrics = (
            agent_server.METRICS_PATH.read_text(encoding="utf-8") if agent_server.METRICS_PATH.exists() else None
        )
        self._previous_event_log_path = agent_server.EVENT_LOG_PATH
        self._tmpdir = tempfile.TemporaryDirectory()
        agent_server.EVENT_LOG_PATH = Path(self._tmpdir.name) / "agent-events.jsonl"

        agent_server.METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        if agent_server.METRICS_PATH.exists():
            agent_server.METRICS_PATH.unlink()

    def tearDown(self) -> None:
        if agent_server.METRICS_PATH.exists():
            agent_server.METRICS_PATH.unlink()
        if self._previous_metrics is not None:
            agent_server.METRICS_PATH.write_text(self._previous_metrics, encoding="utf-8")
        agent_server.EVENT_LOG_PATH = self._previous_event_log_path
        self._tmpdir.cleanup()

    def test_parse_alert_thresholds_with_env_and_fallback(self) -> None:
        old_errors = os.environ.get("AGENT_ALERT_TOOL_ERRORS")
        old_hit_rate = os.environ.get("AGENT_ALERT_CACHE_HIT_RATE_MIN")
        try:
            os.environ["AGENT_ALERT_TOOL_ERRORS"] = "9"
            os.environ["AGENT_ALERT_CACHE_HIT_RATE_MIN"] = "0.35"
            parsed = agent_server.parse_alert_thresholds()
            self.assertEqual(parsed["max_tool_errors"], 9)
            self.assertAlmostEqual(parsed["min_cache_hit_rate"], 0.35, places=5)

            os.environ["AGENT_ALERT_TOOL_ERRORS"] = "not-a-number"
            os.environ["AGENT_ALERT_CACHE_HIT_RATE_MIN"] = "also-bad"
            parsed_bad = agent_server.parse_alert_thresholds()
            self.assertEqual(parsed_bad["max_tool_errors"], 5)
            self.assertAlmostEqual(parsed_bad["min_cache_hit_rate"], 0.2, places=5)
        finally:
            if old_errors is None:
                os.environ.pop("AGENT_ALERT_TOOL_ERRORS", None)
            else:
                os.environ["AGENT_ALERT_TOOL_ERRORS"] = old_errors
            if old_hit_rate is None:
                os.environ.pop("AGENT_ALERT_CACHE_HIT_RATE_MIN", None)
            else:
                os.environ["AGENT_ALERT_CACHE_HIT_RATE_MIN"] = old_hit_rate

    def test_compute_alerts_for_tool_errors_and_cache_hit_rate(self) -> None:
        metrics = {
            "tools": {
                "a": {"errors": 3},
                "b": {"errors": 2},
            },
            "cache": {
                "requests": 12,
                "hit_rate": 0.05,
            },
        }
        alerts = agent_server.compute_alerts(metrics, {"max_tool_errors": 5, "min_cache_hit_rate": 0.2})
        names = {a.get("name") for a in alerts}
        self.assertIn("tool_errors_threshold_exceeded", names)
        self.assertIn("cache_hit_rate_below_minimum", names)

    def test_noop_records_tool_metrics(self) -> None:
        out = agent_server.run_agent_task({"task": "x", "dry_run": True, "plan": []})
        self.assertTrue(out.get("ok"))

        metrics = json.loads(agent_server.METRICS_PATH.read_text(encoding="utf-8"))
        noop = metrics.get("tools", {}).get("noop", {})
        self.assertEqual(int(noop.get("calls", 0)), 1)
        self.assertEqual(int(noop.get("ok", 0)), 1)
        self.assertEqual(int(noop.get("errors", 0)), 0)

        events = [
            json.loads(line)
            for line in agent_server.EVENT_LOG_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        event_names = [e.get("event") for e in events]
        self.assertIn("run_started", event_names)
        self.assertIn("run_completed", event_names)

    def test_error_tool_records_error_metrics(self) -> None:
        out = agent_server.run_agent_task(
            {
                "task": "x",
                "dry_run": True,
                "plan": [
                    {"tool": "unknown_tool", "args": {}},
                ],
            }
        )
        self.assertTrue(out.get("ok"))

        metrics = json.loads(agent_server.METRICS_PATH.read_text(encoding="utf-8"))
        row = metrics.get("tools", {}).get("unknown_tool", {})
        self.assertEqual(int(row.get("calls", 0)), 1)
        self.assertEqual(int(row.get("ok", 0)), 0)
        self.assertEqual(int(row.get("errors", 0)), 1)
        self.assertGreaterEqual(float(row.get("duration_ms_total", 0.0)), 0.0)


if __name__ == "__main__":
    unittest.main()
