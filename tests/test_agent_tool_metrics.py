from __future__ import annotations

import json
import unittest

from agent.server import METRICS_PATH, run_agent_task


class TestAgentToolMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self._previous = METRICS_PATH.read_text(encoding="utf-8") if METRICS_PATH.exists() else None
        METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        if METRICS_PATH.exists():
            METRICS_PATH.unlink()

    def tearDown(self) -> None:
        if METRICS_PATH.exists():
            METRICS_PATH.unlink()
        if self._previous is not None:
            METRICS_PATH.write_text(self._previous, encoding="utf-8")

    def test_noop_records_tool_metrics(self) -> None:
        out = run_agent_task({"task": "x", "dry_run": True, "plan": []})
        self.assertTrue(out.get("ok"))

        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        noop = metrics.get("tools", {}).get("noop", {})
        self.assertEqual(int(noop.get("calls", 0)), 1)
        self.assertEqual(int(noop.get("ok", 0)), 1)
        self.assertEqual(int(noop.get("errors", 0)), 0)

    def test_error_tool_records_error_metrics(self) -> None:
        out = run_agent_task(
            {
                "task": "x",
                "dry_run": True,
                "plan": [
                    {"tool": "unknown_tool", "args": {}},
                ],
            }
        )
        self.assertTrue(out.get("ok"))

        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        row = metrics.get("tools", {}).get("unknown_tool", {})
        self.assertEqual(int(row.get("calls", 0)), 1)
        self.assertEqual(int(row.get("ok", 0)), 0)
        self.assertEqual(int(row.get("errors", 0)), 1)
        self.assertGreaterEqual(float(row.get("duration_ms_total", 0.0)), 0.0)


if __name__ == "__main__":
    unittest.main()
