from __future__ import annotations

import unittest
from pathlib import Path

from agent.server import KV_CACHE_PATH, get_kv_reuse_status


class TestAgentKvCache(unittest.TestCase):
    def setUp(self) -> None:
        self._previous = KV_CACHE_PATH.read_text(encoding="utf-8") if KV_CACHE_PATH.exists() else None
        KV_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        if KV_CACHE_PATH.exists():
            KV_CACHE_PATH.unlink()

    def tearDown(self) -> None:
        if KV_CACHE_PATH.exists():
            KV_CACHE_PATH.unlink()
        if self._previous is not None:
            KV_CACHE_PATH.write_text(self._previous, encoding="utf-8")

    def test_missing_session_id_bypasses(self) -> None:
        out = get_kv_reuse_status({"model": "m1", "kv_cache": {"enabled": True, "prefix": "hello"}})
        self.assertEqual(out.get("status"), "bypass")
        self.assertEqual(out.get("reason"), "missing_session_id")

    def test_prefix_hash_mismatch_rejected(self) -> None:
        out = get_kv_reuse_status(
            {
                "model": "m1",
                "kv_cache": {
                    "enabled": True,
                    "tenant_id": "t1",
                    "session_id": "s1",
                    "prefix": "abc",
                    "prefix_hash": "bad_hash",
                },
            }
        )
        self.assertEqual(out.get("status"), "rejected")
        self.assertEqual(out.get("reason"), "prefix_hash_mismatch")

    def test_miss_then_hit_on_prefix_extension(self) -> None:
        first = get_kv_reuse_status(
            {
                "model": "m1",
                "kv_cache": {
                    "enabled": True,
                    "tenant_id": "t1",
                    "session_id": "s1",
                    "prefix": "hello world",
                },
            }
        )
        self.assertEqual(first.get("status"), "miss")
        self.assertEqual(first.get("reason"), "cold_start")

        second = get_kv_reuse_status(
            {
                "model": "m1",
                "kv_cache": {
                    "enabled": True,
                    "tenant_id": "t1",
                    "session_id": "s1",
                    "prefix": "hello world again",
                },
            }
        )
        self.assertEqual(second.get("status"), "hit")
        self.assertEqual(second.get("reason"), "prefix_extension")
        self.assertGreaterEqual(int(second.get("reused_tokens", 0)), 1)

    def test_budget_eviction_occurs(self) -> None:
        base = {
            "model": "m-budget",
            "kv_cache": {
                "enabled": True,
                "tenant_id": "tenant",
                "model_budget_tokens": 8,
                "entry_max_tokens": 8,
            },
        }
        out = None
        # Budget is clamped to >=256 in implementation; with entry_max_tokens=8,
        # eviction should begin after enough distinct sessions are inserted.
        for i in range(1, 40):
            payload = dict(base)
            payload["kv_cache"] = dict(
                base["kv_cache"],
                session_id=f"s{i}",
                prefix=f"prefix {i} one two three four five six seven",
            )
            out = get_kv_reuse_status(payload)

        self.assertIsNotNone(out)
        self.assertEqual(out.get("status"), "miss")
        self.assertTrue(len(out.get("evicted_entries", [])) >= 1)


if __name__ == "__main__":
    unittest.main()
