from __future__ import annotations

import urllib.error
import unittest

from agent import cache_kv
from agent.server import KV_CACHE_PATH, get_kv_reuse_status


class TestAgentKvCache(unittest.TestCase):
    def setUp(self) -> None:
        self._previous = KV_CACHE_PATH.read_text(encoding="utf-8") if KV_CACHE_PATH.exists() else None
        self._orig_probe = cache_kv.probe_backend_kv_reuse
        self._orig_http_json = cache_kv._http_json
        KV_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        if KV_CACHE_PATH.exists():
            KV_CACHE_PATH.unlink()

    def tearDown(self) -> None:
        cache_kv.probe_backend_kv_reuse = self._orig_probe
        cache_kv._http_json = self._orig_http_json
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

    def test_backend_hit_result_is_used(self) -> None:
        cache_kv.probe_backend_kv_reuse = lambda **_: {
            "status": "hit",
            "reason": "backend_cached_tokens",
            "reused_tokens": 7,
            "backend_url": "http://localhost:4000/v1/completions",
            "backend_latency_ms": 1.5,
        }

        out = get_kv_reuse_status(
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
        self.assertEqual(out.get("status"), "hit")
        self.assertEqual(out.get("reason"), "backend_cached_tokens")
        self.assertEqual(int(out.get("reused_tokens", 0)), 7)
        self.assertEqual(out.get("source"), "backend")
        self.assertIn("backend_url", out)

    def test_backend_miss_result_is_used(self) -> None:
        cache_kv.probe_backend_kv_reuse = lambda **_: {
            "status": "miss",
            "reason": "backend_cached_tokens",
            "reused_tokens": 0,
        }

        out = get_kv_reuse_status(
            {
                "model": "m1",
                "kv_cache": {
                    "enabled": True,
                    "tenant_id": "t1",
                    "session_id": "s1",
                    "prefix": "fresh prefix",
                },
            }
        )
        self.assertEqual(out.get("status"), "miss")
        self.assertEqual(out.get("reason"), "backend_cached_tokens")
        self.assertEqual(int(out.get("reused_tokens", 0)), 0)
        self.assertEqual(out.get("source"), "backend")

    def test_backend_unreachable_returns_error(self) -> None:
        def _raise(**_):
            raise urllib.error.URLError("backend down")

        cache_kv.probe_backend_kv_reuse = _raise

        out = get_kv_reuse_status(
            {
                "model": "m1",
                "kv_cache": {
                    "enabled": True,
                    "tenant_id": "t1",
                    "session_id": "s1",
                    "prefix": "hello",
                },
            }
        )
        self.assertEqual(out.get("status"), "error")
        self.assertEqual(out.get("reason"), "backend_unreachable")
        self.assertEqual(out.get("source"), "backend")

    def test_extract_backend_status_from_usage_cached_tokens(self) -> None:
        parsed = cache_kv._extract_backend_kv_status(
            {
                "usage": {
                    "prompt_tokens": 120,
                    "prompt_tokens_details": {
                        "cached_tokens": 55,
                    },
                }
            }
        )
        self.assertEqual(parsed.get("status"), "hit")
        self.assertEqual(parsed.get("reason"), "backend_cached_tokens")
        self.assertEqual(int(parsed.get("reused_tokens", 0)), 55)

    def test_extract_backend_status_prefers_kv_cache_block(self) -> None:
        parsed = cache_kv._extract_backend_kv_status(
            {
                "kv_cache": {
                    "status": "miss",
                    "reason": "backend_reported",
                    "reused_tokens": 0,
                },
                "usage": {
                    "prompt_tokens_details": {
                        "cached_tokens": 999,
                    }
                },
            }
        )
        self.assertEqual(parsed.get("status"), "miss")
        self.assertEqual(parsed.get("reason"), "backend_reported")
        self.assertEqual(int(parsed.get("reused_tokens", 0)), 0)

    def test_extract_backend_status_requires_kv_signal(self) -> None:
        with self.assertRaises(ValueError):
            cache_kv._extract_backend_kv_status(
                {
                    "id": "cmpl-1",
                    "choices": [{"text": "ok"}],
                }
            )

    def test_probe_backend_kv_reuse_uses_http_response_parser(self) -> None:
        cache_kv._http_json = lambda *_args, **_kwargs: {
            "usage": {
                "prompt_tokens_details": {
                    "cached_tokens": 3,
                }
            }
        }
        out = cache_kv.probe_backend_kv_reuse(
            kv_cfg={"tenant_id": "t1", "session_id": "s1"},
            model="m1",
            prefix="hello world",
            prefix_hash=cache_kv.hash_prefix("hello world"),
        )

        self.assertEqual(out.get("status"), "hit")
        self.assertEqual(int(out.get("reused_tokens", 0)), 3)
        self.assertIn("backend_url", out)


if __name__ == "__main__":
    unittest.main()
