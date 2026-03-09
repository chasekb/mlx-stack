from __future__ import annotations

import unittest

import spec_router.server as sr


class TestSpecRouter(unittest.TestCase):
    def test_run_speculative_loop_counts_acceptance(self) -> None:
        out = sr.run_speculative_loop(
            draft_tokens=["a", "b", "x"],
            target_tokens=["a", "b", "c", "d"],
        )
        self.assertEqual(out["accepted_tokens"], 2)
        self.assertEqual(out["compared_tokens"], 3)
        self.assertEqual(out["output_tokens"], ["a", "b", "c", "d"])

    def test_decode_uses_provided_tokens_without_model_calls(self) -> None:
        out = sr.run_speculative_decode(
            {
                "draft_tokens": ["hello", "world"],
                "target_tokens": ["hello", "cursor"],
            }
        )
        self.assertEqual(out["source"], "provided_tokens")
        self.assertEqual(out["accepted_tokens"], 1)
        self.assertEqual(out["output_tokens"], ["hello", "cursor"])

    def test_streaming_acceptance_loop_model_calls(self) -> None:
        original_request = sr.request_model_tokens
        try:
            draft_seq = ["def", "oops", "world", ""]
            target_seq = ["def", "hello", "world", ""]
            draft_idx = {"i": 0}
            target_idx = {"i": 0}

            def fake_request(*, api_url: str, model: str, prompt: str, max_tokens: int, timeout: float):
                if "fast" in model:
                    tok = draft_seq[draft_idx["i"]]
                    draft_idx["i"] += 1
                    return ([tok] if tok else [], 1.0)
                tok = target_seq[target_idx["i"]]
                target_idx["i"] += 1
                return ([tok] if tok else [], 2.0)

            sr.request_model_tokens = fake_request
            out = sr.run_speculative_decode(
                {
                    "prompt": "Write function",
                    "draft_model": "local-mlx-fast",
                    "target_model": "local-mlx",
                    "max_tokens": 8,
                    "stream_loop": True,
                }
            )
        finally:
            sr.request_model_tokens = original_request

        self.assertEqual(out["source"], "model_calls")
        self.assertEqual(out["loop_mode"], "stream_acceptance")
        self.assertEqual(out["accepted_tokens"], 2)
        self.assertEqual(out["compared_tokens"], 3)
        self.assertEqual(out["rejected_tokens"], 1)
        self.assertEqual(out["output_tokens"], ["def", "hello", "world"])
        self.assertEqual(len(out.get("steps", [])), 3)

    def test_decode_model_calls_with_draft_fallback(self) -> None:
        original_request = sr.request_model_tokens
        try:
            target_seq = ["final", "answer", ""]
            target_idx = {"i": 0}

            def fake_request(*, api_url: str, model: str, prompt: str, max_tokens: int, timeout: float):
                if "fast" in model:
                    raise RuntimeError("draft failed")
                tok = target_seq[target_idx["i"]]
                target_idx["i"] += 1
                return ([tok] if tok else [], 12.5)

            sr.request_model_tokens = fake_request
            out = sr.run_speculative_decode(
                {
                    "prompt": "Solve task",
                    "draft_model": "local-mlx-fast",
                    "target_model": "local-mlx",
                    "max_tokens": 12,
                }
            )
        finally:
            sr.request_model_tokens = original_request

        self.assertEqual(out["source"], "model_calls")
        self.assertEqual(out["draft_token_count"], 2)
        self.assertEqual(out["target_token_count"], 2)
        self.assertIn("draft_error", out)
        self.assertEqual(out["output_tokens"], ["final", "answer"])

    def test_decode_model_calls_non_stream_compatibility_mode(self) -> None:
        original_request = sr.request_model_tokens
        try:
            def fake_request(*, api_url: str, model: str, prompt: str, max_tokens: int, timeout: float):
                if "fast" in model:
                    return (["a", "b", "x"], 4.0)
                return (["a", "b", "c", "d"], 8.0)

            sr.request_model_tokens = fake_request
            out = sr.run_speculative_decode(
                {
                    "prompt": "compat mode",
                    "draft_model": "local-mlx-fast",
                    "target_model": "local-mlx",
                    "max_tokens": 12,
                    "stream_loop": False,
                }
            )
        finally:
            sr.request_model_tokens = original_request

        self.assertEqual(out["source"], "model_calls")
        self.assertEqual(out["accepted_tokens"], 2)
        self.assertEqual(out["compared_tokens"], 3)
        self.assertEqual(out["output_tokens"], ["a", "b", "c", "d"])

    def test_decode_requires_prompt_or_tokens(self) -> None:
        with self.assertRaises(ValueError):
            sr.run_speculative_decode({})


if __name__ == "__main__":
    unittest.main()
