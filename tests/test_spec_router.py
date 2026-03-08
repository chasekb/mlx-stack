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

    def test_decode_model_calls_with_draft_fallback(self) -> None:
        original_request = sr.request_model_tokens
        try:
            def fake_request(*, api_url: str, model: str, prompt: str, max_tokens: int, timeout: float):
                if "fast" in model:
                    raise RuntimeError("draft failed")
                return (["final", "answer"], 12.5)

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
        self.assertEqual(out["draft_token_count"], 0)
        self.assertEqual(out["target_token_count"], 2)
        self.assertIn("draft_error", out)
        self.assertEqual(out["output_tokens"], ["final", "answer"])

    def test_decode_requires_prompt_or_tokens(self) -> None:
        with self.assertRaises(ValueError):
            sr.run_speculative_decode({})


if __name__ == "__main__":
    unittest.main()
