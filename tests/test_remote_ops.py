from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_dev.core import remote_ops


class TestRemoteOps(unittest.TestCase):
    def test_summarize_local_embedding_sink_surfaces_fallback_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "embeddings.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "vector_backend": "deterministic_fallback",
                        "embedding_model": "local-embed",
                        "vector_dim": 16,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summary = remote_ops.summarize_local_embedding_sink(path)

        self.assertEqual(summary["records"], 1)
        self.assertEqual(summary["last_vector_backend"], "deterministic_fallback")
        self.assertEqual(summary["last_embedding_model"], "local-embed")
        self.assertEqual(summary["last_vector_dim"], 16)


if __name__ == "__main__":
    unittest.main()
