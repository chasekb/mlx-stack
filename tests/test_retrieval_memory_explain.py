from __future__ import annotations

import argparse
import io
import json
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import ai_dev.cli as cli


class TestRetrievalMemoryExplain(unittest.TestCase):
    def setUp(self) -> None:
        self.index_path = Path(".ai-dev/index.json")
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._previous = self.index_path.read_text(encoding="utf-8") if self.index_path.exists() else None

    def tearDown(self) -> None:
        if self._previous is None:
            if self.index_path.exists():
                self.index_path.unlink()
        else:
            self.index_path.write_text(self._previous, encoding="utf-8")

    def test_recency_boost_prefers_recent_commits(self) -> None:
        now = time.time()
        recent = cli._recency_boost_from_commit_ts(int(now - 60), now)
        old = cli._recency_boost_from_commit_ts(int(now - 86_400 * 90), now)
        self.assertGreater(recent, old)

    def test_score_symbol_match_has_breakdown(self) -> None:
        now = time.time()
        scored = cli._score_symbol_match(
            symbol={
                "path": "src/example.py",
                "name": "alpha_handler",
                "git_branch": "main",
                "git_commit_ts": int(now - 120),
            },
            query_terms={"alpha"},
            path_prefix="src/",
            changed_files={"src/example.py"},
            current_branch="main",
            include_changed_bias=True,
            now_ts=now,
        )
        self.assertIsNotNone(scored)
        self.assertGreater(scored["score"], 0)
        self.assertIn("score_breakdown", scored)
        self.assertIn("lexical", scored["score_breakdown"])

    def test_memory_explain_json_output(self) -> None:
        branch = cli.get_git_branch_name(Path(".").resolve())
        now_ts = int(time.time())
        index_obj = {
            "schema_version": 2,
            "generated_at": "now",
            "root": str(Path(".").resolve()),
            "file_count": 1,
            "symbols": [
                {
                    "path": "agent/server.py",
                    "name": "alpha_symbol",
                    "line": 1,
                    "kind": "def",
                    "git_branch": branch,
                    "git_commit_sha": "abc",
                    "git_commit_ts": now_ts,
                }
            ],
            "chunks": [
                {
                    "path": "agent/server.py",
                    "chunk_id": 1,
                    "start_line": 1,
                    "end_line": 10,
                    "text_preview": "alpha preview",
                    "terms": ["alpha", "preview"],
                    "git_branch": branch,
                    "git_commit_sha": "abc",
                    "git_commit_ts": now_ts,
                }
            ],
            "files": [],
        }
        self.index_path.write_text(json.dumps(index_obj), encoding="utf-8")

        args = argparse.Namespace(
            query="alpha",
            top_k=3,
            path_prefix=None,
            no_changed_bias=True,
            json=True,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli.command_memory_explain(args)
        self.assertEqual(rc, 0)

        payload = json.loads(buf.getvalue())
        self.assertIn("top_symbols", payload)
        self.assertIn("top_chunks", payload)
        self.assertTrue(len(payload["top_symbols"]) >= 1)
        self.assertIn("score_breakdown", payload["top_symbols"][0])


if __name__ == "__main__":
    unittest.main()
