from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

import embedding_queue.server as eq


class TestEmbeddingQueue(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "embedding_jobs_test.db"
        self._old_db_path = eq.DB_PATH
        eq.DB_PATH = self.db_path
        eq._init_db()

    def tearDown(self) -> None:
        eq.DB_PATH = self._old_db_path
        self.tmpdir.cleanup()

    def test_retry_then_dead_letter(self) -> None:
        enq = eq.enqueue_job(kind="file_change", payload={"path": "a.py"}, max_attempts=1)
        self.assertEqual(enq.get("status"), "queued")
        job_id = int(enq["job_id"])

        claimed = eq.claim_next_job()
        self.assertIsNotNone(claimed)
        self.assertEqual(int(claimed["id"]), job_id)
        self.assertEqual(int(claimed["attempts"]), 1)

        failed = eq.fail_job(job_id, "boom")
        self.assertEqual(failed.get("status"), "dead_letter")

        stats = eq.get_stats()
        self.assertEqual(stats.get("dead_letter"), 1)

    def test_retry_status_before_dead_letter(self) -> None:
        enq = eq.enqueue_job(kind="file_change", payload={"path": "b.py"}, max_attempts=2)
        job_id = int(enq["job_id"])

        _ = eq.claim_next_job()
        failed = eq.fail_job(job_id, "transient")
        self.assertEqual(failed.get("status"), "retry")
        self.assertGreater(float(failed.get("next_attempt_at", 0)), 0)

        stats = eq.get_stats()
        self.assertEqual(stats.get("retry"), 1)

    def test_complete_job_moves_to_done(self) -> None:
        enq = eq.enqueue_job(kind="file_change", payload={"path": "c.py"}, max_attempts=3)
        job_id = int(enq["job_id"])
        _ = eq.claim_next_job()
        eq.complete_job(job_id)

        stats = eq.get_stats()
        self.assertEqual(stats.get("done"), 1)


if __name__ == "__main__":
    unittest.main()
