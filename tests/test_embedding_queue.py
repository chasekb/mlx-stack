from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

import embedding_queue.server as eq


class TestEmbeddingQueue(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "embedding_jobs_test.db"
        self.event_log_path = Path(self.tmpdir.name) / "embed-queue-events.jsonl"
        self._old_db_path = eq.DB_PATH
        self._old_event_log_path = eq.EVENT_LOG_PATH
        eq.DB_PATH = self.db_path
        eq.EVENT_LOG_PATH = self.event_log_path
        eq._init_db()

    def tearDown(self) -> None:
        eq.DB_PATH = self._old_db_path
        eq.EVENT_LOG_PATH = self._old_event_log_path
        self.tmpdir.cleanup()

    def test_parse_dead_letter_threshold_default_and_env(self) -> None:
        old = os.environ.get("EMBED_QUEUE_ALERT_DEAD_LETTER")
        try:
            os.environ["EMBED_QUEUE_ALERT_DEAD_LETTER"] = "7"
            self.assertEqual(eq.parse_dead_letter_threshold(), 7)

            os.environ["EMBED_QUEUE_ALERT_DEAD_LETTER"] = "bad"
            self.assertEqual(eq.parse_dead_letter_threshold(), 5)
        finally:
            if old is None:
                os.environ.pop("EMBED_QUEUE_ALERT_DEAD_LETTER", None)
            else:
                os.environ["EMBED_QUEUE_ALERT_DEAD_LETTER"] = old

    def test_compute_alerts_dead_letter_threshold(self) -> None:
        alerts = eq.compute_alerts({"dead_letter": 3}, dead_letter_threshold=3)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].get("name"), "dead_letter_threshold_exceeded")

        none_alerts = eq.compute_alerts({"dead_letter": 2}, dead_letter_threshold=3)
        self.assertEqual(none_alerts, [])

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

    def test_queue_events_are_emitted_for_lifecycle(self) -> None:
        enq = eq.enqueue_job(kind="file_change", payload={"path": "d.py"}, max_attempts=2)
        job_id = int(enq["job_id"])

        _ = eq.claim_next_job()
        _ = eq.fail_job(job_id, "first failure")

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE embedding_jobs SET next_attempt_at=0 WHERE id=?", (job_id,))
            conn.commit()

        _ = eq.claim_next_job()
        eq.complete_job(job_id)

        events = [
            json.loads(line)
            for line in self.event_log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        names = [e.get("event") for e in events]
        self.assertIn("job_enqueued", names)
        self.assertIn("job_claimed", names)
        self.assertIn("job_failed", names)
        self.assertIn("job_completed", names)


if __name__ == "__main__":
    unittest.main()
