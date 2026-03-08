from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer


DB_PATH = Path('.ai-dev/embedding_jobs.db')


def _db_connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _db_connect() as conn:
        conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS embedding_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                max_attempts INTEGER NOT NULL DEFAULT 3,
                next_attempt_at REAL NOT NULL DEFAULT 0,
                last_error TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            '''
        )
        conn.commit()


def enqueue_job(kind: str, payload: dict, max_attempts: int = 3) -> dict:
    now = time.time()
    with _db_connect() as conn:
        cur = conn.execute(
            '''
            INSERT INTO embedding_jobs (
                kind, payload_json, status, attempts, max_attempts, next_attempt_at, created_at, updated_at
            ) VALUES (?, ?, 'queued', 0, ?, 0, ?, ?)
            ''',
            (kind, json.dumps(payload), max(1, int(max_attempts)), now, now),
        )
        conn.commit()
        return {'job_id': int(cur.lastrowid), 'status': 'queued'}


def claim_next_job() -> dict | None:
    now = time.time()
    with _db_connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        row = conn.execute(
            '''
            SELECT * FROM embedding_jobs
            WHERE status IN ('queued', 'retry')
              AND next_attempt_at <= ?
            ORDER BY id ASC
            LIMIT 1
            ''',
            (now,),
        ).fetchone()

        if row is None:
            conn.commit()
            return None

        next_attempts = int(row['attempts']) + 1
        conn.execute(
            '''
            UPDATE embedding_jobs
            SET status='in_progress', attempts=?, updated_at=?
            WHERE id=?
            ''',
            (next_attempts, now, int(row['id'])),
        )
        conn.commit()

        return {
            'id': int(row['id']),
            'kind': row['kind'],
            'payload': json.loads(row['payload_json']),
            'attempts': next_attempts,
            'max_attempts': int(row['max_attempts']),
        }


def complete_job(job_id: int) -> None:
    now = time.time()
    with _db_connect() as conn:
        conn.execute(
            "UPDATE embedding_jobs SET status='done', updated_at=? WHERE id=?",
            (now, int(job_id)),
        )
        conn.commit()


def fail_job(job_id: int, error: str) -> dict:
    now = time.time()
    with _db_connect() as conn:
        row = conn.execute(
            'SELECT attempts, max_attempts FROM embedding_jobs WHERE id=?',
            (int(job_id),),
        ).fetchone()
        if row is None:
            return {'error': 'job_not_found'}

        attempts = int(row['attempts'])
        max_attempts = int(row['max_attempts'])
        if attempts >= max_attempts:
            status = 'dead_letter'
            next_attempt_at = 0
        else:
            status = 'retry'
            next_attempt_at = now + min(60.0, float(2 ** attempts))

        conn.execute(
            '''
            UPDATE embedding_jobs
            SET status=?, next_attempt_at=?, last_error=?, updated_at=?
            WHERE id=?
            ''',
            (status, next_attempt_at, str(error)[:2000], now, int(job_id)),
        )
        conn.commit()
        return {'ok': True, 'status': status, 'next_attempt_at': next_attempt_at}


def get_stats() -> dict:
    with _db_connect() as conn:
        counts = {
            row['status']: int(row['count'])
            for row in conn.execute(
                'SELECT status, COUNT(*) AS count FROM embedding_jobs GROUP BY status'
            ).fetchall()
        }
    return {
        'queued': counts.get('queued', 0),
        'retry': counts.get('retry', 0),
        'in_progress': counts.get('in_progress', 0),
        'done': counts.get('done', 0),
        'dead_letter': counts.get('dead_letter', 0),
    }


class Handler(BaseHTTPRequestHandler):
    def _reply(self, payload: dict, status: int = 200) -> None:
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode('utf-8'))

    def _read_json(self) -> dict:
        try:
            n = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            n = 0
        body = self.rfile.read(max(0, n))
        if not body:
            return {}
        try:
            parsed = json.loads(body.decode('utf-8'))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def do_GET(self):
        if self.path == '/health':
            self._reply({'ok': True, 'service': 'embed-queue', 'db_path': str(DB_PATH)})
            return
        if self.path == '/stats':
            self._reply({'ok': True, 'service': 'embed-queue', 'stats': get_stats()})
            return
        self._reply({'error': 'not found'}, status=404)

    def do_POST(self):
        if self.path == '/jobs/enqueue':
            payload = self._read_json()
            kind = str(payload.get('kind', 'file_change') or 'file_change')
            job_payload = payload.get('payload', {}) if isinstance(payload.get('payload', {}), dict) else {}
            max_attempts = int(payload.get('max_attempts', 3) or 3)
            out = enqueue_job(kind=kind, payload=job_payload, max_attempts=max_attempts)
            self._reply({'ok': True, 'service': 'embed-queue', **out})
            return

        if self.path == '/jobs/claim':
            job = claim_next_job()
            self._reply({'ok': True, 'service': 'embed-queue', 'job': job})
            return

        if self.path == '/jobs/complete':
            payload = self._read_json()
            job_id = int(payload.get('job_id', 0) or 0)
            if job_id <= 0:
                self._reply({'error': 'missing_job_id'}, status=400)
                return
            complete_job(job_id)
            self._reply({'ok': True, 'service': 'embed-queue', 'job_id': job_id, 'status': 'done'})
            return

        if self.path == '/jobs/fail':
            payload = self._read_json()
            job_id = int(payload.get('job_id', 0) or 0)
            if job_id <= 0:
                self._reply({'error': 'missing_job_id'}, status=400)
                return
            error = str(payload.get('error', 'unknown_error'))
            out = fail_job(job_id=job_id, error=error)
            if out.get('error'):
                self._reply(out, status=404)
                return
            self._reply({'ok': True, 'service': 'embed-queue', 'job_id': job_id, **out})
            return

        self._reply({'error': 'not found'}, status=404)


if __name__ == '__main__':
    _init_db()
    server = HTTPServer(('0.0.0.0', 8093), Handler)
    print('Embedding queue listening on :8093')
    server.serve_forever()
