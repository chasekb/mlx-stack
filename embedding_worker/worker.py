from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def http_json(method: str, url: str, payload: dict | None = None, timeout: float = 10.0) -> dict:
    data = json.dumps(payload or {}).encode('utf-8') if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={'Content-Type': 'application/json'},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode('utf-8')
    return json.loads(raw) if raw else {}


def fake_embed(text: str, dims: int = 16) -> list[float]:
    digest = hashlib.sha256((text or '').encode('utf-8')).digest()
    vals = [((digest[i % len(digest)] / 255.0) * 2.0 - 1.0) for i in range(dims)]
    norm = math.sqrt(sum(v * v for v in vals)) or 1.0
    return [round(v / norm, 6) for v in vals]


def process_job(job: dict, output_path: Path) -> None:
    payload = job.get('payload', {}) if isinstance(job.get('payload', {}), dict) else {}
    source_text = str(payload.get('text', '') or '')
    if not source_text:
        source_text = str(payload.get('path', '') or '')
    if not source_text:
        source_text = json.dumps(payload, sort_keys=True)

    rec = {
        'embedded_at': datetime.now(timezone.utc).isoformat(),
        'job_id': int(job.get('id', 0) or 0),
        'kind': str(job.get('kind', 'unknown')),
        'metadata': payload,
        'vector': fake_embed(source_text),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(rec) + '\n')


def run_once(queue_url: str, output_path: Path, timeout: float) -> bool:
    claim = http_json('POST', queue_url.rstrip('/') + '/jobs/claim', payload={}, timeout=timeout)
    job = claim.get('job') if isinstance(claim, dict) else None
    if not isinstance(job, dict):
        return False

    job_id = int(job.get('id', 0) or 0)
    if job_id <= 0:
        return False

    try:
        process_job(job=job, output_path=output_path)
    except Exception as e:
        http_json(
            'POST',
            queue_url.rstrip('/') + '/jobs/fail',
            payload={'job_id': job_id, 'error': str(e)},
            timeout=timeout,
        )
        return True

    http_json(
        'POST',
        queue_url.rstrip('/') + '/jobs/complete',
        payload={'job_id': job_id},
        timeout=timeout,
    )
    return True


def main() -> int:
    p = argparse.ArgumentParser(description='Background embedding worker')
    p.add_argument('--queue-url', default='http://localhost:8093')
    p.add_argument('--output-path', default='.ai-dev/embeddings.jsonl')
    p.add_argument('--poll-interval', type=float, default=2.0)
    p.add_argument('--timeout', type=float, default=10.0)
    p.add_argument('--once', action='store_true', help='Process at most one available job and exit')
    args = p.parse_args()

    output_path = Path(args.output_path)

    if args.once:
        try:
            run_once(queue_url=args.queue_url, output_path=output_path, timeout=args.timeout)
            return 0
        except urllib.error.URLError as e:
            print(f'worker failed to reach queue: {e}')
            return 2

    print(f'Embedding worker polling {args.queue_url} every {args.poll_interval}s')
    while True:
        try:
            processed = run_once(queue_url=args.queue_url, output_path=output_path, timeout=args.timeout)
        except urllib.error.URLError as e:
            print(f'worker queue error: {e}')
            processed = False

        if not processed:
            time.sleep(max(0.25, args.poll_interval))


if __name__ == '__main__':
    raise SystemExit(main())
