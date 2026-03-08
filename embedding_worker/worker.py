from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


EMBEDDING_SCHEMA_VERSION = 2
EVENT_LOG_PATH = Path('.ai-dev/events/embed-worker.jsonl')


def emit_event(event_type: str, **fields: object) -> None:
    EVENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        'ts': time.time(),
        'service': 'embed-worker',
        'event': event_type,
        **fields,
    }
    with EVENT_LOG_PATH.open('a', encoding='utf-8') as f:
        f.write(json.dumps(rec) + '\n')


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


def _coerce_vector(vec: object) -> list[float]:
    if not isinstance(vec, list):
        return []
    out: list[float] = []
    for x in vec:
        try:
            out.append(float(x))
        except (TypeError, ValueError):
            return []
    return out


def _extract_source_text(payload: dict) -> str:
    source_text = str(payload.get('text', '') or '')
    if not source_text:
        source_text = str(payload.get('path', '') or '')
    if not source_text:
        source_text = json.dumps(payload, sort_keys=True)
    return source_text


def _embed_via_http(text: str, embed_url: str, embed_model: str, timeout: float) -> list[float]:
    body = {
        'model': embed_model,
        'input': text,
    }
    resp = http_json('POST', embed_url, payload=body, timeout=timeout)
    data = resp.get('data', []) if isinstance(resp, dict) else []
    if not isinstance(data, list) or not data:
        raise ValueError('missing_embedding_data')
    first = data[0] if isinstance(data[0], dict) else {}
    vec = _coerce_vector(first.get('embedding'))
    if not vec:
        raise ValueError('invalid_embedding_vector')
    return vec


def _load_schema(schema_path: Path) -> dict:
    if not schema_path.exists():
        return {}
    try:
        parsed = json.loads(schema_path.read_text(encoding='utf-8'))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _save_schema(schema_path: Path, schema: dict) -> None:
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(json.dumps(schema, indent=2) + '\n', encoding='utf-8')


def _append_migration_event(migration_log_path: Path, event: dict) -> None:
    migration_log_path.parent.mkdir(parents=True, exist_ok=True)
    with migration_log_path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(event) + '\n')


def _rotate_output_for_migration(output_path: Path, old_schema: dict, migration_log_path: Path) -> None:
    if not output_path.exists() or not output_path.is_file():
        return
    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    rotated = output_path.with_suffix(output_path.suffix + f'.migrated-{ts}.bak')
    shutil.move(str(output_path), str(rotated))
    _append_migration_event(
        migration_log_path,
        {
            'event': 'embedding_schema_migration',
            'migrated_at': datetime.now(timezone.utc).isoformat(),
            'old_schema': old_schema,
            'rotated_output_path': str(rotated),
        },
    )


def ensure_embedding_schema(
    schema_path: Path,
    output_path: Path,
    migration_log_path: Path,
    *,
    embedding_model: str,
    vector_dim: int,
    backend: str,
    allow_migrate: bool,
) -> dict:
    now_iso = datetime.now(timezone.utc).isoformat()
    expected = {
        'schema_version': EMBEDDING_SCHEMA_VERSION,
        'embedding_model': embedding_model,
        'vector_dim': int(vector_dim),
        'backend': backend,
    }
    current = _load_schema(schema_path)
    if not current:
        out = {**expected, 'created_at': now_iso, 'updated_at': now_iso}
        _save_schema(schema_path, out)
        return out

    compatible = (
        int(current.get('schema_version', 0) or 0) == expected['schema_version']
        and str(current.get('embedding_model', '')) == expected['embedding_model']
        and int(current.get('vector_dim', 0) or 0) == expected['vector_dim']
        and str(current.get('backend', '')) == expected['backend']
    )
    if compatible:
        current['updated_at'] = now_iso
        _save_schema(schema_path, current)
        return current

    if not allow_migrate:
        raise RuntimeError(
            'embedding_schema_mismatch: pass --allow-schema-migrate to rotate old embeddings and continue'
        )

    _rotate_output_for_migration(output_path=output_path, old_schema=current, migration_log_path=migration_log_path)
    out = {**expected, 'created_at': now_iso, 'updated_at': now_iso}
    _save_schema(schema_path, out)
    _append_migration_event(
        migration_log_path,
        {
            'event': 'embedding_schema_initialized',
            'initialized_at': now_iso,
            'schema': out,
        },
    )
    return out


def qdrant_upsert(
    *,
    qdrant_url: str,
    collection: str,
    point_id: int,
    vector: list[float],
    payload: dict,
    timeout: float,
) -> dict:
    base = qdrant_url.rstrip('/')
    coll_url = f'{base}/collections/{collection}'
    try:
        http_json('GET', coll_url, timeout=timeout)
    except Exception:
        http_json(
            'PUT',
            coll_url,
            payload={'vectors': {'size': len(vector), 'distance': 'Cosine'}},
            timeout=timeout,
        )

    return http_json(
        'PUT',
        f'{coll_url}/points?wait=true',
        payload={
            'points': [
                {
                    'id': int(point_id),
                    'vector': vector,
                    'payload': payload,
                }
            ]
        },
        timeout=timeout,
    )


def process_job(
    job: dict,
    output_path: Path,
    schema_path: Path,
    migration_log_path: Path,
    *,
    embed_url: str,
    embed_model: str,
    timeout: float,
    qdrant_url: str,
    qdrant_collection: str,
    qdrant_enabled: bool,
    allow_schema_migrate: bool,
    force_fake_embed: bool,
) -> None:
    emit_event('job_processing_started', job_id=int(job.get('id', 0) or 0), kind=str(job.get('kind', 'unknown')))
    payload = job.get('payload', {}) if isinstance(job.get('payload', {}), dict) else {}
    source_text = _extract_source_text(payload)

    vector_backend = 'local_http'
    if force_fake_embed:
        vector = fake_embed(source_text)
        vector_backend = 'deterministic_fallback'
    else:
        try:
            vector = _embed_via_http(source_text, embed_url=embed_url, embed_model=embed_model, timeout=timeout)
        except Exception:
            vector = fake_embed(source_text)
            vector_backend = 'deterministic_fallback'

    schema = ensure_embedding_schema(
        schema_path=schema_path,
        output_path=output_path,
        migration_log_path=migration_log_path,
        embedding_model=embed_model,
        vector_dim=len(vector),
        backend=vector_backend,
        allow_migrate=allow_schema_migrate,
    )

    qdrant_status = {'enabled': qdrant_enabled, 'upserted': False}
    if qdrant_enabled:
        try:
            qdrant_upsert(
                qdrant_url=qdrant_url,
                collection=qdrant_collection,
                point_id=int(job.get('id', 0) or 0),
                vector=vector,
                payload={
                    'kind': str(job.get('kind', 'unknown')),
                    'metadata': payload,
                    'schema_version': EMBEDDING_SCHEMA_VERSION,
                    'embedding_model': embed_model,
                    'vector_backend': vector_backend,
                },
                timeout=timeout,
            )
            qdrant_status = {'enabled': True, 'upserted': True}
            emit_event('qdrant_upsert_succeeded', job_id=int(job.get('id', 0) or 0), collection=qdrant_collection)
        except Exception as e:
            qdrant_status = {'enabled': True, 'upserted': False, 'error': str(e)[:500]}
            emit_event('qdrant_upsert_failed', job_id=int(job.get('id', 0) or 0), error=str(e)[:300])

    rec = {
        'embedded_at': datetime.now(timezone.utc).isoformat(),
        'schema_version': EMBEDDING_SCHEMA_VERSION,
        'job_id': int(job.get('id', 0) or 0),
        'kind': str(job.get('kind', 'unknown')),
        'embedding_model': embed_model,
        'vector_backend': vector_backend,
        'vector_dim': len(vector),
        'schema': {
            'embedding_model': schema.get('embedding_model', ''),
            'vector_dim': int(schema.get('vector_dim', 0) or 0),
            'backend': schema.get('backend', ''),
        },
        'qdrant': qdrant_status,
        'metadata': payload,
        'vector': vector,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(rec) + '\n')
    emit_event(
        'job_processing_completed',
        job_id=int(job.get('id', 0) or 0),
        vector_backend=vector_backend,
        vector_dim=len(vector),
        qdrant_upserted=bool(qdrant_status.get('upserted', False)),
    )


def run_once(queue_url: str, output_path: Path, timeout: float) -> bool:
    claim = http_json('POST', queue_url.rstrip('/') + '/jobs/claim', payload={}, timeout=timeout)
    job = claim.get('job') if isinstance(claim, dict) else None
    if not isinstance(job, dict):
        return False

    job_id = int(job.get('id', 0) or 0)
    if job_id <= 0:
        return False

    try:
        process_job(
            job=job,
            output_path=output_path,
            schema_path=Path(os.environ.get('EMBED_SCHEMA_PATH', '.ai-dev/embedding_schema.json')),
            migration_log_path=Path(os.environ.get('EMBED_MIGRATION_LOG_PATH', '.ai-dev/embedding_migrations.jsonl')),
            embed_url=os.environ.get('EMBED_URL', 'http://localhost:4000/v1/embeddings'),
            embed_model=os.environ.get('EMBED_MODEL', 'local-embed'),
            timeout=timeout,
            qdrant_url=os.environ.get('QDRANT_URL', 'http://localhost:6333'),
            qdrant_collection=os.environ.get('QDRANT_COLLECTION', 'ai_dev_embeddings'),
            qdrant_enabled=os.environ.get('QDRANT_ENABLED', '1') not in ('0', 'false', 'False'),
            allow_schema_migrate=os.environ.get('ALLOW_SCHEMA_MIGRATE', '0') in ('1', 'true', 'True'),
            force_fake_embed=os.environ.get('FORCE_FAKE_EMBED', '0') in ('1', 'true', 'True'),
        )
    except Exception as e:
        emit_event('job_processing_failed', job_id=job_id, error=str(e)[:300])
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
    emit_event('job_marked_done', job_id=job_id)
    return True


def main() -> int:
    p = argparse.ArgumentParser(description='Background embedding worker')
    p.add_argument('--queue-url', default='http://localhost:8093')
    p.add_argument('--output-path', default='.ai-dev/embeddings.jsonl')
    p.add_argument('--poll-interval', type=float, default=2.0)
    p.add_argument('--timeout', type=float, default=10.0)
    p.add_argument('--embed-url', default='http://localhost:4000/v1/embeddings')
    p.add_argument('--embed-model', default='local-embed')
    p.add_argument('--schema-path', default='.ai-dev/embedding_schema.json')
    p.add_argument('--migration-log-path', default='.ai-dev/embedding_migrations.jsonl')
    p.add_argument('--qdrant-url', default='http://localhost:6333')
    p.add_argument('--qdrant-collection', default='ai_dev_embeddings')
    p.add_argument('--disable-qdrant', action='store_true')
    p.add_argument('--allow-schema-migrate', action='store_true')
    p.add_argument('--force-fake-embed', action='store_true')
    p.add_argument('--once', action='store_true', help='Process at most one available job and exit')
    args = p.parse_args()

    os.environ['EMBED_URL'] = str(args.embed_url)
    os.environ['EMBED_MODEL'] = str(args.embed_model)
    os.environ['EMBED_SCHEMA_PATH'] = str(args.schema_path)
    os.environ['EMBED_MIGRATION_LOG_PATH'] = str(args.migration_log_path)
    os.environ['QDRANT_URL'] = str(args.qdrant_url)
    os.environ['QDRANT_COLLECTION'] = str(args.qdrant_collection)
    os.environ['QDRANT_ENABLED'] = '0' if args.disable_qdrant else '1'
    os.environ['ALLOW_SCHEMA_MIGRATE'] = '1' if args.allow_schema_migrate else '0'
    os.environ['FORCE_FAKE_EMBED'] = '1' if args.force_fake_embed else '0'

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
