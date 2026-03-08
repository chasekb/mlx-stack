from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import embedding_worker.worker as ew


class TestEmbeddingWorker(unittest.TestCase):
    def test_process_job_uses_http_embedding_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / 'embeddings.jsonl'
            schema_path = Path(tmp) / 'embedding_schema.json'
            migration_path = Path(tmp) / 'embedding_migrations.jsonl'

            original_embed = ew._embed_via_http
            try:
                ew._embed_via_http = lambda *a, **k: [0.1, 0.2, 0.3]
                ew.process_job(
                    {'id': 1, 'kind': 'file_change', 'payload': {'text': 'hello world'}},
                    output_path=out_path,
                    schema_path=schema_path,
                    migration_log_path=migration_path,
                    embed_url='http://example.invalid/v1/embeddings',
                    embed_model='local-embed-model',
                    timeout=1.0,
                    qdrant_url='http://localhost:6333',
                    qdrant_collection='ai_dev_embeddings',
                    qdrant_enabled=False,
                    allow_schema_migrate=False,
                    force_fake_embed=False,
                )
            finally:
                ew._embed_via_http = original_embed

            line = out_path.read_text(encoding='utf-8').strip().splitlines()[-1]
            rec = json.loads(line)
            self.assertEqual(rec['vector_backend'], 'local_http')
            self.assertEqual(rec['vector_dim'], 3)
            self.assertEqual(rec['schema_version'], ew.EMBEDDING_SCHEMA_VERSION)

            schema = json.loads(schema_path.read_text(encoding='utf-8'))
            self.assertEqual(schema['embedding_model'], 'local-embed-model')
            self.assertEqual(int(schema['vector_dim']), 3)

    def test_process_job_fallback_and_qdrant_error_is_nonfatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / 'embeddings.jsonl'
            schema_path = Path(tmp) / 'embedding_schema.json'
            migration_path = Path(tmp) / 'embedding_migrations.jsonl'

            original_embed = ew._embed_via_http
            original_qdrant_upsert = ew.qdrant_upsert
            try:
                def _raise_embed(*args, **kwargs):
                    raise RuntimeError('embed backend unavailable')

                def _raise_qdrant(*args, **kwargs):
                    raise RuntimeError('qdrant unavailable')

                ew._embed_via_http = _raise_embed
                ew.qdrant_upsert = _raise_qdrant

                ew.process_job(
                    {'id': 2, 'kind': 'file_change', 'payload': {'text': 'fallback me'}},
                    output_path=out_path,
                    schema_path=schema_path,
                    migration_log_path=migration_path,
                    embed_url='http://example.invalid/v1/embeddings',
                    embed_model='local-embed-model',
                    timeout=1.0,
                    qdrant_url='http://localhost:6333',
                    qdrant_collection='ai_dev_embeddings',
                    qdrant_enabled=True,
                    allow_schema_migrate=False,
                    force_fake_embed=False,
                )
            finally:
                ew._embed_via_http = original_embed
                ew.qdrant_upsert = original_qdrant_upsert

            rec = json.loads(out_path.read_text(encoding='utf-8').strip().splitlines()[-1])
            self.assertEqual(rec['vector_backend'], 'deterministic_fallback')
            self.assertEqual(rec['qdrant']['enabled'], True)
            self.assertEqual(rec['qdrant']['upserted'], False)

    def test_schema_mismatch_requires_migrate_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / 'embeddings.jsonl'
            schema_path = Path(tmp) / 'embedding_schema.json'
            migration_path = Path(tmp) / 'embedding_migrations.jsonl'

            schema_path.write_text(
                json.dumps(
                    {
                        'schema_version': ew.EMBEDDING_SCHEMA_VERSION,
                        'embedding_model': 'old-model',
                        'vector_dim': 16,
                        'backend': 'deterministic_fallback',
                    }
                ),
                encoding='utf-8',
            )

            with self.assertRaises(RuntimeError):
                ew.ensure_embedding_schema(
                    schema_path=schema_path,
                    output_path=out_path,
                    migration_log_path=migration_path,
                    embedding_model='new-model',
                    vector_dim=16,
                    backend='deterministic_fallback',
                    allow_migrate=False,
                )

    def test_schema_migration_rotates_existing_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / 'embeddings.jsonl'
            schema_path = Path(tmp) / 'embedding_schema.json'
            migration_path = Path(tmp) / 'embedding_migrations.jsonl'

            out_path.write_text('{"old":true}\n', encoding='utf-8')
            schema_path.write_text(
                json.dumps(
                    {
                        'schema_version': ew.EMBEDDING_SCHEMA_VERSION,
                        'embedding_model': 'old-model',
                        'vector_dim': 16,
                        'backend': 'deterministic_fallback',
                    }
                ),
                encoding='utf-8',
            )

            out = ew.ensure_embedding_schema(
                schema_path=schema_path,
                output_path=out_path,
                migration_log_path=migration_path,
                embedding_model='new-model',
                vector_dim=16,
                backend='deterministic_fallback',
                allow_migrate=True,
            )
            self.assertEqual(out['embedding_model'], 'new-model')
            backups = list(Path(tmp).glob('embeddings.jsonl.migrated-*.bak'))
            self.assertEqual(len(backups), 1)
            self.assertTrue(migration_path.exists())


if __name__ == '__main__':
    unittest.main()
