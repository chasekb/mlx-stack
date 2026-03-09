# Local AI Dev Orchestrator (MLX + LiteLLM + Cursor)

This project gives you a practical local orchestration CLI for an Apple Silicon coding stack inspired by your shared chat:

- `MLX-LM` model serving
- `LiteLLM` OpenAI-compatible gateway
- optional `Qdrant` + simple RAG service
- optional agent service
- easy Cursor model configuration output

## What this project does

It provides an `ai-dev` CLI to:

1. Bootstrap stack files and folders (`init`)
2. Start/stop/check Podman Compose services (`up`, `down`, `status`)
3. Generate model conversion commands (`pull-models`)
4. Build a local lightweight code index (`index`)
5. Generate Cursor configuration snippet (`configure-cursor`)
6. Show configured multi-model profiles (`models`)
7. Resolve model routing by task tag (`route-model`)
8. Retrieve repo-aware symbols/chunks for code generation (`retrieve`)
9. Explain git-aware retrieval scoring for transparency (`memory explain`)

## Quick start

### 1) Create a virtual environment and install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2) Initialize the stack files

```bash
ai-dev init
```

### 3) Review generated files

- `podman-compose.yml`
- `litellm_config.yaml`
- `mlx/entrypoint.sh`
- `mlx/Dockerfile`
- `rag/server.py`
- `agent/server.py`
- `.ai-dev/config.json`

### 4) Start services

```bash
ai-dev up
```

### 5) Generate Cursor-compatible API settings

```bash
ai-dev configure-cursor
```

### 6) Build local project index (lightweight)

```bash
ai-dev index .
```

### 7) View configured model profiles

```bash
ai-dev models
```

This milestone enables multi-model configuration through `.ai-dev/config.json`, and `ai-dev init` now regenerates `litellm_config.yaml` from those model profiles.

### 8) Route model selection by task tag

```bash
ai-dev route-model fast
ai-dev route-model quality
ai-dev route-model longctx
```

You can also generate Cursor config using a task tag:

```bash
ai-dev configure-cursor --task-tag fast
```

Supported task tags: `default`, `quality`, `fast`, `longctx`, `analysis`.

### 9) Retrieve repo-aware context

Build the index first:

```bash
ai-dev index .
```

Then query symbols/chunks:

```bash
ai-dev retrieve "model routing for fast tasks"
ai-dev retrieve "agent retrieval" --path-prefix agent/ --top-k 8
```

JSON output is available for service integration:

```bash
ai-dev retrieve "cursor config" --json
```

The agent service also exposes retrieval over HTTP once running:

```bash
curl "http://localhost:8091/retrieve?q=cursor%20config&top_k=5"
```

### 9.1) Automatic incremental indexing (Milestone 4)

You can now run incremental indexing without rebuilding everything each time:

```bash
# one incremental pass (changed files only when possible)
ai-dev index --once .

# keep index fresh in a polling daemon
ai-dev index --daemon --interval 2 .
```

Optional: install git hooks so branch/merge events trigger a safe background reindex:

```bash
ai-dev index --install-git-hooks .
```

This installs/updates `.git/hooks/post-checkout` and `.git/hooks/post-merge` to run:

```bash
python3 -m ai_dev.cli index --once .
```

### 10) Function-calling agent loop (Milestone 3 foundation)

The agent service now exposes a JSON tool schema and task-run endpoint:

```bash
# List available tools and input schemas
curl "http://localhost:8091/tools"

# Run an agent task with a provided plan (dry-run by default)
curl -X POST "http://localhost:8091/agent/run" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Inspect code and suggest improvements",
    "dry_run": true,
    "max_steps": 4,
    "plan": [
      {"tool": "git_diff", "args": {}},
      {"tool": "search_code", "args": {"regex": "def command_", "file_pattern": "*.py"}},
      {"tool": "read_file", "args": {"path": "ai_dev/cli.py", "max_chars": 1200}}
    ]
  }'
```

Run traces are saved to `.ai-dev/runs/<run_id>.json` and can be retrieved via:

```bash
curl "http://localhost:8091/runs/<run_id>"
```

### 11) Prompt caching + metrics (Milestone 5)

The agent service now supports local prompt/result caching for `/agent/run`.

- Cache key includes normalized task payload fields (`task`, `dry_run`, `max_steps`, `plan`, options)
- Cache namespace includes git branch and index signature for invalidation
- TTL is configurable per request (default: 600 seconds)

Example:

```bash
curl -X POST "http://localhost:8091/agent/run" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Inspect code and suggest improvements",
    "dry_run": true,
    "max_steps": 2,
    "plan": [{"tool":"git_diff","args":{}}],
    "cache": {"enabled": true, "ttl_seconds": 600}
  }'
```

Response includes cache metadata:

- `cache.hit` (`false` on miss, `true` on hit)
- `cache.namespace`
- `cache.key`
- `cache.compute_ms`

You can view aggregate cache metrics at:

```bash
curl "http://localhost:8091/metrics"
```

Local cache/metrics files are written under `.ai-dev/`:

- `.ai-dev/prompt_cache.json`
- `.ai-dev/metrics.json`

### 12) Speculative decode foundation (Milestone 6)

This repo now includes a local `spec-router` service (optional profile) that can run
a draft-vs-target token acceptance loop via HTTP.

Start optional services (includes `spec-router`):

```bash
ai-dev up --with-optional
```

Health check:

```bash
curl "http://localhost:8092/health"
```

Direct decode call:

```bash
curl -X POST "http://localhost:8092/spec/decode" \
  -H "Content-Type: application/json" \
  -d '{
    "draft_tokens": ["def", "hello", "("],
    "target_tokens": ["def", "hello", "(name)"]
  }'
```

CLI helper:

```bash
ai-dev spec-decode \
  --draft-tokens def hello "(" \
  --target-tokens def hello "(name)"
```

Prompt-driven draft/target mode (model-backed):

```bash
ai-dev spec-decode \
  --prompt "Write a Python function to add two numbers" \
  --draft-model local-mlx-fast \
  --target-model local-mlx \
  --draft-url http://localhost:4000/v1/completions \
  --target-url http://localhost:4000/v1/completions \
  --max-tokens 96
```

In prompt mode, `spec-router` calls both draft and target model endpoints, compares tokenized outputs,
and reports acceptance stats plus per-call timing (`draft_call_ms`, `target_call_ms`).

Streaming acceptance loop mode (default for prompt-based speculative decode):

```bash
ai-dev spec-decode \
  --prompt "Write a Python function to add two numbers" \
  --draft-model local-mlx-fast \
  --target-model local-mlx \
  --max-tokens 32
```

In streaming acceptance mode, `spec-router` performs iterative one-token draft/target calls and returns:

- `loop_mode: "stream_acceptance"`
- per-step acceptance trace in `steps`
- `accepted_tokens`, `rejected_tokens`, `compared_tokens`
- accumulated `draft_call_ms` and `target_call_ms`

Compatibility mode remains available by setting `stream_loop: false` in the request body,
which uses the previous full-completion compare behavior.

### 13) Background embedding workers (Milestone 7)

This repo now includes a lightweight background embedding pipeline:

- `embed-queue` service (`embedding_queue/server.py`) using SQLite-backed jobs
- `embed-worker` service (`embedding_worker/worker.py`) that polls jobs, requests real embeddings, and writes vectors to `.ai-dev/embeddings.jsonl`
- retry + dead-letter behavior in the queue API
- optional Qdrant upsert path with non-fatal fallback
- embedding schema/version tracking and migration guardrails
- CLI helpers:
  - `ai-dev embed-enqueue`
  - `ai-dev embed-stats`

Start optional services:

```bash
ai-dev up --with-optional
```

Enqueue a job:

```bash
ai-dev embed-enqueue --path ai_dev/cli.py --kind file_change --text "def main parser"
```

Check queue stats:

```bash
ai-dev embed-stats
```

Run worker once manually (optional local debug):

```bash
python3 embedding_worker/worker.py --queue-url http://localhost:8093 --once
```

Productionization options:

```bash
python3 embedding_worker/worker.py \
  --queue-url http://localhost:8093 \
  --embed-url http://localhost:4000/v1/embeddings \
  --embed-model local-embed \
  --qdrant-url http://localhost:6333 \
  --qdrant-collection ai_dev_embeddings \
  --allow-schema-migrate
```

Useful flags:

- `--disable-qdrant` to run JSONL-only mode
- `--force-fake-embed` to force deterministic fallback vectors (debug/safe mode)
- `--schema-path` and `--migration-log-path` to customize schema/migration files

Worker metadata files:

- `.ai-dev/embedding_schema.json`
- `.ai-dev/embedding_migrations.jsonl`

### 14) Git-aware code memory (Milestone 8)

Retrieval now includes git-aware ranking signals in addition to lexical matching:

- current branch match bias
- recent commit recency bias
- existing changed-file and path-prefix bias

Run retrieval as usual:

```bash
ai-dev retrieve "agent retrieval cache"
```

Use transparent scoring output:

```bash
ai-dev memory explain "agent retrieval cache"
ai-dev memory explain "agent retrieval cache" --json
```

The explain command returns per-result score components (`score_breakdown`) for:

- lexical match
- path prefix
- changed file
- branch match
- recency

### 15) Shared KV cache (Milestone 9)

The agent service now includes a lightweight shared KV-cache simulation for
session-aware prefix reuse across related requests.

Key capabilities:

- tenant/session/model isolation
- prefix-boundary validation (reuse only when new prefix extends prior prefix)
- prefix hash validation (`prefix_hash`)
- per-model token budget with LRU-style eviction

`POST /agent/run` now accepts optional `kv_cache` settings:

```json
{
  "task": "Refine implementation",
  "model": "local-mlx",
  "dry_run": true,
  "plan": [{"tool":"git_diff","args":{}}],
  "kv_cache": {
    "enabled": true,
    "tenant_id": "acme",
    "session_id": "sess-123",
    "prompt_prefix": "Analyze current diff and suggest next edits",
    "model_budget_tokens": 8000,
    "entry_max_tokens": 2048
  }
}
```

Response includes a `kv_cache` block with status and reason:

- `hit` (prefix extension reuse)
- `miss` (cold start or boundary mismatch)
- `bypass` (missing session/prefix)
- `rejected` (hash mismatch)

Metrics now include KV summaries:

```bash
curl "http://localhost:8091/metrics"
```

Local KV state is stored in:

- `.ai-dev/kv_cache.json`

### 16) Observability events and alert thresholds (D residual hardening)

Recent hardening adds structured event logs and configurable alerts across core services.

Structured event logs:

- Agent service events:
  - path: `.ai-dev/events/agent.jsonl`
  - examples: `run_started`, `run_completed`, `alerts_emitted`
- Embedding queue events:
  - path: `.ai-dev/events/embed-queue.jsonl`
  - examples: `job_enqueued`, `job_claimed`, `job_failed`, `job_completed`, `alerts_emitted`
- Embedding worker events:
  - path: `.ai-dev/events/embed-worker.jsonl`
  - examples: `job_processing_started`, `job_processing_completed`, `job_processing_failed`, `job_marked_done`, `qdrant_upsert_succeeded`, `qdrant_upsert_failed`

Alert thresholds (environment variables):

- Agent (`GET /metrics`):
  - `AGENT_ALERT_TOOL_ERRORS` (default `5`)
  - `AGENT_ALERT_CACHE_HIT_RATE_MIN` (default `0.2`)
- Embedding queue (`GET /stats`):
  - `EMBED_QUEUE_ALERT_DEAD_LETTER` (default `5`)

When thresholds are crossed, API responses include:

- `alerts`: active warning entries
- `alert_thresholds`: currently applied threshold values

Example:

```bash
AGENT_ALERT_TOOL_ERRORS=3 AGENT_ALERT_CACHE_HIT_RATE_MIN=0.25 python3 agent/server.py
EMBED_QUEUE_ALERT_DEAD_LETTER=2 python3 embedding_queue/server.py
```

Then inspect:

```bash
curl "http://localhost:8091/metrics"
curl "http://localhost:8093/stats"
```

## Notes

- This is a developer-friendly starter orchestration layer, intentionally lightweight.
- The indexing command uses a local JSON-based lexical index (no cloud).
- You can swap in full embeddings/RAG later (Qdrant + LlamaIndex) while keeping the CLI workflow.

## Refactor progress (F slice)

Recent modularization progress extracted shared CLI internals into `ai_dev/core/`:

- `ai_dev/core/indexing.py` (source-file iteration, symbol extraction, chunk building)
- `ai_dev/core/retrieval.py` (tokenization, recency utilities, symbol/chunk scoring)

`ai_dev/cli.py` now delegates to these modules for indexing/retrieval helper behavior,
which keeps command behavior stable while reducing coupling in the main CLI module.

## CI Build (GitHub Actions + local)

This repository includes `.github/workflows/build.yml` with a remote build smoke test on:

- pushes to `main`
- pull requests targeting `main`
- manual `workflow_dispatch`

### Run the same build steps locally

#### Option A: plain shell (same commands as CI)

```bash
python3 -m pip install --upgrade pip
pip3 install -e .
ai-dev --help
ai-dev init
ai-dev configure-cursor
ai-dev index .
```

#### Option B: run the GitHub Action locally with `act`

```bash
brew install act
act -j build
```

> Note: `act` runs workflows in containers and may pull runner images on first run.
