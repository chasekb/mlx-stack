# From Current Stack to Advanced Local AI Platform

This roadmap starts from your current working baseline (`ai-dev`, MLX container, LiteLLM gateway, basic indexing) and moves toward:

- multi-model inference
- repo-aware code generation
- function-calling agents
- automatic repo indexing
- prompt caching
- speculative decoding
- background embedding workers
- git-aware code memory
- shared KV cache

## Milestone status

- [x] 0) Current baseline
- [x] 1) Multi-model inference
- [x] 2) Repo-aware code generation
- [x] 3) Function-calling agents
- [x] 4) Automatic repo indexing
- [x] 5) Prompt caching
- [x] 6) Speculative decoding
- [x] 7) Background embedding workers
- [x] 8) Git-aware code memory
- [x] 9) Shared KV cache

---

## 0) Current baseline (already true)

- `ai-dev init/up/down/status/index/configure-cursor` exists.
- MLX + LiteLLM services are defined.
- CI smoke workflow runs in GitHub Actions.

This is enough to start iterating feature-by-feature.

---

## 1) Multi-model inference (first milestone)

## Goal
Serve and route multiple local models through LiteLLM.

## Implementation

1. **Add model profiles to `.ai-dev/config.json`**
   - e.g. `coder_fast`, `coder_strong`, `analysis_longctx`.
2. **Extend `litellm_config.yaml` generation** in `ai_dev/cli.py`
   - emit multiple `model_list` entries.
3. **Add `ai-dev models` command**
   - list available profiles and endpoints.
4. **Add simple route policy**
   - choose model by task tags (`fast`, `quality`, `longctx`).

## Done when
- You can call different models via one OpenAI-compatible endpoint and switch by model name.

### Completion notes

- Added model profiles in `.ai-dev/config.json` schema (`ai_dev/cli.py`).
- `ai-dev init` now generates `litellm_config.yaml` from configured model profiles.
- Added `ai-dev models` to list model profiles.
- Added tag-based route policy with `ai-dev route-model <task_tag>`.
- Added `ai-dev configure-cursor --task-tag <tag>` to select routed model for Cursor settings.

---

## 2) Repo-aware code generation

## Goal
Provide contextual code generation grounded in project files.

## Implementation

1. Replace lexical-only index with a two-layer index:
   - **symbol index** (functions/classes/exports)
   - **chunk index** (semantic chunks + metadata).
2. Add `ai-dev retrieve` command:
   - query → top-k chunks/symbols.
3. Add retrieval hook in agent flow:
   - inject top context into prompts before generation.
4. Add path filters:
   - prioritize changed files and relevant directories.

## Done when
- Prompted coding tasks reliably include accurate local code context.

### Completion notes

- Upgraded `ai-dev index` to produce a two-layer index:
  - `symbols` (function/class signatures with file+line)
  - `chunks` (line-bounded chunks with terms and previews)
- Added `ai-dev retrieve <query>` with:
  - top-k symbol/chunk ranking
  - `--path-prefix` filters
  - git-changed-file bias (optional disable via `--no-changed-bias`)
  - JSON output for service integration
- Added retrieval hook endpoint to agent service:
  - `GET /retrieve?q=...&top_k=...&path_prefix=...`
  - reads `.ai-dev/index.json` and returns ranked retrieval payload

---

## 3) Function-calling agents

## Goal
Agent can plan, call tools, and apply edits safely.

## Implementation

1. Upgrade `agent/server.py` into a JSON-RPC style tool loop.
2. Define tool schema set:
   - `search_code`, `read_file`, `write_patch`, `run_tests`, `git_diff`, `commit_changes`.
3. Add execution guardrails:
   - dry-run mode
   - allowlist commands
   - max-step budget.
4. Add trace logging to `.ai-dev/runs/<id>.json`.

## Done when
- Agent can complete controlled code tasks with auditable tool traces.

### Completion notes

- Upgraded `agent/server.py` with a JSON-style function-calling loop endpoint:
  - `POST /agent/run` accepts `{ task, dry_run, max_steps, plan[] }`
- Added tool schema discovery endpoint:
  - `GET /tools` returns available tools and input contracts.
- Implemented initial tool set with guardrails:
  - `retrieve`, `search_code`, `read_file`, `git_diff`, `run_tests`, `write_patch`, `commit_changes`
  - allowlist enforced (`ALLOWED_TOOLS`)
  - `dry_run` blocks mutating tools (`write_patch`, `commit_changes`)
  - bounded `max_steps` execution budget
- Added run trace persistence for auditability:
  - traces written to `.ai-dev/runs/<run_id>.json`
  - retrievable via `GET /runs/<run_id>`

---

## 4) Automatic repo indexing

## Goal
Keep index fresh without manual `ai-dev index` calls.

## Implementation

1. Add file watcher worker (`watchdog`) to detect file changes.
2. Incremental indexing pipeline:
   - changed files only
   - tombstones for deleted files.
3. Git hooks integration:
   - optional post-checkout/post-merge reindex trigger.
4. Add `ai-dev index --daemon` and `ai-dev index --once` modes.

## Done when
- Retrieval quality stays current after edits/branch switches.

### Completion notes

- Added incremental indexing support to `ai-dev index` in `ai_dev/cli.py`:
  - `ai-dev index --once .` performs one incremental pass
  - `ai-dev index --daemon --interval <seconds> .` runs continuous incremental passes
- Added index state tracking file:
  - `.ai-dev/index_state.json` stores per-file metadata (`size`, `mtime_ns`) for change detection.
- Added reuse + tombstone-aware behavior:
  - unchanged files reuse prior indexed symbols/chunks
  - deleted files are removed from the new index output
  - no-op incremental runs skip index rewrite with a clear message
- Added optional git hook integration:
  - `ai-dev index --install-git-hooks .` installs/updates `.git/hooks/post-checkout` and `.git/hooks/post-merge`
  - hooks trigger `python3 -m ai_dev.cli index --once .` in a safe best-effort mode.

---

## 5) Prompt caching

## Goal
Reduce repeated prompt latency/cost.

## Implementation

1. Add cache middleware in LiteLLM or agent layer.
2. Cache key design:
   - model + normalized prompt + tool context hash + options.
3. TTL + invalidation:
   - invalidate on index version change and branch switch.
4. Add metrics:
   - hit rate, saved tokens, latency reduction.

## Done when
- Repeated/refinement prompts return significantly faster with high hit rate.

### Completion notes

- Added local prompt/result caching to `agent/server.py` for `POST /agent/run`.
- Cache key design implemented via normalized task payload hashing:
  - `task`, `model`, `dry_run`, `max_steps`, `plan`, `tool_context_hash`, `options`
- Cache namespace now includes invalidation-sensitive context:
  - current git branch
  - current index signature (`schema_version`, `generated_at`, `file_count`)
- Added configurable cache controls in request payload:
  - `cache.enabled` (default `true`)
  - `cache.ttl_seconds` (default `600`, bounded)
  - `cache.refresh` (force bypass + rewrite)
- Added cache metrics tracking and endpoint:
  - `GET /metrics`
  - persists to `.ai-dev/metrics.json` with requests/hits/misses/hit_rate/saved_calls.

---

## 6) Speculative decoding

## Goal
Use draft model + target model for faster generation.

## Implementation

1. Serve a small draft model and stronger target model.
2. Add orchestrator service (`spec-router`) between LiteLLM and MLX servers.
3. Implement accept/reject token loop and fallback.
4. Enable only for latency-sensitive tasks (code completion/editing).

## Done when
- Median generation latency drops while quality remains comparable.

### Completion notes

- Added `spec-router` service scaffold for speculative decoding loop:
  - new service source: `spec_router/server.py`
  - endpoints:
    - `GET /health`
    - `POST /spec/decode`
- Added optional compose service in generated `podman-compose.yml` via `ai-dev init`:
  - service name: `spec-router`
  - port: `8092`
  - volume: `./spec_router:/app`
- Added CLI helper command:
  - `ai-dev spec-decode --draft-tokens ... --target-tokens ...`
  - supports JSON output and text tokenization fallback from `--draft-text` / `--target-text`.
- Added stack schema support for spec-router port:
  - `stack.spec_router_port` in default config.

---

## 7) Background embedding workers

## Goal
Continuously embed chunks/events without blocking inference.

## Implementation

1. Add queue (Redis or SQLite-backed queue to start).
2. Worker service consumes file-change jobs.
3. Embed with local model and upsert into Qdrant.
4. Add backpressure + retries + dead-letter queue.

## Done when
- Embeddings are updated asynchronously and reliably.

### Completion notes

- Added a lightweight SQLite-backed embedding queue service:
  - source: `embedding_queue/server.py`
  - endpoints:
    - `GET /health`
    - `GET /stats`
    - `POST /jobs/enqueue`
    - `POST /jobs/claim`
    - `POST /jobs/complete`
    - `POST /jobs/fail`
- Added background worker service scaffold:
  - source: `embedding_worker/worker.py`
  - polls queue, processes jobs, writes vectors to `.ai-dev/embeddings.jsonl`
  - includes retry/dead-letter cooperation with queue API.
- Integrated both services into stack generation and runtime compose wiring:
  - `ai_dev/cli.py` now generates `embed-queue` and `embed-worker` services in `podman-compose.yml`
  - `command_init()` now writes `embedding_queue/server.py` and `embedding_worker/worker.py`
  - added stack schema key: `stack.embed_queue_port`.
- Added CLI helpers for queue operations:
  - `ai-dev embed-enqueue`
  - `ai-dev embed-stats`.

---

## 8) Git-aware code memory

## Goal
Make retrieval branch/commit aware.

## Implementation

1. Attach git metadata to each chunk:
   - commit SHA, branch, file path, symbol, timestamp.
2. Maintain branch namespace in vector store.
3. During retrieval, bias current branch + recent commits.
4. Add `ai-dev memory explain <query>` for transparency.

## Done when
- Retrieval reflects the active branch and recent development history.

### Completion notes

- Extended indexing metadata in `ai_dev/cli.py` to attach git attributes per file/symbol/chunk:
  - `git_branch`
  - `git_commit_sha`
  - `git_commit_ts`
- Enhanced `ai-dev retrieve` scoring with branch/recency-aware ranking signals:
  - current-branch match bias
  - recent-commit recency bias
  - existing changed-file and path-prefix bias retained
- Added transparent explanation command:
  - `ai-dev memory explain <query>`
  - includes per-result `score_breakdown` for lexical, path, changed-file, branch, and recency components.

---

## 9) Shared KV cache

## Goal
Reuse model KV states across related requests/sessions.

## Implementation

1. Introduce session-aware inference gateway.
2. Track conversation prefix hashes for cache reuse eligibility.
3. Implement per-model memory budget and LRU eviction.
4. Add safeguards:
   - tenant/session isolation
   - prompt-boundary validation.

## Done when
- Follow-up turns and repeated context windows are notably faster.

### Completion notes

- Added shared KV-cache state file in agent runtime:
  - `.ai-dev/kv_cache.json`
- Implemented session-aware KV reuse logic in `agent/server.py`:
  - tenant/session/model scoped keys
  - prefix hashing and prefix-boundary validation
  - reuse status returned as `hit/miss/bypass/rejected` with reason metadata
- Added per-model memory budget + LRU-style eviction safeguards:
  - configurable per-request budget (`kv_cache.model_budget_tokens`)
  - entry token estimation + cap (`kv_cache.entry_max_tokens`)
  - oldest non-active entries evicted when over budget
- Exposed KV observability:
  - `POST /agent/run` now returns a `kv_cache` block
  - `GET /metrics` now includes KV model usage summaries (entries/used/budget tokens).

---

## Suggested execution order (practical)

1. Multi-model inference
2. Automatic/incremental indexing
3. Repo-aware retrieval for code generation
4. Function-calling agent loop
5. Background embedding workers + Qdrant hardening
6. Prompt caching
7. Git-aware memory
8. Speculative decoding
9. Shared KV cache

---

## Minimal next sprint (1 week)

- Add model profiles + `ai-dev models`.
- Add incremental index update command.
- Add retrieval endpoint in `agent/server.py`.
- Add basic tool-calling contract and execution logs.
- Add metrics file (`.ai-dev/metrics.json`) for latency/hit rates.

---

## Missing / incomplete functionality backlog (post-milestone hardening)

The milestone checkboxes above are complete at a **foundation** level, but the
following items remain incomplete for production-grade behavior.

### A) Agent mutation path hardening

1. [x] Implement `write_patch` in agent tools (no longer `not_implemented`).
2. [x] Add guarded patch application flow:
   - [x] preflight diff validation
   - [x] file allow/deny policy
   - [x] rollback safety for failed post-apply verification.

#### A completion notes (hardening slice)

- `agent/server.py` now implements `tool_write_patch` with:
  - dry-run mutation blocking
  - payload size guard
  - patch target extraction + path normalization
  - denylist/root-boundary enforcement (including `.git/`)
  - `git apply --check` preflight before apply
  - snapshot + rollback on post-apply verification failure
- Synced the `AGENT_SERVER` embedded template in `ai_dev/cli.py` to prevent runtime/template drift.

### B) Real inference-level acceleration (vs simulation)

1. Upgrade shared KV cache from orchestration metadata to actual model-backend KV reuse.
2. Upgrade speculative decoding from token-list scaffold to real draft/target integration.
   - [x] live draft/target model-call decode path (prompt -> draft/target completion requests)
   - [x] streaming token acceptance loop across draft/target backends.

#### B completion notes (spec decode integration slice)

- Upgraded `spec_router/server.py` to support model-backed speculative decode runs:
  - supports prompt-driven draft + target model calls (OpenAI-compatible completions endpoint)
  - preserves token-list direct mode for deterministic tests/backward compatibility
  - gracefully handles draft call failures while still producing target output.
- Added richer decode response metadata:
  - source mode (`provided_tokens` vs `model_calls`)
  - per-model token counts and latency timings
  - draft error surface when fallback behavior is triggered.
- Added test coverage in `tests/test_spec_router.py` for:
  - acceptance math in token mode
  - prompt mode model-call behavior
  - draft fallback path
  - input validation errors.

#### B completion notes (streaming acceptance loop slice)

- Extended `spec_router/server.py` with iterative speculative token acceptance mode for model-backed decoding:
  - added `run_streaming_acceptance_loop(...)` that advances one token at a time
  - compares draft vs target token per step and appends accepted/replaced target token output
  - stops on target stream exhaustion or configured `max_tokens` budget.
- Added richer runtime telemetry for streaming speculative loop:
  - `loop_mode: stream_acceptance`
  - per-step trace entries (`step`, `draft`, `target`, `accepted`)
  - `rejected_tokens` count in addition to accepted/comparison metrics
  - accumulated `draft_call_ms` / `target_call_ms` across token steps.
- Preserved compatibility mode for prior non-stream prompt behavior:
  - `stream_loop: false` uses existing full-completion comparison path.
- Added/extended unit coverage in `tests/test_spec_router.py`:
  - streaming acceptance path with mixed accept/reject decisions
  - streaming path with draft failure fallback while target continues
  - explicit non-stream compatibility mode validation.

### C) Embedding pipeline productionization

1. [x] Replace fake/hash embeddings with real local embedding model inference.
2. [x] Add robust Qdrant upsert/query integration path (beyond JSONL sink fallback).
3. [x] Add embedding schema/versioning + reindex migration tooling.

#### C completion notes (embedding productionization slice)

- Upgraded `embedding_worker/worker.py` to support real embedding inference via HTTP (`/v1/embeddings`) with deterministic fallback behavior:
  - primary path: local embedding endpoint (`--embed-url`, `--embed-model`)
  - fallback path: hash-based deterministic vectors when embedding backend is unavailable
  - explicit override for deterministic mode via `--force-fake-embed`.
- Added robust Qdrant integration in worker runtime:
  - optional collection ensure/create before upsert
  - point upsert with job metadata payload
  - non-fatal Qdrant failure handling that preserves JSONL sink continuity
  - runtime controls via `--qdrant-url`, `--qdrant-collection`, `--disable-qdrant`.
- Added embedding schema/versioning and migration controls:
  - schema file: `.ai-dev/embedding_schema.json`
  - migration log: `.ai-dev/embedding_migrations.jsonl`
  - strict schema compatibility checks on model/dim/backend
  - safe rotate-on-migrate path for legacy embeddings JSONL via `--allow-schema-migrate`.
- Added productionization tests in `tests/test_embedding_worker.py` covering:
  - HTTP embedding success path
  - fallback + Qdrant failure non-fatal behavior
  - schema mismatch failure without migration flag
  - schema migration rotation behavior.

### D) Quality, testing, and observability

1. Add unit/integration tests for:
   - [x] retrieval scoring and explain output
   - [x] queue retry/dead-letter behavior
   - [x] KV cache prefix/budget/eviction edge cases
   - [x] agent tool execution paths (`write_patch` dry-run/deny/success paths).
2. Expand observability:
   - structured event traces across services
   - [x] richer metrics for cache/queue/retrieval/tool latencies
   - [x] optional alert thresholds.

### E) Template/runtime drift prevention

1. Reduce duplication between runtime service files and template literals in `ai_dev/cli.py`.
2. [x] Add CI checks to enforce template/runtime parity where duplication remains.

#### E completion notes (parity enforcement slice)

- Added `tools/check_template_parity.py` to enforce parity between:
  - `agent/server.py`
  - embedded `AGENT_SERVER` template in `ai_dev/cli.py`
- Integrated parity validation into CI (`.github/workflows/build.yml`).
- Added `tests/test_agent_write_patch.py` and wired unit-test execution in CI.

#### D completion notes (testing slice)

- Added `tests/test_agent_kv_cache.py` covering KV-cache edge cases:
  - missing-session bypass
  - prefix-hash mismatch rejection
  - miss->hit on prefix extension
  - budget-triggered eviction behavior
- Added `tests/test_embedding_queue.py` covering queue reliability paths:
  - retry transition
  - dead-letter transition on max attempts
  - completion path to `done`
- Added `tests/test_retrieval_memory_explain.py` covering:
  - recency boost behavior for recent vs old commits
  - score breakdown shape in symbol scoring
  - JSON payload contract for `ai-dev memory explain`
- Added tool-level observability in `agent/server.py`:
  - per-tool call counts (`calls`, `ok`, `errors`)
  - per-tool latency aggregates (`duration_ms_total`, `avg_duration_ms`)
  - last error tracking for failed tools
  - per-step duration capture in run traces
- Added `tests/test_agent_tool_metrics.py` to validate tool metrics aggregation and error accounting.

#### D completion notes (observability + alerts residual slice)

- Added structured event traces across core services:
  - `agent/server.py` -> `.ai-dev/events/agent.jsonl`
    - emits `run_started`, `run_completed`, and `alerts_emitted` events
  - `embedding_queue/server.py` -> `.ai-dev/events/embed-queue.jsonl`
    - emits queue lifecycle events (`job_enqueued`, `job_claimed`, `job_failed`, `job_completed`) and `alerts_emitted`
  - `embedding_worker/worker.py` -> `.ai-dev/events/embed-worker.jsonl`
    - emits worker lifecycle events (`job_processing_started`, `job_processing_completed`, `job_processing_failed`, `job_marked_done`) and Qdrant upsert outcomes.
- Added configurable alert thresholds surfaced in service APIs:
  - agent thresholds via env:
    - `AGENT_ALERT_TOOL_ERRORS` (default `5`)
    - `AGENT_ALERT_CACHE_HIT_RATE_MIN` (default `0.2`)
  - embed queue threshold via env:
    - `EMBED_QUEUE_ALERT_DEAD_LETTER` (default `5`)
- Enriched endpoint payloads:
  - `GET /metrics` (agent) now includes `alerts` and `alert_thresholds`
  - `GET /stats` (embed-queue) now includes `alerts` and `alert_thresholds`.
- Extended observability tests:
  - `tests/test_agent_tool_metrics.py`
    - alert-threshold parsing behavior
    - alert computation behavior
    - run lifecycle event emission
  - `tests/test_embedding_queue.py`
    - dead-letter threshold parsing and alert behavior
    - queue lifecycle event emission
  - `tests/test_embedding_worker.py`
    - worker event emission for success, Qdrant failure, and processing failure paths.

### F) Codebase refactoring and modularization

1. [~] Refactor `ai_dev/cli.py` into smaller modules (command groups + shared utilities) to reduce coupling and file size.
2. [x] Extract embedded service templates/constants from `ai_dev/cli.py` into dedicated template files or a template package.
3. Split `agent/server.py` into focused modules (cache, kv-cache, tool execution, HTTP handlers) to improve maintainability.
4. Add lightweight architecture boundaries (e.g., `ai_dev/core`, `ai_dev/services`, `ai_dev/templates`) and update imports accordingly.

#### F completion notes (template extraction slice)

- Extracted large embedded templates from `ai_dev/cli.py` into `ai_dev/templates/service_templates.py` with package re-exports in `ai_dev/templates/__init__.py`.
- Updated CLI initialization flow to import template constants from `ai_dev.templates` while preserving generated runtime file content.
- Upgraded `tools/check_template_parity.py` to validate parity across all extracted service templates:
  - `agent/server.py`
  - `rag/server.py`
  - `spec_router/server.py`
  - `embedding_queue/server.py`
  - `embedding_worker/worker.py`
- Updated packaging config in `pyproject.toml` to include subpackages (`ai_dev.*`), ensuring `ai_dev.templates` is distributed.

#### F completion notes (core extraction tranche)

- Added `ai_dev/core/` package and extracted reusable indexing/retrieval logic from `ai_dev/cli.py` into:
  - `ai_dev/core/indexing.py`
  - `ai_dev/core/retrieval.py`
- Updated `ai_dev/cli.py` to delegate key indexing/retrieval helpers to the new core modules while preserving existing CLI behavior and command contracts.
- Preserved backward compatibility for current tests by keeping existing helper function names in `ai_dev/cli.py` as thin wrappers around core implementations.
