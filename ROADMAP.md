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
- [ ] 6) Speculative decoding
- [ ] 7) Background embedding workers
- [ ] 8) Git-aware code memory
- [ ] 9) Shared KV cache

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
