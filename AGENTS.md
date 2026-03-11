# AGENTS.md

## Purpose

This file defines project-specific operating guidance for AI/code agents working in this repository.

Primary objective: continue implementing `ROADMAP.md` toward **production-grade completion** in safe, incremental tranches.

---

## Repository context

This project is a local AI development orchestrator centered on:

- CLI orchestration: `ai_dev/cli.py`
- Modular CLI/core logic: `ai_dev/core/*`
- Parser wiring: `ai_dev/command_groups.py`
- Service runtimes: `agent/`, `spec_router/`, `embedding_queue/`, `embedding_worker/`, `rag/`
- Template sources: `ai_dev/templates/*`
- Tests: `tests/`
- Drift prevention: `tools/check_template_parity.py`
- Planning/progress authority: `ROADMAP.md`

---

## Required workflow (for every implementation tranche)

1. Choose a **small, low-risk, high-value** tranche aligned with remaining roadmap gaps.
2. Preserve compatibility-first behavior (thin wrappers are acceptable when needed for tests/callers).
3. Update documentation when behavior/refactor progress changes:
   - `ROADMAP.md` completion notes
   - `README.md` refactor/progress notes
4. Run validation gates:

   ```bash
   python3 tools/check_template_parity.py
   python3 -m unittest discover -s tests -p 'test_*.py'
   python3 -m compileall ai_dev spec_router embedding_queue embedding_worker agent tests
   ```

5. Commit with clear scoped message (e.g., `refactor(cli): ...`, `feat(agent): ...`, `fix(queue): ...`).
6. Push to `origin/main` after green validation.

---

## Non-negotiable invariants

1. **Template/runtime parity must hold**
   - If runtime service code changes and mirrored template content exists, keep both synchronized.
   - `tools/check_template_parity.py` must pass.

2. **CLI contract stability**
   - Avoid breaking existing command flags/output contracts unless explicitly requested.
   - Maintain compatibility wrappers in `ai_dev/cli.py` when tests/importers depend on them.

3. **Command parser ownership**
   - Parser wiring belongs in `ai_dev/command_groups.py`.
   - Keep typed handler contract aligned with CLI handlers.

4. **Roadmap-first prioritization**
   - Prefer work that closes explicit unchecked backlog items in `ROADMAP.md`.
   - Favor tranche-by-tranche hardening over broad rewrites.

---

## Coding and change-style guidance

- Keep changes minimal and focused.
- Prefer extraction/delegation into `ai_dev/core/*` or focused modules over expanding monoliths.
- Preserve existing public function names where tests or integrations import them.
- Avoid introducing unrelated refactors in the same commit.

---

## Definition of done for a tranche

A tranche is done only when all are true:

- Implementation complete and scoped.
- Relevant docs updated (`ROADMAP.md`, `README.md`).
- Validation gates pass.
- Commit created with clear message.
- Changes pushed to `origin/main`.
