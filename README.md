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

## Notes

- This is a developer-friendly starter orchestration layer, intentionally lightweight.
- The indexing command uses a local JSON-based lexical index (no cloud).
- You can swap in full embeddings/RAG later (Qdrant + LlamaIndex) while keeping the CLI workflow.

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
