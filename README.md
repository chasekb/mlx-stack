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

## Notes

- This is a developer-friendly starter orchestration layer, intentionally lightweight.
- The indexing command uses a local JSON-based lexical index (no cloud).
- You can swap in full embeddings/RAG later (Qdrant + LlamaIndex) while keeping the CLI workflow.
