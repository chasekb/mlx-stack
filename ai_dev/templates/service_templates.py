"""Generated service/file templates used by ai_dev.cli command_init().

To reduce runtime/template duplication, these template constants are sourced
from canonical files in the repository tree.
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_repo_file(relative_path: str) -> str:
    target = (REPO_ROOT / relative_path).resolve()
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(
            f"Required template source not found: {relative_path} (expected at {target})"
        )
    return target.read_text(encoding="utf-8")


PODMAN_COMPOSE_YAML = _read_repo_file("podman-compose.yml")
LITELLM_CONFIG = _read_repo_file("litellm_config.yaml")
MLX_ENTRYPOINT = _read_repo_file("mlx/entrypoint.sh")
MLX_DOCKERFILE = _read_repo_file("mlx/Dockerfile")

RAG_SERVER = _read_repo_file("rag/server.py")
SPEC_ROUTER_SERVER = _read_repo_file("spec_router/server.py")
EMBED_QUEUE_SERVER = _read_repo_file("embedding_queue/server.py")
EMBED_WORKER = _read_repo_file("embedding_worker/worker.py")
AGENT_SERVER = _read_repo_file("agent/server.py")
AGENT_HTTP_API = _read_repo_file("agent/http_api.py")
AGENT_HTTP_SERVICE = _read_repo_file("agent/http_service.py")
