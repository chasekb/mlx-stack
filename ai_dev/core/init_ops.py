from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ai_dev.templates import (
    AGENT_HTTP_API,
    AGENT_HTTP_SERVICE,
    AGENT_SERVER,
    EMBED_QUEUE_SERVER,
    EMBED_WORKER,
    MLX_DOCKERFILE,
    MLX_ENTRYPOINT,
    PODMAN_COMPOSE_YAML,
    RAG_SERVER,
    SPEC_ROUTER_SERVER,
)


DEFAULT_TEMPLATE_FILES: list[tuple[Path, str, bool]] = [
    (Path("podman-compose.yml"), PODMAN_COMPOSE_YAML, False),
    (Path("mlx/entrypoint.sh"), MLX_ENTRYPOINT, True),
    (Path("mlx/Dockerfile"), MLX_DOCKERFILE, False),
    (Path("rag/server.py"), RAG_SERVER, False),
    (Path("agent/server.py"), AGENT_SERVER, False),
    (Path("agent/http_api.py"), AGENT_HTTP_API, False),
    (Path("agent/http_service.py"), AGENT_HTTP_SERVICE, False),
    (Path("spec_router/server.py"), SPEC_ROUTER_SERVER, False),
    (Path("embedding_queue/server.py"), EMBED_QUEUE_SERVER, False),
    (Path("embedding_worker/worker.py"), EMBED_WORKER, False),
]


def command_init(
    _,
    *,
    app_dir: Path,
    config_path: Path,
    load_config_fn,
    write_file_fn,
    generate_litellm_config_fn,
    template_files: list[tuple[Path, str, bool]] | None = None,
) -> int:
    app_dir.mkdir(parents=True, exist_ok=True)

    config = load_config_fn()
    config["created_at"] = config.get("created_at") or datetime.now(timezone.utc).isoformat()

    template_files = DEFAULT_TEMPLATE_FILES if template_files is None else template_files

    for file_path, content, executable in template_files:
        write_file_fn(file_path, content, executable=executable)

    write_file_fn(Path("litellm_config.yaml"), generate_litellm_config_fn(config))
    write_file_fn(config_path, json.dumps(config, indent=2) + "\n")

    print("Initialized local AI dev stack files.")
    return 0
