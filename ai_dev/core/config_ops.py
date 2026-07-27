from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_CONFIG = {
    "created_at": "",
    "stack": {
        "mlx_port": 8082,
        "mlx_api_base": "http://host.containers.internal:8082/v1",
        "litellm_port": 4000,
        "spec_router_port": 8092,
        "embed_queue_port": 8093,
        "default_model": "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit",
        "mlx_model_path": "models/local-mlx",
        "mlx_bind_host": "0.0.0.0",
        "mlx_prompt_cache_size": 4,
        "mlx_prompt_cache_bytes": "2GB",
        "mlx_decode_concurrency": 1,
        "mlx_prompt_concurrency": 1,
        "mlx_draft_model_path": "",
        "mlx_num_draft_tokens": 0,
        "embed_url": "http://litellm:4000/v1/embeddings",
        "embed_model": "local-embed",
        "qdrant_url": "http://qdrant:6333",
        "qdrant_collection": "ai_dev_embeddings",
        "force_fake_embed": False,
        "allow_schema_migrate": False,
    },
    "models": [
        {
            "name": "local-mlx-fast",
            "backend_model": "openai/models/local-mlx-fast",
            "api_base": "http://host.containers.internal:8082/v1",
            "api_key": "local-dev",
            "hf_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "mlx_model": "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
            "quantization": "4bit",
            "tags": ["fast", "default"],
        },
        {
            "name": "local-mlx",
            "backend_model": "openai/models/local-mlx",
            "api_base": "http://host.containers.internal:8082/v1",
            "api_key": "local-dev",
            "hf_model": "Qwen/Qwen2.5-Coder-3B-Instruct",
            "mlx_model": "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit",
            "quantization": "4bit",
            "tags": ["quality", "default"],
        },
        {
            "name": "local-mlx-longctx",
            "backend_model": "openai/models/local-mlx-longctx",
            "api_base": "http://host.containers.internal:8082/v1",
            "api_key": "local-dev",
            "hf_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "mlx_model": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
            "quantization": "4bit",
            "tags": ["longctx", "analysis"],
        },
        {
            "name": "local-mlx-agentic",
            "backend_model": "openai/models/local-mlx-agentic",
            "api_base": "http://host.containers.internal:8082/v1",
            "api_key": "local-dev",
            "hf_model": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
            "mlx_model": "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
            "quantization": "4bit",
            "optional": True,
            "notes": "High-memory optional agentic coding profile; not selected by default.",
            "tags": ["agentic", "optional"],
        },
        {
            "name": "local-embed",
            "backend_model": "openai/local-embed",
            "api_base": "http://host.containers.internal:8082/v1",
            "api_key": "local-dev",
            "embedding": True,
            "vector_dim": 1024,
            "notes": "Explicit local embedding route; workers surface deterministic_fallback if unavailable.",
            "tags": ["embed", "retrieval"],
        },
    ],
    "routing": {
        "fast": "local-mlx-fast",
        "quality": "local-mlx",
        "longctx": "local-mlx-longctx",
        "analysis": "local-mlx-longctx",
        "agentic": "local-mlx-agentic",
        "default": "local-mlx",
    },
    "cursor": {
        "base_url": "http://localhost:4000/v1",
        "api_key": "local-dev",
        "model": "local-mlx",
    },
}


TASK_TAG_ALIASES = {
    "default": ["default", "quality"],
    "quality": ["quality", "default"],
    "fast": ["fast", "default"],
    "longctx": ["longctx", "analysis", "default"],
    "analysis": ["analysis", "longctx", "quality", "default"],
    "agentic": ["agentic", "analysis", "longctx", "quality", "default"],
}


def ensure_config_schema(cfg: dict, *, default_config: dict) -> dict:
    if "models" not in cfg or not isinstance(cfg["models"], list) or not cfg["models"]:
        cfg["models"] = copy.deepcopy(default_config["models"])

    if "cursor" not in cfg or not isinstance(cfg["cursor"], dict):
        cfg["cursor"] = copy.deepcopy(default_config["cursor"])

    if not cfg["cursor"].get("model"):
        cfg["cursor"]["model"] = cfg["models"][0]["name"]

    if not cfg["cursor"].get("base_url"):
        cfg["cursor"]["base_url"] = default_config["cursor"]["base_url"]

    if not cfg["cursor"].get("api_key"):
        cfg["cursor"]["api_key"] = default_config["cursor"]["api_key"]

    if "stack" not in cfg or not isinstance(cfg["stack"], dict):
        cfg["stack"] = copy.deepcopy(default_config["stack"])
    else:
        for k, v in default_config["stack"].items():
            cfg["stack"].setdefault(k, v)

    if "routing" not in cfg or not isinstance(cfg["routing"], dict):
        cfg["routing"] = copy.deepcopy(default_config["routing"])
    else:
        for k, v in default_config["routing"].items():
            cfg["routing"].setdefault(k, v)

    for m in cfg.get("models", []):
        if not m.get("output_path"):
            m["output_path"] = f"models/{m.get('name', 'local-mlx')}"

    return cfg


def load_config(
    *,
    config_path: Path,
    default_config: dict,
    ensure_config_schema_fn,
) -> dict:
    if config_path.exists():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        return ensure_config_schema_fn(cfg)

    cfg = copy.deepcopy(default_config)
    cfg["created_at"] = datetime.now(timezone.utc).isoformat()
    return cfg
