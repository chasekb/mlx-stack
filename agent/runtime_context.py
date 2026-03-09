from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / ".ai-dev" / "index.json"


def get_git_branch(root: Path = ROOT) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return "unknown"
    return (proc.stdout or "").strip() or "unknown"


def get_index_signature(index_path: Path = INDEX_PATH) -> str:
    if not index_path.exists():
        return "no-index"
    try:
        index_obj = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return "index-unreadable"
    generated_at = str(index_obj.get("generated_at", "unknown"))
    schema_version = str(index_obj.get("schema_version", "?"))
    file_count = str(index_obj.get("file_count", "?"))
    return f"sv{schema_version}:{generated_at}:{file_count}"


def compute_cache_namespace(root: Path = ROOT, index_path: Path = INDEX_PATH) -> str:
    return f"branch={get_git_branch(root=root)}|index={get_index_signature(index_path=index_path)}"


def normalize_task_payload(payload: dict) -> dict:
    return {
        "task": str(payload.get("task", "")).strip(),
        "model": payload.get("model"),
        "dry_run": bool(payload.get("dry_run", True)),
        "max_steps": int(payload.get("max_steps", 6)),
        "plan": payload.get("plan", []),
        "tool_context_hash": payload.get("tool_context_hash"),
        "options": payload.get("options", {}),
    }


def compute_cache_key(payload: dict) -> str:
    canonical = json.dumps(normalize_task_payload(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
