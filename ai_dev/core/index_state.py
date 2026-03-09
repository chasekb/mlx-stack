from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def load_index_state(path: Path, expected_root: Path) -> dict:
    if not path.exists():
        return {}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if str(expected_root) != str(state.get("root", "")):
        return {}
    return state


def save_index_state(path: Path, root: Path, file_meta: dict[str, dict]) -> None:
    payload = {
        "schema_version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "files": file_meta,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
