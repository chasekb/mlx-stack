from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def command_init(
    _,
    *,
    app_dir: Path,
    config_path: Path,
    load_config_fn,
    write_file_fn,
    generate_litellm_config_fn,
    template_files: list[tuple[Path, str, bool]],
) -> int:
    app_dir.mkdir(parents=True, exist_ok=True)

    config = load_config_fn()
    config["created_at"] = config.get("created_at") or datetime.now(timezone.utc).isoformat()

    for file_path, content, executable in template_files:
        write_file_fn(file_path, content, executable=executable)

    write_file_fn(Path("litellm_config.yaml"), generate_litellm_config_fn(config))
    write_file_fn(config_path, json.dumps(config, indent=2) + "\n")

    print("Initialized local AI dev stack files.")
    return 0
