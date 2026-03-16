from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Mapping


def run_command(cmd: list[str], *, cwd: Path | None = None, env: Mapping[str, str] | None = None) -> int:
    proc = subprocess.run(cmd, cwd=cwd, env=dict(env) if env is not None else None)
    return proc.returncode


def write_file(path: Path, content: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if executable:
        current_mode = path.stat().st_mode
        path.chmod(current_mode | 0o111)
