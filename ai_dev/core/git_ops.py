from __future__ import annotations

import subprocess
from pathlib import Path


def get_git_changed_files(root: Path) -> set[str]:
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return set()
    changed = set()
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip()
        if path:
            changed.add(path)
    return changed


def get_git_branch_name(root: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return "unknown"
    return (proc.stdout or "").strip() or "unknown"


def get_file_git_metadata(root: Path, rel_path: str, branch_name: str) -> dict:
    proc = subprocess.run(
        ["git", "log", "-1", "--format=%H|%ct", "--", rel_path],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return {
            "git_branch": branch_name,
            "git_commit_sha": "",
            "git_commit_ts": 0,
        }

    out = (proc.stdout or "").strip()
    if "|" not in out:
        return {
            "git_branch": branch_name,
            "git_commit_sha": "",
            "git_commit_ts": 0,
        }

    sha, ts = out.split("|", 1)
    try:
        ts_int = int(ts)
    except ValueError:
        ts_int = 0

    return {
        "git_branch": branch_name,
        "git_commit_sha": sha,
        "git_commit_ts": ts_int,
    }
