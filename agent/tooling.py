from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / ".ai-dev" / "index.json"
PATCH_DENY_PREFIXES = (
    ".git/",
)


def ensure_under_root(path: Path) -> bool:
    try:
        path.resolve().relative_to(ROOT.resolve())
        return True
    except Exception:
        return False


def normalize_patch_path(path_text: str) -> str | None:
    rel = str(path_text or "").strip().strip('"').strip("'")
    if not rel or rel == "/dev/null":
        return None
    rel = rel.replace("\\", "/")
    if rel.startswith("a/") or rel.startswith("b/"):
        rel = rel[2:]
    while rel.startswith("./"):
        rel = rel[2:]
    if not rel:
        return None
    return rel


def extract_patch_paths(patch_text: str) -> list[str]:
    seen: dict[str, bool] = {}
    lines = patch_text.splitlines()

    for line in lines:
        rel = None
        if line.startswith("diff --git "):
            m = re.match(r"^diff --git a/(.+?) b/(.+?)$", line)
            if m:
                rel = normalize_patch_path(m.group(2))
        elif line.startswith("+++ "):
            rel = normalize_patch_path(line[4:])
        elif line.startswith("*** Add File:") or line.startswith("*** Update File:") or line.startswith("*** Delete File:"):
            m = re.match(r"^\*\*\* (?:Add|Update|Delete) File: (.+?)(?:\s+->.+)?$", line)
            if m:
                rel = normalize_patch_path(m.group(1))

        if rel:
            seen[rel] = True

    return sorted(seen.keys())


def path_allowed_for_patch(rel_path: str) -> bool:
    if not rel_path:
        return False
    p = Path(rel_path)
    if p.is_absolute() or ".." in p.parts:
        return False
    normalized = rel_path.replace("\\", "/")
    for deny_prefix in PATCH_DENY_PREFIXES:
        if normalized.startswith(deny_prefix):
            return False
    target = (ROOT / p).resolve()
    return ensure_under_root(target)


def snapshot_paths(rel_paths: list[str]) -> dict:
    snapshot = {}
    for rel in rel_paths:
        target = (ROOT / rel).resolve()
        if target.exists() and target.is_dir():
            raise ValueError(f"target_is_directory:{rel}")
        if target.exists() and target.is_file():
            snapshot[rel] = {"exists": True, "content": target.read_text(encoding="utf-8", errors="ignore")}
        else:
            snapshot[rel] = {"exists": False, "content": ""}
    return snapshot


def restore_snapshot(snapshot: dict) -> None:
    for rel, prior in snapshot.items():
        target = (ROOT / rel).resolve()
        if not ensure_under_root(target):
            continue
        if bool(prior.get("exists", False)):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(prior.get("content", "")), encoding="utf-8")
        else:
            if target.exists() and target.is_file():
                target.unlink()


def tool_search_code(args: dict) -> dict:
    regex = str(args.get("regex", "")).strip()
    if not regex:
        return {"error": "missing_regex"}
    file_pattern = str(args.get("file_pattern", "*") or "*")
    limit = int(args.get("limit", 50))
    cmd = ["bash", "-lc", f"grep -RInE --include='{file_pattern}' {json.dumps(regex)} {json.dumps(str(ROOT))} | head -n {max(1, min(limit, 200))}"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {"ok": proc.returncode in (0, 1), "output": proc.stdout.strip(), "stderr": proc.stderr.strip()}


def tool_read_file(args: dict) -> dict:
    rel = str(args.get("path", "")).strip()
    if not rel:
        return {"error": "missing_path"}
    target = (ROOT / rel).resolve()
    if not target.exists() or not target.is_file() or not ensure_under_root(target):
        return {"error": "invalid_path"}
    max_chars = int(args.get("max_chars", 12000))
    content = target.read_text(encoding="utf-8", errors="ignore")[: max(1, max_chars)]
    return {"ok": True, "path": rel, "content": content}


def tool_git_diff(_: dict) -> dict:
    proc = subprocess.run(["git", "--no-pager", "diff", "--stat"], cwd=ROOT, capture_output=True, text=True)
    return {"ok": proc.returncode == 0, "output": proc.stdout.strip(), "stderr": proc.stderr.strip()}


def tool_run_tests(args: dict, dry_run: bool) -> dict:
    command = str(args.get("command", "python3 -m pytest -q") or "python3 -m pytest -q")
    if dry_run:
        return {"ok": True, "dry_run": True, "command": command}
    proc = subprocess.run(["bash", "-lc", command], cwd=ROOT, capture_output=True, text=True)
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-8000:],
        "stderr": proc.stderr[-4000:],
    }


def tool_write_patch(args: dict, dry_run: bool) -> dict:
    if dry_run:
        return {"ok": False, "error": "blocked_in_dry_run"}

    patch = str(args.get("patch", "") or "")
    if not patch.strip():
        return {"ok": False, "error": "missing_patch"}
    if len(patch) > 500_000:
        return {"ok": False, "error": "patch_too_large", "max_chars": 500000}

    rel_paths = extract_patch_paths(patch)
    if not rel_paths:
        return {"ok": False, "error": "no_target_files_detected"}

    denied = [p for p in rel_paths if not path_allowed_for_patch(p)]
    if denied:
        return {"ok": False, "error": "patch_target_denied", "denied_paths": denied}

    try:
        before_state = snapshot_paths(rel_paths)
    except ValueError as exc:
        return {"ok": False, "error": "invalid_patch_target", "detail": str(exc)}

    preflight = subprocess.run(
        ["git", "apply", "--check", "--whitespace=nowarn", "-"],
        cwd=ROOT,
        input=patch,
        text=True,
        capture_output=True,
    )
    if preflight.returncode != 0:
        return {
            "ok": False,
            "error": "preflight_failed",
            "stderr": (preflight.stderr or "").strip(),
            "stdout": (preflight.stdout or "").strip(),
        }

    apply_proc = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", "-"],
        cwd=ROOT,
        input=patch,
        text=True,
        capture_output=True,
    )
    if apply_proc.returncode != 0:
        return {
            "ok": False,
            "error": "apply_failed",
            "stderr": (apply_proc.stderr or "").strip(),
            "stdout": (apply_proc.stdout or "").strip(),
        }

    failed_verification = []
    for rel in rel_paths:
        target = (ROOT / rel).resolve()
        if not ensure_under_root(target):
            failed_verification.append(rel)

    if failed_verification:
        restore_snapshot(before_state)
        return {
            "ok": False,
            "error": "post_apply_verification_failed",
            "invalid_paths": failed_verification,
            "rolled_back": True,
        }

    return {"ok": True, "applied_files": rel_paths, "file_count": len(rel_paths)}


def tool_commit_changes(args: dict, dry_run: bool) -> dict:
    if dry_run:
        return {"ok": False, "error": "blocked_in_dry_run"}
    msg = str(args.get("message", "Agent commit")).strip()
    if not msg:
        return {"ok": False, "error": "missing_message"}
    proc = subprocess.run(["git", "commit", "-am", msg], cwd=ROOT, capture_output=True, text=True)
    return {"ok": proc.returncode == 0, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}


def execute_tool_call(
    tool: str,
    args: dict,
    dry_run: bool,
    allowed_tools: set[str],
    retrieve_fn: Callable[[dict, str, int, str | None], dict],
) -> dict:
    if tool not in allowed_tools:
        return {"ok": False, "error": "tool_not_allowed", "tool": tool}
    if tool == "retrieve":
        if not INDEX_PATH.exists():
            return {"ok": False, "error": "missing_index"}
        query = str(args.get("query", "")).strip()
        if not query:
            return {"ok": False, "error": "missing_query"}
        index_obj = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        top_k = int(args.get("top_k", 5))
        path_prefix = args.get("path_prefix")
        return {"ok": True, "result": retrieve_fn(index_obj, query=query, top_k=max(1, min(top_k, 20)), path_prefix=path_prefix)}
    if tool == "search_code":
        return tool_search_code(args)
    if tool == "read_file":
        return tool_read_file(args)
    if tool == "git_diff":
        return tool_git_diff(args)
    if tool == "run_tests":
        return tool_run_tests(args, dry_run=dry_run)
    if tool == "write_patch":
        return tool_write_patch(args, dry_run=dry_run)
    if tool == "commit_changes":
        return tool_commit_changes(args, dry_run=dry_run)
    return {"ok": False, "error": "unhandled_tool"}
