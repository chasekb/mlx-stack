import json
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse
from http.server import BaseHTTPRequestHandler, HTTPServer


ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / ".ai-dev" / "index.json"
RUNS_DIR = ROOT / ".ai-dev" / "runs"
ALLOWED_TOOLS = {
    "retrieve",
    "search_code",
    "read_file",
    "git_diff",
    "run_tests",
    "write_patch",
    "commit_changes",
}

TOOL_SCHEMAS = {
    "retrieve": {
        "description": "Retrieve relevant symbols/chunks from local index",
        "input": {"query": "string", "top_k": "int?", "path_prefix": "string?"},
    },
    "search_code": {
        "description": "Regex search across repository files",
        "input": {"regex": "string", "file_pattern": "string?", "limit": "int?"},
    },
    "read_file": {
        "description": "Read a file from repo",
        "input": {"path": "string", "max_chars": "int?"},
    },
    "git_diff": {
        "description": "Get current git diff summary",
        "input": {},
    },
    "run_tests": {
        "description": "Run tests in dry-run or execute mode",
        "input": {"command": "string?"},
    },
    "write_patch": {
        "description": "Apply patch to repo (blocked in dry-run)",
        "input": {"patch": "string"},
    },
    "commit_changes": {
        "description": "Commit current changes (blocked in dry-run)",
        "input": {"message": "string"},
    },
}


def tokenize(text: str) -> list[str]:
    return [tok for tok in re.split(r"[^a-zA-Z0-9_]+", text.lower()) if len(tok) >= 2]


def retrieve(index_obj: dict, query: str, top_k: int = 5, path_prefix: Optional[str] = None) -> dict:
    query_terms = set(tokenize(query))
    if not query_terms:
        return {"query": query, "top_symbols": [], "top_chunks": []}

    path_prefix = path_prefix or ""

    symbol_results = []
    for s in index_obj.get("symbols", []):
        score = 0.0
        name_terms = set(tokenize(s.get("name", "")))
        score += len(query_terms.intersection(name_terms)) * 3
        score += 1 if any(t in s.get("name", "").lower() for t in query_terms) else 0
        p = s.get("path", "")
        if path_prefix and p.startswith(path_prefix):
            score += 1.5
        if score > 0:
            symbol_results.append({"score": score, **s})

    chunk_results = []
    for c in index_obj.get("chunks", []):
        score = 0.0
        chunk_terms = set(c.get("terms", []))
        score += len(query_terms.intersection(chunk_terms))
        p = c.get("path", "")
        if path_prefix and p.startswith(path_prefix):
            score += 2.0
        if score > 0:
            chunk_results.append(
                {
                    "score": score,
                    "path": p,
                    "chunk_id": c.get("chunk_id"),
                    "start_line": c.get("start_line"),
                    "end_line": c.get("end_line"),
                    "text_preview": c.get("text_preview", ""),
                }
            )

    symbol_results.sort(key=lambda x: x["score"], reverse=True)
    chunk_results.sort(key=lambda x: x["score"], reverse=True)

    return {
        "query": query,
        "top_symbols": symbol_results[:top_k],
        "top_chunks": chunk_results[:top_k],
    }


def ensure_under_root(path: Path) -> bool:
    try:
        path.resolve().relative_to(ROOT.resolve())
        return True
    except Exception:
        return False


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
    return {"ok": False, "error": "not_implemented"}


def tool_commit_changes(args: dict, dry_run: bool) -> dict:
    if dry_run:
        return {"ok": False, "error": "blocked_in_dry_run"}
    msg = str(args.get("message", "Agent commit")).strip()
    if not msg:
        return {"ok": False, "error": "missing_message"}
    proc = subprocess.run(["git", "commit", "-am", msg], cwd=ROOT, capture_output=True, text=True)
    return {"ok": proc.returncode == 0, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}


def execute_tool_call(tool: str, args: dict, dry_run: bool) -> dict:
    if tool not in ALLOWED_TOOLS:
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
        return {"ok": True, "result": retrieve(index_obj, query=query, top_k=max(1, min(top_k, 20)), path_prefix=path_prefix)}
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


def run_agent_task(payload: dict) -> dict:
    task = str(payload.get("task", "")).strip()
    dry_run = bool(payload.get("dry_run", True))
    max_steps = int(payload.get("max_steps", 6))
    max_steps = max(1, min(max_steps, 25))
    plan = payload.get("plan", [])
    run_id = uuid.uuid4().hex[:12]

    trace = {
        "run_id": run_id,
        "task": task,
        "dry_run": dry_run,
        "max_steps": max_steps,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "steps": [],
    }

    if not isinstance(plan, list) or not plan:
        trace["steps"].append({"tool": "noop", "result": {"ok": True, "detail": "No plan steps provided"}})
    else:
        for step in plan[:max_steps]:
            tool = str(step.get("tool", "")).strip()
            args = step.get("args", {}) if isinstance(step.get("args", {}), dict) else {}
            result = execute_tool_call(tool, args, dry_run=dry_run)
            trace["steps"].append({"tool": tool, "args": args, "result": result})

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_path = RUNS_DIR / f"{run_id}.json"
    run_path.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")

    return {
        "ok": True,
        "run_id": run_id,
        "run_path": str(run_path.relative_to(ROOT)),
        "step_count": len(trace["steps"]),
        "dry_run": dry_run,
        "steps": trace["steps"],
    }


class Handler(BaseHTTPRequestHandler):
    def _reply(self, payload, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode('utf-8'))

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == '/tools':
            self._reply({'ok': True, 'service': 'agent', 'tools': TOOL_SCHEMAS})
            return

        if parsed.path.startswith('/runs/'):
            run_id = parsed.path.split('/runs/', 1)[1].strip()
            target = (RUNS_DIR / f"{run_id}.json").resolve()
            if not target.exists() or not target.is_file() or not ensure_under_root(target):
                self._reply({'error': 'run_not_found'}, status=404)
                return
            payload = json.loads(target.read_text(encoding='utf-8'))
            self._reply({'ok': True, 'service': 'agent', 'run': payload})
            return

        if parsed.path == '/retrieve':
            if not INDEX_PATH.exists():
                self._reply({'error': 'missing_index', 'detail': 'Run `ai-dev index .` first.'}, status=400)
                return

            qs = parse_qs(parsed.query)
            query = (qs.get('q', [''])[0] or '').strip()
            if not query:
                self._reply({'error': 'missing_query', 'detail': 'Provide q=<query>'}, status=400)
                return

            try:
                top_k = int((qs.get('top_k', ['5'])[0] or '5'))
            except ValueError:
                top_k = 5
            top_k = max(1, min(top_k, 20))
            path_prefix = (qs.get('path_prefix', [''])[0] or '').strip() or None

            index_obj = json.loads(INDEX_PATH.read_text(encoding='utf-8'))
            payload = retrieve(index_obj, query=query, top_k=top_k, path_prefix=path_prefix)
            self._reply({'ok': True, 'service': 'agent', 'retrieval': payload})
            return

        if parsed.path == '/health':
            self._reply({'ok': True, 'service': 'agent'})
            return

        self._reply({'error': 'not found'}, status=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != '/agent/run':
            self._reply({'error': 'not found'}, status=404)
            return

        try:
            content_length = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            content_length = 0
        body = self.rfile.read(max(0, content_length))

        try:
            payload = json.loads(body.decode('utf-8') if body else '{}')
        except Exception:
            self._reply({'error': 'invalid_json'}, status=400)
            return

        result = run_agent_task(payload if isinstance(payload, dict) else {})
        self._reply({'ok': True, 'service': 'agent', 'result': result})


if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', 8091), Handler)
    print('Agent service listening on :8091')
    server.serve_forever()
