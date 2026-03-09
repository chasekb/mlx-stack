from __future__ import annotations


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
