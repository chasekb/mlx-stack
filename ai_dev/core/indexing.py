from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from .retrieval import tokenize


def iter_source_files(root: Path, max_bytes: int) -> Iterable[Path]:
    skip_dirs = {".git", ".venv", "node_modules", "__pycache__", ".ai-dev"}
    allowed = {
        ".py",
        ".md",
        ".txt",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".sh",
        ".sql",
        ".go",
        ".rs",
        ".java",
        ".c",
        ".cpp",
        ".h",
    }
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in skip_dirs for part in p.parts):
            continue
        if p.suffix.lower() not in allowed:
            continue
        if p.stat().st_size > max_bytes:
            continue
        yield p


def collect_source_files(root: Path, max_bytes: int) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for p in iter_source_files(root, max_bytes=max_bytes):
        files[str(p.relative_to(root))] = p
    return files


def extract_symbols(file_path: Path, content: str) -> list[dict]:
    suffix = file_path.suffix.lower()
    symbols: list[dict] = []
    lines = content.splitlines()

    def add(name: str, line_no: int, kind: str) -> None:
        symbols.append({"name": name, "line": line_no, "kind": kind})

    for i, line in enumerate(lines, start=1):
        if suffix == ".py":
            m = re.match(r"^\s*(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", line)
            if m:
                add(m.group(2), i, m.group(1))
        elif suffix in {".js", ".ts", ".jsx", ".tsx"}:
            m = re.match(r"^\s*(export\s+)?(async\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)", line)
            if m:
                add(m.group(3), i, "function")
            m2 = re.match(r"^\s*(export\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)", line)
            if m2:
                add(m2.group(2), i, "class")
        elif suffix == ".go":
            m = re.match(r"^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_][A-Za-z0-9_]*)", line)
            if m:
                add(m.group(1), i, "func")
        elif suffix == ".rs":
            m = re.match(r"^\s*(?:pub\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)", line)
            if m:
                add(m.group(1), i, "fn")
    return symbols


def build_chunks(content: str, lines_per_chunk: int = 80) -> list[dict]:
    lines = content.splitlines()
    chunks = []
    chunk_id = 0
    for start in range(0, len(lines), lines_per_chunk):
        chunk_id += 1
        end = min(start + lines_per_chunk, len(lines))
        text = "\n".join(lines[start:end])
        tok_counter = Counter(tokenize(text))
        chunks.append(
            {
                "chunk_id": chunk_id,
                "start_line": start + 1,
                "end_line": end,
                "token_count": sum(tok_counter.values()),
                "top_terms": dict(tok_counter.most_common(15)),
                "text_preview": text[:300],
                "terms": list(tok_counter.keys()),
            }
        )
    return chunks
