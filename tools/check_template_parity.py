from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_dev.templates import AGENT_HTTP_API, AGENT_SERVER, EMBED_QUEUE_SERVER, EMBED_WORKER, RAG_SERVER, SPEC_ROUTER_SERVER

TEMPLATE_RUNTIME_PAIRS = {
    "AGENT_SERVER": (AGENT_SERVER, ROOT / "agent" / "server.py"),
    "AGENT_HTTP_API": (AGENT_HTTP_API, ROOT / "agent" / "http_api.py"),
    "RAG_SERVER": (RAG_SERVER, ROOT / "rag" / "server.py"),
    "SPEC_ROUTER_SERVER": (SPEC_ROUTER_SERVER, ROOT / "spec_router" / "server.py"),
    "EMBED_QUEUE_SERVER": (EMBED_QUEUE_SERVER, ROOT / "embedding_queue" / "server.py"),
    "EMBED_WORKER": (EMBED_WORKER, ROOT / "embedding_worker" / "worker.py"),
}


def main() -> int:
    mismatches: list[str] = []

    for template_name, (template_source, runtime_path) in TEMPLATE_RUNTIME_PAIRS.items():
        runtime_source = runtime_path.read_text(encoding="utf-8")
        if template_source != runtime_source:
            mismatches.append(
                f"- {template_name} template does not match {runtime_path.relative_to(ROOT)}"
            )

    if mismatches:
        print("Template/runtime drift detected.", file=sys.stderr)
        for msg in mismatches:
            print(msg, file=sys.stderr)
        print("Re-sync templates before committing.", file=sys.stderr)
        return 1

    print("Template/runtime parity check passed for all service templates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
