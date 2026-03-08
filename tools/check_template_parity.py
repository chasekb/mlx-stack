from __future__ import annotations

import ast
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = ROOT / "ai_dev" / "cli.py"
AGENT_PATH = ROOT / "agent" / "server.py"


def get_agent_template_from_cli(cli_source: str) -> str:
    module = ast.parse(cli_source, filename=str(CLI_PATH))
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "AGENT_SERVER":
                    return ast.literal_eval(node.value)
    raise RuntimeError("AGENT_SERVER assignment not found in ai_dev/cli.py")


def main() -> int:
    cli_source = CLI_PATH.read_text(encoding="utf-8")
    agent_source = AGENT_PATH.read_text(encoding="utf-8")
    embedded = get_agent_template_from_cli(cli_source)

    if embedded != agent_source:
        print("Template/runtime drift detected for AGENT_SERVER.", file=sys.stderr)
        print("- ai_dev/cli.py::AGENT_SERVER does not match agent/server.py", file=sys.stderr)
        print("- Re-sync the embedded template before committing.", file=sys.stderr)
        return 1

    print("AGENT_SERVER template parity check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
