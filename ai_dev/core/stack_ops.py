from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def compose_command(compose_file: Path) -> list[str]:
    if not compose_file.exists():
        print("Missing podman-compose.yml. Run `ai-dev init` first.", file=sys.stderr)
        raise SystemExit(2)
    return ["podman", "compose", "-f", str(compose_file)]


def command_up(args, *, compose_command_fn, run_fn) -> int:
    cmd = compose_command_fn() + ["up", "-d"]
    if args.with_optional:
        cmd.extend(["--profile", "optional"])
    return run_fn(cmd)


def command_down(_, *, compose_command_fn, run_fn) -> int:
    cmd = compose_command_fn() + ["down"]
    return run_fn(cmd)


def command_status(_, *, compose_command_fn, run_fn) -> int:
    cmd = compose_command_fn() + ["ps"]
    return run_fn(cmd)


def generate_litellm_config(cfg: dict, default_models: list[dict]) -> str:
    models = cfg.get("models") or default_models
    lines = ["model_list:"]
    for m in models:
        name = m.get("name", "local-mlx")
        backend_model = m.get("backend_model", "openai/local-mlx")
        api_base = m.get("api_base", "http://mlx:8081/v1")
        api_key = m.get("api_key", "local-dev")
        lines.extend(
            [
                f"  - model_name: {name}",
                "    litellm_params:",
                f"      model: {backend_model}",
                f"      api_base: {api_base}",
                f"      api_key: {api_key}",
            ]
        )

    master_key = cfg.get("cursor", {}).get("api_key", "local-dev")
    lines.extend(["", "general_settings:", f"  master_key: {master_key}"])
    return "\n".join(lines) + "\n"


def command_pull_models(args, *, load_config_fn, python_executable: str, subprocess_run_fn=subprocess.run) -> int:
    cfg = load_config_fn()
    if args.profile:
        profiles = [m for m in cfg.get("models", []) if m.get("name") == args.profile]
    else:
        profiles = cfg.get("models", [])

    if not profiles:
        print("No matching model profiles found.", file=sys.stderr)
        return 2

    commands: list[tuple[str, list[str]]] = []
    for m in profiles:
        name = m.get("name", "local-mlx")
        hf_model = m.get("hf_model") or args.model
        q = m.get("quantization", f"{args.quantization}bit").replace("bit", "")
        output_path = m.get("output_path", f"models/{name}")
        Path(output_path).mkdir(parents=True, exist_ok=True)

        cmd = [
            python_executable,
            "-m",
            "mlx_lm",
            "convert",
            "--hf-path",
            hf_model,
            "--mlx-path",
            output_path,
            "--quantize",
            "--q-bits",
            q,
        ]
        commands.append((name, cmd))

    if args.dry_run:
        print("Dry run (commands to execute):\n")
        for name, cmd in commands:
            print(f"# Profile: {name}")
            print(" ".join(cmd))
            print("")
        return 0

    rc = 0
    for name, cmd in commands:
        print(f"[pull-models] Converting profile: {name}")
        proc = subprocess_run_fn(cmd)
        if proc.returncode != 0:
            rc = proc.returncode
            print(
                f"[pull-models] Failed for profile '{name}'. "
                "If mlx-lm is not installed in this Python env, install it first.",
                file=sys.stderr,
            )
            if not args.continue_on_error:
                return rc

    if rc == 0:
        print("[pull-models] Completed all model conversions.")
    else:
        print("[pull-models] Completed with errors.", file=sys.stderr)

    return rc
