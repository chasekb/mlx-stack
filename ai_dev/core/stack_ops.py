from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse


def compose_command(compose_file: Path) -> list[str]:
    if not compose_file.exists():
        print("Missing podman-compose.yml. Run `ai-dev init` first.", file=sys.stderr)
        raise SystemExit(2)
    return ["podman", "compose", "-f", str(compose_file)]


def _container_names_from_compose_file(compose_file: Path) -> list[str]:
    if not compose_file.exists():
        return []
    names: list[str] = []
    for line in compose_file.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*container_name:\s*([^\s#]+)", line)
        if match:
            names.append(match.group(1).strip().strip('"\''))
    return names


def _compose_project_pod_name(project_root: Path) -> str:
    return f"{project_root.name}_default"


def _force_cleanup_compose_resources(
    *,
    compose_file: Path,
    project_root: Path,
    run_fn,
) -> int:
    container_names = _container_names_from_compose_file(compose_file)
    rc = 0

    if container_names:
        rc = run_fn(["podman", "stop", "-t", "0", *container_names]) or rc
        rc = run_fn(["podman", "rm", "-f", *container_names]) or rc

    pod_name = _compose_project_pod_name(project_root)
    rc = run_fn(["podman", "pod", "rm", "-f", pod_name]) or rc
    rc = run_fn(["podman", "network", "rm", pod_name]) or rc
    return rc


def _mlx_state_path(app_dir: Path) -> Path:
    return app_dir / "mlx_host_process.json"


def _mlx_log_path(app_dir: Path) -> Path:
    return app_dir / "mlx_host.log"


def _read_mlx_state(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_mlx_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def _remove_mlx_state(path: Path) -> None:
    if path.exists():
        path.unlink()


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _endpoint_is_reachable(url: str, *, host_override: str | None = None, timeout: float = 0.5) -> bool:
    parsed = urlparse(url)
    host = host_override or parsed.hostname or "127.0.0.1"
    if host == "host.containers.internal":
        host = "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _resolve_host_mlx_python(cfg: dict, *, project_root: Path, python_executable: str) -> str:
    stack_cfg = cfg.get("stack") or {}
    configured = stack_cfg.get("mlx_python")
    if configured:
        candidate = Path(configured)
        if not candidate.is_absolute():
            candidate = project_root / candidate
        if candidate.exists():
            return str(candidate)

    venv_candidate = project_root / ".venv" / "bin" / "python"
    if venv_candidate.exists():
        return str(venv_candidate)
    return python_executable


def _build_host_mlx_command(cfg: dict, *, project_root: Path, python_executable: str) -> list[str]:
    stack_cfg = cfg.get("stack") or {}
    model_path = stack_cfg.get("mlx_model_path", "models/local-mlx")
    bind_host = stack_cfg.get("mlx_bind_host", "0.0.0.0")
    port = str(stack_cfg.get("mlx_port", 8082))
    resolved_python = _resolve_host_mlx_python(cfg, project_root=project_root, python_executable=python_executable)

    model_path_obj = Path(model_path)
    if not model_path_obj.is_absolute():
        model_path_obj = project_root / model_path_obj

    cmd = [
        resolved_python,
        "-m",
        "mlx_lm",
        "server",
        "--model",
        str(model_path_obj),
        "--host",
        bind_host,
        "--port",
        port,
    ]

    option_map = (
        ("mlx_prompt_cache_size", "--prompt-cache-size"),
        ("mlx_prompt_cache_bytes", "--prompt-cache-bytes"),
        ("mlx_decode_concurrency", "--decode-concurrency"),
        ("mlx_prompt_concurrency", "--prompt-concurrency"),
        ("mlx_num_draft_tokens", "--num-draft-tokens"),
    )
    for cfg_key, flag in option_map:
        value = stack_cfg.get(cfg_key)
        if value not in (None, "", 0, "0"):
            cmd.extend([flag, str(value)])

    draft_model_path = str(stack_cfg.get("mlx_draft_model_path", "") or "").strip()
    if draft_model_path:
        draft_model_path_obj = Path(draft_model_path)
        if not draft_model_path_obj.is_absolute():
            draft_model_path_obj = project_root / draft_model_path_obj
        cmd.extend(["--draft-model", str(draft_model_path_obj)])

    return cmd


def _host_url_from_port(port: object, path: str = "/health") -> str:
    return f"http://localhost:{port}{path}"


def _stack_status_lines(cfg: dict, *, app_dir: Path, endpoint_reachable_fn=_endpoint_is_reachable) -> list[str]:
    stack_cfg = cfg.get("stack") or {}
    mlx_api_base = stack_cfg.get("mlx_api_base", "http://host.containers.internal:8082/v1")
    state = _read_mlx_state(_mlx_state_path(app_dir))
    host_mlx_reachable = endpoint_reachable_fn(mlx_api_base)
    host_mlx_status = "reachable" if host_mlx_reachable else "unreachable"
    pid_suffix = f", managed pid={state.get('pid')}" if state else ""

    acceleration = {
        "prompt_cache_size": stack_cfg.get("mlx_prompt_cache_size", ""),
        "prompt_cache_bytes": stack_cfg.get("mlx_prompt_cache_bytes", ""),
        "decode_concurrency": stack_cfg.get("mlx_decode_concurrency", ""),
        "prompt_concurrency": stack_cfg.get("mlx_prompt_concurrency", ""),
        "draft_model_path": stack_cfg.get("mlx_draft_model_path", ""),
        "num_draft_tokens": stack_cfg.get("mlx_num_draft_tokens", 0),
    }

    litellm_url = _host_url_from_port(stack_cfg.get("litellm_port", 4000), "/health")
    agent_url = _host_url_from_port(8091, "/health")
    rag_url = _host_url_from_port(8090, "/health")
    spec_url = _host_url_from_port(stack_cfg.get("spec_router_port", 8092), "/health")
    queue_url = _host_url_from_port(stack_cfg.get("embed_queue_port", 8093), "/health")
    qdrant_url = _host_url_from_port(6333, "/")

    def state_for(url: str) -> str:
        return "reachable" if endpoint_reachable_fn(url) else "unreachable"

    return [
        f"host-mlx: {host_mlx_status} ({mlx_api_base}{pid_suffix})",
        "host-mlx acceleration: " + json.dumps(acceleration, sort_keys=True),
        f"litellm: {state_for(litellm_url)} ({litellm_url})",
        f"agent: {state_for(agent_url)} ({agent_url})",
        f"spec-router: {state_for(spec_url)} ({spec_url})",
        f"embed-queue: {state_for(queue_url)} ({queue_url})",
        f"rag: {state_for(rag_url)} ({rag_url})",
        f"qdrant: {state_for(qdrant_url)} ({qdrant_url})",
        "embedding: "
        + json.dumps(
            {
                "embed_url": stack_cfg.get("embed_url", "http://litellm:4000/v1/embeddings"),
                "embed_model": stack_cfg.get("embed_model", "local-embed"),
                "qdrant_url": stack_cfg.get("qdrant_url", "http://qdrant:6333"),
                "qdrant_collection": stack_cfg.get("qdrant_collection", "ai_dev_embeddings"),
                "force_fake_embed": bool(stack_cfg.get("force_fake_embed", False)),
            },
            sort_keys=True,
        ),
    ]


def _ensure_host_mlx_running(
    cfg: dict,
    *,
    app_dir: Path,
    project_root: Path,
    python_executable: str,
    popen_fn=subprocess.Popen,
    sleep_fn=time.sleep,
    endpoint_reachable_fn=_endpoint_is_reachable,
    pid_is_alive_fn=_pid_is_alive,
) -> int:
    stack_cfg = cfg.get("stack") or {}
    mlx_api_base = stack_cfg.get("mlx_api_base", "http://host.containers.internal:8082/v1")
    state_path = _mlx_state_path(app_dir)
    log_path = _mlx_log_path(app_dir)

    if endpoint_reachable_fn(mlx_api_base):
        print(f"[ai-dev up] Host MLX endpoint already reachable at {mlx_api_base}.", file=sys.stderr)
        return 0

    state = _read_mlx_state(state_path)
    if state and pid_is_alive_fn(int(state.get("pid", -1))):
        wait_seconds = 15.0
        deadline = time.time() + wait_seconds
        while time.time() < deadline:
            if endpoint_reachable_fn(mlx_api_base):
                print(f"[ai-dev up] Managed MLX host process is now reachable at {mlx_api_base}.", file=sys.stderr)
                return 0
            sleep_fn(0.25)
        print(
            "[ai-dev up] Found a managed MLX host process but the endpoint is still unreachable. "
            f"See log: {log_path}",
            file=sys.stderr,
        )
        return 2

    _remove_mlx_state(state_path)
    cmd = _build_host_mlx_command(cfg, project_root=project_root, python_executable=python_executable)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        proc = popen_fn(
            cmd,
            cwd=str(project_root),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    _write_mlx_state(
        state_path,
        {
            "pid": proc.pid,
            "api_base": mlx_api_base,
            "command": cmd,
        },
    )

    deadline = time.time() + 30.0
    while time.time() < deadline:
        if endpoint_reachable_fn(mlx_api_base):
            print(f"[ai-dev up] Started host MLX server at {mlx_api_base} (pid {proc.pid}).", file=sys.stderr)
            return 0
        if proc.poll() is not None:
            print(
                "[ai-dev up] Failed to start host MLX server. "
                f"Process exited with code {proc.returncode}. See log: {log_path}",
                file=sys.stderr,
            )
            _remove_mlx_state(state_path)
            return 2
        sleep_fn(0.25)

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except OSError:
        pass
    _remove_mlx_state(state_path)
    print(
        "[ai-dev up] Timed out waiting for host MLX server to become reachable. "
        f"See log: {log_path}",
        file=sys.stderr,
    )
    return 2


def _stop_managed_host_mlx(app_dir: Path, *, pid_is_alive_fn=_pid_is_alive) -> int:
    state_path = _mlx_state_path(app_dir)
    state = _read_mlx_state(state_path)
    if not state:
        return 0

    pid = int(state.get("pid", -1))
    if pid > 0 and pid_is_alive_fn(pid):
        try:
            os.killpg(pid, signal.SIGTERM)
        except OSError:
            pass
    _remove_mlx_state(state_path)
    print(f"[ai-dev down] Stopped managed host MLX process (pid {pid}).", file=sys.stderr)
    return 0


def command_up(
    args,
    *,
    compose_command_fn,
    run_fn,
    load_config_fn,
    app_dir: Path,
    project_root: Path,
    python_executable: str,
    popen_fn=subprocess.Popen,
    sleep_fn=time.sleep,
    endpoint_reachable_fn=_endpoint_is_reachable,
    pid_is_alive_fn=_pid_is_alive,
) -> int:
    cfg = load_config_fn()
    rc = _ensure_host_mlx_running(
        cfg,
        app_dir=app_dir,
        project_root=project_root,
        python_executable=python_executable,
        popen_fn=popen_fn,
        sleep_fn=sleep_fn,
        endpoint_reachable_fn=endpoint_reachable_fn,
        pid_is_alive_fn=pid_is_alive_fn,
    )
    if rc != 0:
        return rc
    cmd = compose_command_fn() + ["up", "-d"]
    return run_fn(cmd, env={"COMPOSE_PROFILES": "optional"})


def command_down(
    _,
    *,
    compose_command_fn,
    run_fn,
    app_dir: Path,
    project_root: Path,
    compose_file: Path | None = None,
    pid_is_alive_fn=_pid_is_alive,
) -> int:
    cmd = compose_command_fn() + ["down"]
    rc = run_fn(cmd, env={"COMPOSE_PROFILES": "optional"})
    if rc != 0:
        cleanup_compose_file = compose_file or (project_root / "podman-compose.yml")
        cleanup_rc = _force_cleanup_compose_resources(
            compose_file=cleanup_compose_file,
            project_root=project_root,
            run_fn=run_fn,
        )
        rc = cleanup_rc if cleanup_rc == 0 else rc
    stop_rc = _stop_managed_host_mlx(app_dir, pid_is_alive_fn=pid_is_alive_fn)
    return rc or stop_rc


def command_status(_, *, compose_command_fn, run_fn, load_config_fn, app_dir: Path, endpoint_reachable_fn=_endpoint_is_reachable) -> int:
    cmd = compose_command_fn() + ["ps"]
    rc = run_fn(cmd)
    cfg = load_config_fn()
    for line in _stack_status_lines(cfg, app_dir=app_dir, endpoint_reachable_fn=endpoint_reachable_fn):
        print(line)
    return rc


def generate_litellm_config(cfg: dict, default_models: list[dict]) -> str:
    models = cfg.get("models") or default_models
    stack_cfg = cfg.get("stack") or {}
    default_api_base = stack_cfg.get("mlx_api_base", "http://host.containers.internal:8082/v1")
    lines = ["model_list:"]
    for m in models:
        name = m.get("name", "local-mlx")
        backend_model = m.get("backend_model", "openai/local-mlx")
        api_base = m.get("api_base") or default_api_base
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
    lines.extend(
        [
            "",
            "litellm_settings:",
            "  request_timeout: 120",
            "  num_retries: 2",
            "  drop_params: true",
            "",
            "general_settings:",
            f"  master_key: {master_key}",
        ]
    )
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

    commands: list[tuple[str, Path, list[str]]] = []
    for m in profiles:
        name = m.get("name", "local-mlx")
        if m.get("embedding") and not m.get("hf_model"):
            print(f"[pull-models] Skipping embedding-only profile '{name}' (no hf_model conversion target).")
            continue
        hf_model = m.get("hf_model") or args.model
        q = m.get("quantization", f"{args.quantization}bit").replace("bit", "")
        output_path = m.get("output_path", f"models/{name}")
        output_path_path = Path(output_path)
        output_path_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            python_executable,
            "-m",
            "mlx_lm",
            "convert",
            "--hf-path",
            hf_model,
            "--mlx-path",
            str(output_path_path),
            "--quantize",
            "--q-bits",
            q,
        ]
        commands.append((name, output_path_path, cmd))

    if args.dry_run:
        print("Dry run (commands to execute):\n")
        for name, _, cmd in commands:
            print(f"# Profile: {name}")
            print(" ".join(cmd))
            print("")
        return 0

    rc = 0
    for name, output_path_path, cmd in commands:
        if output_path_path.exists():
            if output_path_path.is_dir():
                if any(output_path_path.iterdir()):
                    print(
                        f"[pull-models] Output path already exists for profile '{name}': {output_path_path}. Skipping."
                    )
                    continue
                output_path_path.rmdir()
            else:
                rc = 2
                print(
                    f"[pull-models] Output path already exists as a file for profile '{name}': {output_path_path}. "
                    "Remove it or set a different output_path in .ai-dev/config.json and retry.",
                    file=sys.stderr,
                )
                if not args.continue_on_error:
                    return rc
                continue

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
