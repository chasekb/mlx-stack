from __future__ import annotations

import argparse
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from ai_dev.core import stack_ops


class _FakeProc:
    def __init__(self, pid: int = 4242, returncode: int | None = None) -> None:
        self.pid = pid
        self.returncode = returncode

    def poll(self):
        return self.returncode


class TestStackOpsHostMlx(unittest.TestCase):
    def test_generate_litellm_config_prefers_stack_mlx_api_base(self) -> None:
        cfg = {
            "stack": {"mlx_api_base": "http://host.containers.internal:8082/v1"},
            "models": [
                {
                    "name": "local-mlx",
                    "backend_model": "openai/local-mlx",
                    "api_key": "local-dev",
                }
            ],
            "cursor": {"api_key": "local-dev"},
        }

        rendered = stack_ops.generate_litellm_config(cfg, cfg["models"])
        self.assertIn("api_base: http://host.containers.internal:8082/v1", rendered)
        self.assertIn("litellm_settings:", rendered)
        self.assertIn("request_timeout: 120", rendered)

    def test_build_host_mlx_command_passes_native_server_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            cfg = {
                "stack": {
                    "mlx_model_path": "models/local-mlx",
                    "mlx_port": 8082,
                    "mlx_prompt_cache_size": 8,
                    "mlx_prompt_cache_bytes": "4GB",
                    "mlx_decode_concurrency": 2,
                    "mlx_prompt_concurrency": 3,
                    "mlx_draft_model_path": "models/local-mlx-fast",
                    "mlx_num_draft_tokens": 4,
                }
            }

            cmd = stack_ops._build_host_mlx_command(cfg, project_root=project_root, python_executable="python3")

        self.assertIn("--prompt-cache-size", cmd)
        self.assertIn("8", cmd)
        self.assertIn("--prompt-cache-bytes", cmd)
        self.assertIn("4GB", cmd)
        self.assertIn("--decode-concurrency", cmd)
        self.assertIn("--prompt-concurrency", cmd)
        self.assertIn("--draft-model", cmd)
        self.assertIn("--num-draft-tokens", cmd)
        self.assertIn("4", cmd)

    def test_command_up_starts_host_mlx_before_compose(self) -> None:
        calls: list[tuple[list[str], dict | None]] = []
        reachability = iter([False, True])

        def load_config():
            return {
                "stack": {
                    "mlx_api_base": "http://host.containers.internal:8082/v1",
                    "mlx_model_path": "models/local-mlx",
                    "mlx_python": ".venv/bin/python",
                    "mlx_port": 8082,
                }
            }

        def compose_command():
            return ["podman", "compose", "-f", "podman-compose.yml"]

        def run_fn(cmd, env=None):
            calls.append((cmd, env))
            return 0

        fake_proc = _FakeProc()

        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            app_dir = project_root / ".ai-dev"
            (project_root / ".venv" / "bin").mkdir(parents=True, exist_ok=True)
            (project_root / ".venv" / "bin" / "python").write_text("", encoding="utf-8")

            rc = stack_ops.command_up(
                argparse.Namespace(with_optional=False),
                compose_command_fn=compose_command,
                run_fn=run_fn,
                load_config_fn=load_config,
                app_dir=app_dir,
                project_root=project_root,
                python_executable="python3",
                popen_fn=lambda *args, **kwargs: fake_proc,
                sleep_fn=lambda _: None,
                endpoint_reachable_fn=lambda *_args, **_kwargs: next(reachability),
                pid_is_alive_fn=lambda _pid: False,
            )

            self.assertEqual(rc, 0)
            self.assertEqual(
                calls,
                [(["podman", "compose", "-f", "podman-compose.yml", "up", "-d"], {"COMPOSE_PROFILES": "optional"})],
            )
            state = stack_ops._read_mlx_state(app_dir / "mlx_host_process.json")
            self.assertIsNotNone(state)
            assert state is not None
            self.assertEqual(state["pid"], 4242)
            self.assertIn("mlx_lm", " ".join(state["command"]))

    def test_command_status_reports_service_health_and_embedding_config(self) -> None:
        calls: list[tuple[list[str], dict | None]] = []

        def run_fn(cmd, env=None):
            calls.append((cmd, env))
            return 0

        cfg = {
            "stack": {
                "mlx_api_base": "http://host.containers.internal:8082/v1",
                "litellm_port": 4000,
                "spec_router_port": 8092,
                "embed_queue_port": 8093,
                "mlx_prompt_cache_size": 4,
                "embed_model": "local-embed",
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            app_dir = Path(tmpdir) / ".ai-dev"
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = stack_ops.command_status(
                    argparse.Namespace(),
                    compose_command_fn=lambda: ["podman", "compose"],
                    run_fn=run_fn,
                    load_config_fn=lambda: cfg,
                    app_dir=app_dir,
                    endpoint_reachable_fn=lambda url, **_kwargs: "8082" in url or "4000" in url,
                )

        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("host-mlx: reachable", out)
        self.assertIn("host-mlx acceleration", out)
        self.assertIn("litellm: reachable", out)
        self.assertIn("agent: unreachable", out)
        self.assertIn("local-embed", out)
        self.assertEqual(calls, [(["podman", "compose", "ps"], None)])

    def test_command_down_stops_managed_host_mlx(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            app_dir = Path(tmpdir) / ".ai-dev"
            state_path = app_dir / "mlx_host_process.json"
            stack_ops._write_mlx_state(state_path, {"pid": 5151, "api_base": "http://host.containers.internal:8082/v1"})

            rc = stack_ops.command_down(
                argparse.Namespace(),
                compose_command_fn=lambda: ["podman", "compose"],
                run_fn=lambda cmd, env=None: 0,
                app_dir=app_dir,
                project_root=project_root,
                pid_is_alive_fn=lambda _pid: False,
            )

            self.assertEqual(rc, 0)
            self.assertFalse(state_path.exists())

    def test_command_down_force_cleans_failed_compose_teardown(self) -> None:
        calls: list[tuple[list[str], dict | None]] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            compose_file = project_root / "podman-compose.yml"
            compose_file.write_text(
                """
services:
  litellm:
    container_name: ai-dev-litellm
  agent:
    container_name: ai-dev-agent
""".strip()
                + "\n",
                encoding="utf-8",
            )
            app_dir = project_root / ".ai-dev"

            def run_fn(cmd, env=None):
                calls.append((cmd, env))
                if cmd == ["podman", "compose", "down"]:
                    return 1
                return 0

            rc = stack_ops.command_down(
                argparse.Namespace(),
                compose_command_fn=lambda: ["podman", "compose"],
                run_fn=run_fn,
                app_dir=app_dir,
                project_root=project_root,
                compose_file=compose_file,
                pid_is_alive_fn=lambda _pid: False,
            )

            self.assertEqual(rc, 0)
            self.assertEqual(
                calls,
                [
                    (["podman", "compose", "down"], {"COMPOSE_PROFILES": "optional"}),
                    (["podman", "stop", "-t", "0", "ai-dev-litellm", "ai-dev-agent"], None),
                    (["podman", "rm", "-f", "ai-dev-litellm", "ai-dev-agent"], None),
                    (["podman", "pod", "rm", "-f", f"{project_root.name}_default"], None),
                    (["podman", "network", "rm", f"{project_root.name}_default"], None),
                ],
            )

    def test_command_pull_models_skips_existing_output_dir(self) -> None:
        calls: list[list[str]] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            output_dir = project_root / "models" / "local-mlx"
            output_dir.mkdir(parents=True)
            (output_dir / "config.json").write_text("{}", encoding="utf-8")
            (output_dir / "model.safetensors").write_text("stub", encoding="utf-8")

            def load_config():
                return {
                    "models": [
                        {
                            "name": "local-mlx",
                            "hf_model": "Qwen/Qwen2.5-Coder-3B-Instruct",
                            "output_path": str(output_dir),
                            "quantization": "4bit",
                        }
                    ]
                }

            rc = stack_ops.command_pull_models(
                argparse.Namespace(profile=None, model="Qwen/Qwen2.5-Coder-7B-Instruct", quantization="4", dry_run=False, continue_on_error=False),
                load_config_fn=load_config,
                python_executable="python3",
                subprocess_run_fn=lambda cmd: calls.append(cmd) or _FakeProc(returncode=0),
            )

        self.assertEqual(rc, 0)
        self.assertEqual(calls, [])

    def test_command_pull_models_skips_embedding_only_profiles(self) -> None:
        calls: list[list[str]] = []

        def load_config():
            return {
                "models": [
                    {
                        "name": "local-embed",
                        "embedding": True,
                        "output_path": "models/local-embed",
                    }
                ]
            }

        rc = stack_ops.command_pull_models(
            argparse.Namespace(profile=None, model="Qwen/Qwen2.5-Coder-7B-Instruct", quantization="4", dry_run=False, continue_on_error=False),
            load_config_fn=load_config,
            python_executable="python3",
            subprocess_run_fn=lambda cmd: calls.append(cmd) or _FakeProc(returncode=0),
        )

        self.assertEqual(rc, 0)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()