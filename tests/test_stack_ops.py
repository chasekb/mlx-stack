from __future__ import annotations

import argparse
import tempfile
import unittest
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
            "stack": {"mlx_api_base": "http://host.containers.internal:8081/v1"},
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
        self.assertIn("api_base: http://host.containers.internal:8081/v1", rendered)

    def test_command_up_starts_host_mlx_before_compose(self) -> None:
        calls: list[list[str]] = []
        reachability = iter([False, True])

        def load_config():
            return {
                "stack": {
                    "mlx_api_base": "http://host.containers.internal:8081/v1",
                    "mlx_model_path": "models/local-mlx",
                    "mlx_python": ".venv/bin/python",
                    "mlx_port": 8081,
                }
            }

        def compose_command():
            return ["podman", "compose", "-f", "podman-compose.yml"]

        def run_fn(cmd):
            calls.append(cmd)
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
            self.assertEqual(calls, [["podman", "compose", "-f", "podman-compose.yml", "up", "-d"]])
            state = stack_ops._read_mlx_state(app_dir / "mlx_host_process.json")
            self.assertIsNotNone(state)
            assert state is not None
            self.assertEqual(state["pid"], 4242)
            self.assertIn("mlx_lm", " ".join(state["command"]))

    def test_command_down_stops_managed_host_mlx(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app_dir = Path(tmpdir) / ".ai-dev"
            state_path = app_dir / "mlx_host_process.json"
            stack_ops._write_mlx_state(state_path, {"pid": 5151, "api_base": "http://host.containers.internal:8081/v1"})

            rc = stack_ops.command_down(
                argparse.Namespace(),
                compose_command_fn=lambda: ["podman", "compose"],
                run_fn=lambda cmd: 0,
                app_dir=app_dir,
                pid_is_alive_fn=lambda _pid: False,
            )

            self.assertEqual(rc, 0)
            self.assertFalse(state_path.exists())


if __name__ == "__main__":
    unittest.main()