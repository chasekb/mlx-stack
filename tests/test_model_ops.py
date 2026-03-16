from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from ai_dev.core import model_ops


class TestModelOps(unittest.TestCase):
    def test_command_configure_cursor_prefers_host_mlx_endpoint_and_model(self) -> None:
        written: dict[str, str] = {}

        cfg = {
            "stack": {
                "mlx_api_base": "http://host.containers.internal:8081/v1",
                "default_model": "mlx-community/Qwen3.5-2B-4bit",
            },
            "models": [
                {
                    "name": "local-mlx",
                    "mlx_model": "mlx-community/Qwen3.5-2B-4bit",
                    "hf_model": "Qwen/Qwen3.5-2B",
                    "tags": ["quality", "default"],
                }
            ],
            "cursor": {
                "base_url": "http://localhost:4000/v1",
                "api_key": "local-dev",
                "model": "local-mlx",
            },
        }

        def write_file(path: Path, content: str, executable: bool = False) -> None:
            del executable
            written[str(path)] = content

        with tempfile.TemporaryDirectory() as tmpdir:
            rc = model_ops.command_configure_cursor(
                argparse.Namespace(base_url=None, api_key=None, model=None, task_tag=None),
                load_config_fn=lambda: cfg,
                write_file_fn=write_file,
                app_dir=Path(tmpdir),
                task_tag_aliases={"default": ["default"]},
            )

        self.assertEqual(rc, 0)
        saved = json.loads(next(iter(written.values())))
        self.assertEqual(saved["name"], "Local MLX")
        self.assertEqual(saved["baseUrl"], "http://localhost:8081/v1")
        self.assertEqual(saved["model"], "mlx-community/Qwen3.5-2B-4bit")

    def test_command_configure_cursor_task_tag_uses_selected_host_model(self) -> None:
        written: dict[str, str] = {}
        cfg = {
            "stack": {
                "mlx_api_base": "http://host.containers.internal:8081/v1",
                "default_model": "mlx-community/Qwen3.5-2B-4bit",
            },
            "models": [
                {
                    "name": "local-mlx-fast",
                    "mlx_model": "mlx-community/Qwen3.5-0.8B-4bit",
                    "tags": ["fast", "default"],
                },
                {
                    "name": "local-mlx",
                    "mlx_model": "mlx-community/Qwen3.5-2B-4bit",
                    "tags": ["quality"],
                },
            ],
            "cursor": {
                "base_url": "http://localhost:4000/v1",
                "api_key": "local-dev",
                "model": "local-mlx",
            },
        }

        def write_file(path: Path, content: str, executable: bool = False) -> None:
            del executable
            written[str(path)] = content

        with tempfile.TemporaryDirectory() as tmpdir:
            rc = model_ops.command_configure_cursor(
                argparse.Namespace(base_url=None, api_key=None, model=None, task_tag="fast"),
                load_config_fn=lambda: cfg,
                write_file_fn=write_file,
                app_dir=Path(tmpdir),
                task_tag_aliases={"fast": ["fast", "default"]},
            )

        self.assertEqual(rc, 0)
        saved = json.loads(next(iter(written.values())))
        self.assertEqual(saved["model"], "mlx-community/Qwen3.5-0.8B-4bit")


if __name__ == "__main__":
    unittest.main()