from __future__ import annotations

import unittest
from pathlib import Path

from ai_dev.core import init_ops, paths


class TestInitOpsDefaults(unittest.TestCase):
    def test_default_template_files_include_expected_runtime_outputs(self) -> None:
        template_map = {str(path): (content, executable) for path, content, executable in init_ops.DEFAULT_TEMPLATE_FILES}

        self.assertIn("agent/server.py", template_map)
        self.assertIn("agent/http_api.py", template_map)
        self.assertIn("agent/http_service.py", template_map)
        self.assertIn("embedding_worker/worker.py", template_map)
        self.assertTrue(template_map["mlx/entrypoint.sh"][1])

    def test_core_paths_constants_match_expected_locations(self) -> None:
        self.assertEqual(paths.APP_DIR, Path(".ai-dev"))
        self.assertEqual(paths.CONFIG_PATH, Path(".ai-dev/config.json"))
        self.assertEqual(paths.INDEX_PATH, Path(".ai-dev/index.json"))
        self.assertEqual(paths.INDEX_STATE_PATH, Path(".ai-dev/index_state.json"))


if __name__ == "__main__":
    unittest.main()