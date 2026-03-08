from __future__ import annotations

import unittest
import uuid
from pathlib import Path

from agent.server import ROOT, tool_write_patch


class TestAgentWritePatch(unittest.TestCase):
    def test_dry_run_is_blocked(self) -> None:
        out = tool_write_patch({"patch": "diff --git a/x b/x\n"}, dry_run=True)
        self.assertEqual(out.get("error"), "blocked_in_dry_run")

    def test_deny_git_path(self) -> None:
        patch = """diff --git a/.git/config b/.git/config
--- a/.git/config
+++ b/.git/config
@@ -1 +1 @@
-x
+y
"""
        out = tool_write_patch({"patch": patch}, dry_run=False)
        self.assertEqual(out.get("error"), "patch_target_denied")
        self.assertIn(".git/config", out.get("denied_paths", []))

    def test_apply_patch_success(self) -> None:
        rel = f".ai-dev/test_write_patch_{uuid.uuid4().hex}.txt"
        target = ROOT / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("alpha\n", encoding="utf-8")

        patch = f"""diff --git a/{rel} b/{rel}
--- a/{rel}
+++ b/{rel}
@@ -1 +1 @@
-alpha
+beta
"""
        try:
            out = tool_write_patch({"patch": patch}, dry_run=False)
            self.assertTrue(out.get("ok"), msg=str(out))
            self.assertEqual(target.read_text(encoding="utf-8"), "beta\n")
        finally:
            if target.exists():
                target.unlink()


if __name__ == "__main__":
    unittest.main()
