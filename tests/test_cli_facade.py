from __future__ import annotations

import unittest

from ai_dev import cli
from ai_dev.core import cli_facade


class TestCliFacadeCompatibility(unittest.TestCase):
    def test_cli_re_exports_core_facade_symbols(self) -> None:
        self.assertIs(cli.build_parser, cli_facade.build_parser)
        self.assertIs(cli.command_init, cli_facade.command_init)
        self.assertIs(cli.command_index, cli_facade.command_index)
        self.assertIs(cli.command_memory_explain, cli_facade.command_memory_explain)
        self.assertIs(cli.resolve_model_for_tag, cli_facade.resolve_model_for_tag)


if __name__ == "__main__":
    unittest.main()