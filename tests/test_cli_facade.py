from __future__ import annotations

import unittest

from ai_dev import cli
from ai_dev.core import cli_facade


class TestCliFacadeCompatibility(unittest.TestCase):
    def test_cli_re_exports_core_facade_symbols(self) -> None:
        for name in cli_facade.__all__:
            with self.subTest(name=name):
                self.assertIn(name, cli.__all__)
                self.assertIs(getattr(cli, name), getattr(cli_facade, name))

    def test_cli_all_adds_main_on_top_of_facade_exports(self) -> None:
        self.assertEqual(cli.__all__, [*cli_facade.__all__, "main"])


if __name__ == "__main__":
    unittest.main()