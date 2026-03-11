from __future__ import annotations

import argparse
import unittest

from ai_dev import command_groups


def _ok_handler(_: argparse.Namespace) -> int:
    return 0


def _build_handlers() -> dict[str, command_groups.CommandHandler]:
    return {key: _ok_handler for key in command_groups.REQUIRED_HANDLER_KEYS}


class TestCommandGroupsValidation(unittest.TestCase):
    def test_validate_command_handlers_accepts_complete_mapping(self) -> None:
        handlers = _build_handlers()
        command_groups.validate_command_handlers(handlers)

    def test_validate_command_handlers_rejects_missing_keys(self) -> None:
        handlers = _build_handlers()
        del handlers["command_init"]

        with self.assertRaises(ValueError) as ctx:
            command_groups.validate_command_handlers(handlers)

        self.assertIn("missing: command_init", str(ctx.exception))

    def test_validate_command_handlers_rejects_unexpected_keys(self) -> None:
        handlers = _build_handlers()
        handlers["command_unexpected"] = _ok_handler

        with self.assertRaises(ValueError) as ctx:
            command_groups.validate_command_handlers(handlers)

        self.assertIn("unexpected: command_unexpected", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
