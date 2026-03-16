from __future__ import annotations

from ai_dev.core import cli_facade as _cli_facade


for _name in _cli_facade.__all__:
    globals()[_name] = getattr(_cli_facade, _name)


__all__ = [*_cli_facade.__all__, "main"]


def main(argv: list[str] | None = None) -> int:
    parser = _cli_facade.build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
