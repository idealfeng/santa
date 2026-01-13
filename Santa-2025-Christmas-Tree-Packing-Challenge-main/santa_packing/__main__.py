"""Single entrypoint for generating a strong, Kaggle-safe submission."""

from __future__ import annotations

from santa_packing.workflow import cli_main


def main() -> int:
    return int(cli_main())


if __name__ == "__main__":
    raise SystemExit(main())
