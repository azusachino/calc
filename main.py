#!/usr/bin/env python3
"""ame-platform — entrypoint pointing at the two tools in this repo."""


def main() -> None:
    print(
        "ame-platform tools:\n"
        "  uv run client.py <tool> [json-params]   # call one AME agent tool\n"
        "  uv run replay.py [--dry-run] [files...]  # import data/*.json seeds\n"
        "\nSet AME_TOKEN (and optionally AME_API) first."
    )


if __name__ == "__main__":
    main()
