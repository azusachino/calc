#!/usr/bin/env python3
"""Dependency-free client for the AME agent API.

Single door: every tool dispatches through POST /v1/agents/run with
{tool, params} and an Authorization: Bearer <apiKey> header.

GOTCHA: the host is Cloudflare-fronted and 403s python-urllib's default
User-Agent ("error code: 1010"), so we always send a normal UA.

Usage:
    AME_TOKEN=<id>_<secret> uv run client.py stats.user
    AME_TOKEN=... uv run client.py assessment.list '{"status":"active"}'
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_API = "https://ame.azusachino.icu"
USER_AGENT = "curl/8.7.1"  # anything non-urllib gets past Cloudflare rule 1010


class AmeError(RuntimeError):
    """A non-2xx response from the AME dispatcher."""

    def __init__(self, status: int, body: str):
        self.status = status
        self.body = body
        super().__init__(f"HTTP {status}: {body}")


class AmeClient:
    def __init__(self, token: str, api: str = DEFAULT_API):
        if not token:
            raise ValueError("token is required (set AME_TOKEN)")
        self.api = api.rstrip("/")
        self.token = token

    def run(self, tool: str, params: dict | None = None) -> dict:
        """Dispatch one tool through the single door and return its result."""
        payload = json.dumps({"tool": tool, "params": params or {}}).encode()
        req = urllib.request.Request(
            f"{self.api}/v1/agents/run",
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "User-Agent": USER_AGENT,
            },
        )
        try:
            with urllib.request.urlopen(req) as resp:
                body = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            raise AmeError(e.code, e.read().decode(errors="replace")) from None
        if not body.get("ok"):
            raise AmeError(200, json.dumps(body))
        return body.get("result", {})


def from_env() -> AmeClient:
    return AmeClient(os.environ.get("AME_TOKEN", ""), os.environ.get("AME_API", DEFAULT_API))


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    tool = argv[0]
    params = json.loads(argv[1]) if len(argv) > 1 else {}
    result = from_env().run(tool, params)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except (AmeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
