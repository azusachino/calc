#!/usr/bin/env python3
"""Replay raw seed data into AME.

Each file in data/*.json is one `assessment.create` payload (inline questions,
optionally `status: "active"` to publish in a single call). Replaying is just
dispatching `assessment.create` for each file, so the data is the source of
truth and the import is reproducible.

Seed files live under data/<course>/*.json (the dir is scanned recursively).

Usage:
    uv run replay.py --dry-run            # validate + print, no network
    AME_TOKEN=... uv run replay.py        # import every data/**/*.json
    AME_TOKEN=... uv run replay.py data/leetcode/arrays-two-pointers.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from client import AmeError, from_env

DATA_DIR = Path(__file__).parent / "data"

VALID_KINDS = {"mc", "tf", "short", "essay", "code"}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def validate(payload: dict, path: Path) -> list[str]:
    """Catch the data-model rules that 422 (or silently mis-grade) before sending."""
    errors: list[str] = []
    for field in ("title", "mode", "method"):
        if not payload.get(field):
            errors.append(f"{path.name}: missing required field '{field}'")
    if payload.get("mode") not in ("practice", "graded"):
        errors.append(f"{path.name}: mode must be practice|graded")
    questions = payload.get("questions", [])
    if not questions:
        errors.append(f"{path.name}: needs >=1 question to publish")
    for i, q in enumerate(questions):
        where = f"{path.name} q{i}"
        kind = q.get("kind")
        if kind not in VALID_KINDS:
            errors.append(f"{where}: bad kind {kind!r}")
        payload_q = q.get("payload", {})
        if kind == "mc":
            opts = payload_q.get("options")
            if not isinstance(opts, list) or not all(isinstance(o, str) for o in opts):
                errors.append(f"{where}: mc options must be bare strings")
            if not isinstance(payload_q.get("correct_index"), int):
                errors.append(f"{where}: mc needs integer correct_index")
        elif kind == "tf" and not isinstance(payload_q.get("correct"), bool):
            errors.append(f"{where}: tf needs boolean correct")
    return errors


def files(args: list[str]) -> list[Path]:
    paths = [Path(a) for a in args if not a.startswith("-")]
    return sorted(paths) if paths else sorted(DATA_DIR.rglob("*.json"))


def main(argv: list[str]) -> int:
    dry_run = "--dry-run" in argv
    targets = files(argv)
    if not targets:
        print(f"no seed files found in {DATA_DIR}", file=sys.stderr)
        return 1

    all_errors: list[str] = []
    payloads: list[tuple[Path, dict]] = []
    for path in targets:
        payload = load(path)
        errs = validate(payload, path)
        all_errors.extend(errs)
        payloads.append((path, payload))
    if all_errors:
        print("validation failed:", file=sys.stderr)
        for e in all_errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    if dry_run:
        for path, payload in payloads:
            n = len(payload.get("questions", []))
            status = payload.get("status", "draft")
            print(f"[dry-run] {path.name}: '{payload['title']}' "
                  f"({payload['mode']}, {status}, {n} questions)")
        return 0

    client = from_env()
    for path, payload in payloads:
        try:
            result = client.run("assessment.create", payload)
        except AmeError as e:
            print(f"FAIL {path.name}: {e}", file=sys.stderr)
            return 1
        aid = result.get("id", "?")
        print(f"OK   {path.name}: '{payload['title']}' -> {aid}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
