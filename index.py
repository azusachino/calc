#!/usr/bin/env python3
"""Catalog every seed deck, grouped by course.

Reads data/**/*.json and prints one line per deck (status, question count,
title, filename) so the growing seed collection stays surveyable at a glance.
The `course` field inside each file is the grouping key; the containing folder
is the fallback.

Usage:
    uv run index.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def main() -> int:
    by_course: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
    for path in sorted(DATA_DIR.rglob("*.json")):
        deck = json.loads(path.read_text())
        course = deck.get("course") or path.parent.name
        by_course[course].append((path, deck))

    total = 0
    for course in sorted(by_course):
        decks = by_course[course]
        total += len(decks)
        print(f"{course}  ({len(decks)})")
        for path, deck in decks:
            n = len(deck.get("questions", []))
            status = deck.get("status", "draft")
            mark = "active" if status == "active" else status
            print(f"  {mark:7} {n:3}q  {deck.get('title', '?')}  [{path.name}]")
        print()

    print(f"Total: {total} decks / {len(by_course)} courses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
