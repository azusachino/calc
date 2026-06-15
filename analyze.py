#!/usr/bin/env python3
"""Weak-area report across AME assessments.

Joins each assessment's per-question correctRate (assessment.stats) with the
question text/tags (question.list) to show what you keep missing and which
tags are weakest — so you know what to practice (and what to seed more of).

Token comes from the environment or a local .env (auto-loaded by from_env()).

Usage:
    uv run analyze.py                 # decks whose title contains 運転免許
    uv run analyze.py --filter Rust   # any title substring
    uv run analyze.py --all           # every active assessment

GOTCHA: the API rate-limits (HTTP 429); every call goes through call() which
backs off and retries, with a small pause between calls.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict

from client import AmeError, from_env

PAUSE = 0.4          # seconds between calls, to stay under the rate limit
MAX_RETRIES = 6


def call(client, tool: str, params: dict | None = None) -> dict:
    """client.run with exponential backoff on HTTP 429."""
    delay = 2.0
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(PAUSE)
            return client.run(tool, params or {})
        except AmeError as e:
            if e.status == 429 and attempt < MAX_RETRIES - 1:
                print(f"  rate-limited, retrying in {delay:.0f}s...", file=sys.stderr)
                time.sleep(delay)
                delay *= 2
                continue
            raise


def all_questions(client) -> dict[str, tuple[str, list[str], str]]:
    """id -> (prompt, tags, kind) over the whole question bank (paginated)."""
    qmap: dict[str, tuple[str, list[str], str]] = {}
    cursor = None
    seen_cursors: set[str] = set()
    for _ in range(40):  # hard cap: the API keeps returning a cursor past the end
        params = {"limit": 100}
        if cursor:
            params["cursor"] = cursor
        r = call(client, "question.list", params)
        batch = r.get("questions", [])
        for q in batch:
            qmap[q["id"]] = (q.get("prompt", ""), q.get("tags") or [], q.get("kind"))
        cursor = r.get("next_cursor")
        # stop on: end-of-data, a short page, or a cursor we've already followed
        if not cursor or not batch or len(batch) < 100 or cursor in seen_cursors:
            break
        seen_cursors.add(cursor)
    return qmap


def pick_decks(client, argv: list[str]) -> list[tuple[str, str]]:
    """(id, title) of assessments to analyze, per --filter / --all."""
    show_all = "--all" in argv
    needle = "運転免許"
    if "--filter" in argv:
        needle = argv[argv.index("--filter") + 1]
    r = call(client, "assessment.list", {"status": "active"})
    out = []
    for a in r.get("assessments", []):
        title = a.get("title", "")
        if show_all or needle in title:
            out.append((a["id"], title))
    return out


def main(argv: list[str]) -> int:
    client = from_env()
    decks = pick_decks(client, argv)
    if not decks:
        print("no matching assessments", file=sys.stderr)
        return 1
    qmap = all_questions(client)

    tag_hits: dict[str, list[float]] = defaultdict(list)
    wrong: list[tuple[float, str, str, str, list[str]]] = []
    deck_avg: list[tuple[float, str]] = []

    for aid, title in decks:
        s = call(client, "assessment.stats", {"id": aid})
        avg = s.get("avg")
        deck_avg.append((avg if avg is not None else 1.0, title))
        for it in s.get("items", []):
            cr = it.get("correctRate")
            if cr is None:
                continue
            prompt, tags, kind = qmap.get(it["questionId"], ("?", [], "?"))
            for t in tags:
                tag_hits[t].append(cr)
            if cr < 1.0:
                wrong.append((cr, title, kind, prompt, tags))

    print("\n=== per-deck average (weakest first) ===")
    for avg, title in sorted(deck_avg):
        print(f"{avg:.3f}  {title}")

    print("\n=== questions missed (correctRate < 1.0) ===")
    if not wrong:
        print("  none — all questions answered correctly")
    for cr, title, kind, prompt, tags in sorted(wrong):
        print(f"{cr:.2f} ({kind}) {prompt[:54]}  {tags}  [{title[:18]}]")

    print("\n=== weak tags (mean correctRate, n) ===")
    rows = [(sum(v) / len(v), len(v), t) for t, v in tag_hits.items() if v]
    for mean, n, t in sorted(rows):
        flag = "  <-- focus" if mean < 0.9 else ""
        print(f"{mean:.2f}  n={n:<3} {t}{flag}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except (AmeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
