#!/usr/bin/env python3
"""Grade pending essay/code attempts on AME.

essay and code answers land as pending attempts (there is no code runner), and
are resolved with attempt.grade { id, score 0.0-1.0, notes? }. This closes the
loop in the workflow: mint -> publish -> take -> grade -> analyze.

Usage:
    uv run grade.py                                   # list pending attempts
    uv run grade.py <attempt-id> <score> [notes...]   # grade one (0.0-1.0)
"""

from __future__ import annotations

import json
import sys

from client import AmeError, from_env


def prompts_by_id(client) -> dict:
    """Map question id -> prompt so the grader sees what was asked."""
    try:
        questions = client.run("question.list", {"limit": 200}).get("questions", [])
    except AmeError:
        return {}
    return {q.get("id"): q.get("prompt", "") for q in questions}


def answer_text(response) -> str:
    """Best-effort extraction of the learner's free-text answer."""
    if not isinstance(response, dict):
        return str(response)
    for key in ("text", "answer", "code", "essay", "value", "body"):
        if response.get(key):
            return str(response[key])
    return json.dumps(response, ensure_ascii=False)


def list_pending(client) -> int:
    attempts = client.run("attempt.list", {"limit": 100}).get("attempts", [])
    pending = [a for a in attempts if a.get("grade_status") != "graded"]
    if not pending:
        print("no pending attempts to grade")
        return 0
    prompts = prompts_by_id(client)
    for a in pending:
        print(f"\nattempt {a['id']}  ({a.get('grade_status')})")
        print(f"  Q: {prompts.get(a.get('question_id'), a.get('question_id'))}")
        print(f"  A: {answer_text(a.get('response', {}))}")
    print(f"\n{len(pending)} pending. Grade: uv run grade.py <attempt-id> <0.0-1.0> [notes]")
    return 0


def grade_one(client, attempt_id: str, score_str: str, notes: list[str]) -> int:
    score = float(score_str)
    if not 0.0 <= score <= 1.0:
        raise ValueError("score must be between 0.0 and 1.0")
    params: dict = {"id": attempt_id, "score": score}
    if notes:
        params["notes"] = " ".join(notes)
    client.run("attempt.grade", params)
    print(f"graded {attempt_id}: score={score}")
    return 0


def main(argv: list[str]) -> int:
    client = from_env()
    if not argv:
        return list_pending(client)
    if len(argv) < 2:
        print("usage: uv run grade.py <attempt-id> <0.0-1.0> [notes...]", file=sys.stderr)
        return 2
    return grade_one(client, argv[0], argv[1], argv[2:])


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except (AmeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
