# ame-platform

Tools and replayable seed data for [AME](https://ame.azusachino.icu) — an
assessment learning platform with a first-class agent API. Managed as a uv
project.

## What's here

| File | Purpose |
| --- | --- |
| `client.py` | Dependency-free AME agent client. Every tool dispatches through `POST /v1/agents/run`. Sends a normal `User-Agent` so Cloudflare doesn't 403 it. |
| `replay.py` | Imports `data/*.json` seed assessments (validates the data-model rules first). |
| `grade.py` | Lists pending `essay`/`code` attempts and grades them via `attempt.grade`. |
| `analyze.py` | Weak-area report — joins `assessment.stats` correctRate with `question.list` tags to rank what to practice. Backs off on HTTP 429. |
| `index.py` | Catalog of every seed deck grouped by `course` (`make index`) — keeps the growing collection surveyable. |
| `data/<course>/*.json` | Raw, replayable assessment seeds, grouped into per-course subfolders — each file is one `assessment.create` payload with inline questions. Scanned recursively. |

The workflow end to end: **mint** (`data/*.json`) → **publish** (`make replay`) →
take the assessment in the AME UI → **grade** pending essay/code (`make grade`) →
**analyze** (`make call TOOL=assessment.stats PARAMS='{"id":"..."}'`).

## Secrets across devices (sops + age)

The token never enters git in plaintext. `secrets.env` (encrypted) is committed;
`.env` (decrypted) is gitignored. Each device holds its own age private key
outside the repo, at `$XDG_CONFIG_HOME/sops/age/keys.txt` (falling back to
`~/.config/sops/age/keys.txt`) — the path sops reads and the one `make age-key`
writes. `.sops.yaml` lists every device's public key as a recipient.

> macOS note: sops only uses `~/Library/Application Support/sops/age/keys.txt`
> when `XDG_CONFIG_HOME` is unset. If your shell exports it (common with
> nix/fish), keep the key under `$XDG_CONFIG_HOME` or `make unseal` fails with
> "did not find keys".

First device (you have the token):

```bash
cp .env.example .env        # paste the real AME_TOKEN
make seal                   # .env -> secrets.env (commit secrets.env)
```

New device:

```bash
make age-key                # prints this device's public key
# add that key to .sops.yaml `age:`, then on an already-trusted device:
make reseal                 # re-encrypt secrets.env to all recipients, commit
# back on the new device, after pulling:
make unseal                 # secrets.env -> .env
```

Daily use after `make unseal`: both the Makefile and the scripts themselves
auto-load `.env` (the latter via `load_dotenv()` in `client.py`), so `make
replay`, `make call`, and a bare `uv run analyze.py` all just work — no
`set -a; source .env` needed.

## Usage

```bash
uv run replay.py --dry-run          # validate seeds, no network
uv run replay.py                    # import every data/**/*.json
uv run replay.py data/tooling/dev-tooling.json   # import one
uv run index.py                     # catalog decks grouped by course

uv run client.py stats.user                          # ad-hoc tool call
uv run client.py assessment.list '{"status":"active"}'

uv run analyze.py                   # weak-area report (decks matching 運転免許)
uv run analyze.py --all             # every active assessment
```

## Adding seed data

Drop a JSON file in `data/<course>/` (e.g. `data/driving/`) shaped like
`assessment.create` params: `title`, `mode` (`practice`|`graded`),
`method: "agent"`, `course` (the subfolder / grouping key), `objectives[]`, and
an inline `questions[]`. `replay.py` scans `data/` recursively, so a new course
is just a new subfolder. Run `make index` to see the catalog. Set
`status: "active"` to publish on import. Question payloads:

- `mc` → `{ "options": ["bare", "strings"], "correct_index": 0 }`
- `tf` → `{ "correct": true }`
- `short` → `{ "accepted": [...], "normalize": "exact", "judge": "exact" }`
- `code` / `essay` → land as pending attempts, graded later via `attempt.grade`

`replay.py` checks these rules before sending, since violations otherwise 422
or silently mis-grade.
