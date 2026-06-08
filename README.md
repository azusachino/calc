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
| `data/*.json` | Raw, replayable assessment seeds — each file is one `assessment.create` payload with inline questions. |

The workflow end to end: **mint** (`data/*.json`) → **publish** (`make replay`) →
take the assessment in the AME UI → **grade** pending essay/code (`make grade`) →
**analyze** (`make call TOOL=assessment.stats PARAMS='{"id":"..."}'`).

## Secrets across devices (sops + age)

The token never enters git in plaintext. `secrets.env` (encrypted) is committed;
`.env` (decrypted) is gitignored. Each device holds its own age private key
outside the repo (`~/Library/Application Support/sops/age/keys.txt` on macOS);
`.sops.yaml` lists every device's public key as a recipient.

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

Daily use after `make unseal`: the Makefile auto-loads `.env`, so `make replay`
and `make call` just work.

## Usage

```bash
uv run replay.py --dry-run          # validate seeds, no network
uv run replay.py                    # import every data/*.json
uv run replay.py data/dev-tooling.json   # import one

uv run client.py stats.user                          # ad-hoc tool call
uv run client.py assessment.list '{"status":"active"}'
```

## Adding seed data

Drop a JSON file in `data/` shaped like `assessment.create` params: `title`,
`mode` (`practice`|`graded`), `method: "agent"`, `objectives[]`, and an inline
`questions[]`. Set `status: "active"` to publish on import. Question payloads:

- `mc` → `{ "options": ["bare", "strings"], "correct_index": 0 }`
- `tf` → `{ "correct": true }`
- `short` → `{ "accepted": [...], "normalize": "exact", "judge": "exact" }`
- `code` / `essay` → land as pending attempts, graded later via `attempt.grade`

`replay.py` checks these rules before sending, since violations otherwise 422
or silently mis-grade.
