# ame-platform task runner. Secrets load from .env (gitignored); copy .env.example.
-include .env
export

UV ?= uv

.PHONY: help install check dry-run replay call grade clean age-key seal unseal reseal

help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n", $$1, $$2}'

install: ## Sync the uv environment from the lockfile
	$(UV) sync

check: ## Compile sources and validate seed data (no network)
	$(UV) run python -m py_compile client.py replay.py grade.py main.py
	$(UV) run replay.py --dry-run

dry-run: ## Validate + preview seed import without sending
	$(UV) run replay.py --dry-run

replay: guard-token ## Import every data/*.json into AME
	$(UV) run replay.py

call: guard-token ## Call one tool: make call TOOL=stats.user [PARAMS='{"k":1}']
	$(UV) run client.py $(TOOL) $(PARAMS)

grade: guard-token ## List pending attempts, or grade one: make grade ARGS="<id> 0.8 notes"
	$(UV) run grade.py $(ARGS)

guard-token:
	@test -n "$(AME_TOKEN)" || { \
		echo "AME_TOKEN not set — cp .env.example .env and fill it in"; exit 1; }

clean: ## Remove caches
	rm -rf .venv __pycache__ **/__pycache__

# --- secrets (sops + age) -------------------------------------------------
# secrets.env (encrypted) is committed; .env (plaintext) is gitignored.
age-key: ## Generate this device's age key and print its public recipient
	@KF="$${XDG_CONFIG_HOME:-$$HOME/.config}/sops/age/keys.txt"; \
		mkdir -p "$$(dirname "$$KF")"; \
		[ -f "$$KF" ] || age-keygen -o "$$KF"; \
		echo "add this recipient to .sops.yaml:"; age-keygen -y "$$KF"

seal: ## Encrypt local .env -> committed secrets.env
	@test -f .env || { echo "no .env — cp .env.example .env and fill it in"; exit 1; }
	sops --encrypt --input-type dotenv --output-type dotenv .env > secrets.env
	@echo "sealed -> secrets.env (commit it)"

unseal: ## Decrypt committed secrets.env -> local .env
	@test -f secrets.env || { echo "no secrets.env to decrypt"; exit 1; }
	sops --decrypt --input-type dotenv --output-type dotenv secrets.env > .env
	@echo "unsealed -> .env"

reseal: unseal seal ## Re-encrypt to all current .sops.yaml recipients
