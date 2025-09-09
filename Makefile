.PHONY: setup poetry-env test train-legal sign-legal offer fetch value-add-smoke value-add-full swarm-sim \
	swarm-v2-smoke swarm-v2-eval monolithic-r4 value-add-rank-sweep dry-run-all dump-policy

LAT=5000

# default to Gemma
BASE_MODEL ?= google/gemma-3-1b-it
EVAL_SPLIT ?= validation

# ---------------------------------------------------------------------------
# Environment helper – shortcut to install with Poetry (preferred).
# ---------------------------------------------------------------------------

setup: poetry-env  ## Install dependencies (dev extras) via Poetry

poetry-env:
	poetry install --with dev

# ---------------------------------------------------------------------------
# Commands executed inside Poetry's virtualenv
# ---------------------------------------------------------------------------

test:
	poetry run pytest -q

train-legal:
	poetry run python -m scripts.train_task --domain legal --epochs 1 --samples 64 --output out/legal

sign-legal:
	poetry run python -c "import pathlib, plora.signer as s, sys; k=pathlib.Path('keys'); k.mkdir(exist_ok=True); priv=k/'temp_priv.pem'; pub=k/'temp_pub.pem';\
	  (priv.exists() or s.generate_keypair(priv, pub))" && \
	poetry run python -m scripts.sign_plasmid --adapter-dir out/legal --private-key keys/temp_priv.pem

offer:
	poetry run python -m scripts.offer_server --root out &
	@echo "Offer server running"

fetch:
	poetry run python -m scripts.fetch_client --domain legal --dest fetched --public-key keys/temp_pub.pem

DOMAINS := arithmetic legal medical

train-all:
	@for d in $(DOMAINS); do \
		poetry run python -m scripts.train_task \
		    --domain $$d --epochs 1 --samples 64 --output out/$$d ; \
	done

sign-all: train-all
	@poetry run python -c "import pathlib,plora.signer as s; \
	    k=pathlib.Path('keys'); k.mkdir(exist_ok=True); \
	    p=k/'temp_priv.pem'; q=k/'temp_pub.pem'; \
	    (p.exists() or s.generate_keypair(p,q))"
	@for d in $(DOMAINS); do \
		poetry run python -m scripts.sign_plasmid \
		    --adapter-dir out/$$d --private-key keys/temp_priv.pem ; \
	done

offer-all: sign-all
	poetry run python -m scripts.offer_server --root out

# ---------------------------------------------------------------------------
# Fetch all plasmids from running server into ./fetched/<domain>
# ---------------------------------------------------------------------------

fetch-all:
	@for d in $(DOMAINS); do \
		poetry run python -m scripts.fetch_client \
		    --domain $$d \
		    --dest fetched/$$d \
		    --public-key keys/temp_pub.pem ; \
	done

# Value-add experiment (small smoke run)
value-add-smoke:
	PLORA_LATENCY_BUDGET_MS=$(LAT) \
	poetry run python -m scripts.run_lora_value_add \
	  --domains arithmetic,legal,medical \
	  --ranks 2 \
	  --schemes attention \
	  --seeds 41 \
	  --samples 128 \
	  --dev-size 512 \
	  --base-model $(BASE_MODEL) \
	  --eval-split $(EVAL_SPLIT)

# Full value-add experiment (longer, deeper grid)
value-add-full:
	PLORA_LATENCY_BUDGET_MS=$(LAT) \
	poetry run python -m scripts.run_lora_value_add \
	  --domains arithmetic,legal,medical \
	  --ranks 2,4,8,16 \
	  --schemes attention,mlp,all \
	  --seeds 41,42,43 \
	  --samples 1000 \
	  --dev-size 512 \
	  --base-model $(BASE_MODEL) \
	  --eval-split $(EVAL_SPLIT) \
	  --resume

# Build value_add.jsonl from artifacts (full evaluation)
value-add-build-full:
	PLORA_FORCE_CPU=$(CPU) \
	poetry run python -m scripts.build_value_add_jsonl \
	  --artifacts-dir results/value_add \
	  --output results/value_add/value_add.jsonl \
	  --domains arithmetic,legal,medical \
	  --dev-size 512 \
	  --base-model $(BASE_MODEL) \
	  --overwrite

# Build with a filter (regex) and smaller dev set for smoke or chunked runs
value-add-build-filter:
	PLORA_FORCE_CPU=$(CPU) \
	poetry run python -m scripts.build_value_add_jsonl \
	  --artifacts-dir results/value_add \
	  --output results/value_add/value_add.jsonl \
	  --domains arithmetic,legal,medical \
	  --dev-size $(DEV) \
	  --base-model $(BASE_MODEL) \
	  --filter $(FILTER)

# Build with very low resource usage (skip placebos & cross, smaller dev and length)
DEV ?= 256
LEN ?= 256
value-add-build-lowmem:
	PLORA_FORCE_CPU=$(CPU) \
	poetry run python -m scripts.build_value_add_jsonl \
	  --artifacts-dir results/value_add \
	  --output results/value_add/value_add.jsonl \
	  --domains arithmetic,legal,medical \
	  --dev-size $(DEV) \
	  --max-length $(LEN) \
	  --base-model $(BASE_MODEL) \
	  --skip-placebos \
	  --skip-cross \
	  --overwrite

# ---------------------------------------------------------------------------
# Swarm simulation
# ---------------------------------------------------------------------------

swarm-sim:
	poetry run python -m swarm.sim_entry --topology line --agents 5 --mode sim --max_rounds 50 --seed 42

# Swarm Sim v2 (push–pull, in-process) – security on, short dry-run
swarm-v2-smoke:
	poetry run python -m swarm.sim_v2_entry --agents 6 --rounds 5 --graph_p 0.25 --security on --trojan_rate 0.3

# Summarise v2 (and v1 graph) reports into a compact JSON
swarm-v2-eval:
	poetry run python -m scripts.evaluate_v2 --reports results --out results/summary_v2.json

# Monolithic baseline – tiny training loop over 3 domains at rank 4
monolithic-r4:
	poetry run python -m scripts.monolithic_train --domains arithmetic,legal,medical --epochs 1 --samples 64 --rank 4 --output out/monolithic_r4

# Rank sweep runner – writes rank-scoped outputs under results/value_add
value-add-rank-sweep:
	PLORA_LATENCY_BUDGET_MS=$(LAT) \
	poetry run python -m scripts.run_lora_value_add \
	  --domains arithmetic,legal,medical \
	  --ranks 4,8,16 \
	  --schemes all \
	  --seeds 42 \
	  --samples 64 \
	  --dev-size 256 \
	  --base-model $(BASE_MODEL) \
	  --eval-split $(EVAL_SPLIT)

# ---------------------------------------------------------------------------
# End-to-end dry run (CPU or small GPU):
# - Trains tiny per-domain adapters
# - Runs Swarm v2 (push–pull) with Security Gate enabled
# - Summarises v2 reports
# - Trains a monolithic baseline (rank 4)
# - Runs a small rank sweep with rank-scoped outputs
#
# Tip: You can globally cap dataset rows via: export PLORA_SAMPLES=64
# ---------------------------------------------------------------------------
dry-run-all:
	@echo "[1/6] Running unit tests" && \
	poetry run pytest -q && \
	echo "[2/6] Training domain adapters" && \
	$(MAKE) train-all && \
	echo "[3/6] Swarm v2 simulation (security on)" && \
	$(MAKE) swarm-v2-smoke && \
	echo "[4/6] Summarising v2 reports" && \
	$(MAKE) swarm-v2-eval && \
	echo "[5/6] Training monolithic baseline (rank 4)" && \
	$(MAKE) monolithic-r4 && \
	echo "[6/6] Value-add rank sweep (small)" && \
	$(MAKE) value-add-rank-sweep && \
	echo "Done. See results/ and out/ for artefacts."

# ---------------------------------------------------------------------------
# Dump effective Security Policy
#
# Reads an optional JSON policy file and prints the resolved policy after
# applying CLI overrides (targets/ranks/signatures/thresholds). Useful to
# verify what the Security Gate will enforce before a run.
#
# Example: make dump-policy POLICY=policy.json TARGETS=assets/allowed_targets.txt RANKS=4,8,16 SIG=off
# ---------------------------------------------------------------------------
POLICY ?=
TARGETS ?= assets/allowed_targets.txt
RANKS ?= 4,8,16
SIG ?= off
dump-policy:
	poetry run python -m scripts.dump_policy \
		$$(test -n "$(POLICY)" && echo --policy_file $(POLICY)) \
		--allowed_targets_file $(TARGETS) \
		--allowed_ranks $(RANKS) \
		--signatures $(SIG)
