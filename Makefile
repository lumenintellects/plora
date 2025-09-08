.PHONY: setup poetry-env test train-legal sign-legal offer fetch value-add-smoke value-add-full

LAT=400

# default to Gemma
BASE_MODEL ?= google/gemma-3-1b-it

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
	  --base-model $(BASE_MODEL)

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


