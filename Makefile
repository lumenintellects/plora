.PHONY: setup poetry-env test train-legal sign-legal offer fetch value-add-smoke value-add-full swarm-sim \
	swarm-v2-smoke swarm-v2-eval monolithic-r4 value-add-rank-sweep dump-policy train-all sign-all \
	dry-run-lite dry-run-reset dry-run-status full-experiment full-experiment-reset full-experiment-status \
	docker-build docker-run docker-test docker-dry-run

# Resolve poetry: prefer PATH, fall back to python3 -m poetry (pip --user install)
POETRY := $(if $(shell command -v poetry 2>/dev/null),poetry,python3 -m poetry)

# Dynamic domains (from current active YAML config via plora.config)
DOMAINS_CSV := $(shell $(POETRY) run python -c 'from plora.config import get; print(",".join(get("domains", [])))' 2>/dev/null)
DOMAINS := $(shell $(POETRY) run python -c 'from plora.config import get; print(" ".join(get("domains", [])))' 2>/dev/null)

.PHONY: alt-train-merge
alt-train-merge:
	$(POETRY) run python -m scripts.alternating_train_merge \
		--domains $(DOMAINS_CSV) \
		--cycles $$($(POETRY) run python -m plora.config alt_train_merge.cycles) \
		--samples $$($(POETRY) run python -m plora.config alt_train_merge.samples) \
		--rank $$($(POETRY) run python -m plora.config alt_train_merge.rank) \
		--out results/alt_train_merge

.PHONY: ablation
ablation:
	$(POETRY) run python -m scripts.ablation_runner --domains $(DOMAINS_CSV) --ranks $$($(POETRY) run python -m plora.config value_add.ranks | tr -d '[] ') --schemes $$($(POETRY) run python -m plora.config value_add.schemes | tr -d "[]'\" ") --samples $$($(POETRY) run python -m plora.config samples) --epochs 1 --out results/ablation.jsonl

# ---------------------------------------------------------------------------
# Environment helper – shortcut to install with Poetry (preferred).
# ---------------------------------------------------------------------------

setup: poetry-env  ## Install dependencies (dev extras) via Poetry

poetry-env:
	@$(POETRY) --version >/dev/null 2>&1 || { \
		echo "Poetry not found - installing via pip..."; \
		python3 -m pip install --user poetry; \
	}
	$(POETRY) install --with dev

# ---------------------------------------------------------------------------
# Commands executed inside Poetry's virtualenv
# ---------------------------------------------------------------------------

test:
	$(POETRY) run pytest -q

train-legal:
	$(POETRY) run python -m scripts.train_task --domain legal --epochs 1 --output out/legal

sign-legal:
	$(POETRY) run python -c "import pathlib, plora.signer as s, sys; k=pathlib.Path('keys'); k.mkdir(exist_ok=True); priv=k/'temp_priv.pem'; pub=k/'temp_pub.pem';\
	  (priv.exists() or s.generate_keypair(priv, pub))" && \
	$(POETRY) run python -m scripts.sign_plasmid --adapter-dir out/legal --private-key keys/temp_priv.pem

offer:
	$(POETRY) run python -m scripts.offer_server --root out &
	@echo "Offer server running"

fetch:
	$(POETRY) run python -m scripts.fetch_client --domain legal --dest fetched --public-key keys/temp_pub.pem

# Replace static domain list with dynamic config-driven list
# (already defined above as DOMAINS / DOMAINS_CSV)

train-all:
	@rank=$$($(POETRY) run python -m plora.config 'value_add.ranks[0]'); \
	samples=$$($(POETRY) run python -m plora.config samples); \
	scheme=$$($(POETRY) run python -m plora.config 'value_add.schemes[0]' | tr -d '"'); \
	for d in $(DOMAINS); do \
		$(POETRY) run python -m scripts.train_task \
		  --domain $$d --epochs 1 --samples $$samples --rank $$rank --scheme $$scheme --output out/$$d ; \
	done

sign-all:
	@$(POETRY) run python -c "import pathlib,plora.signer as s; \
	    k=pathlib.Path('keys'); k.mkdir(exist_ok=True); \
	    p=k/'temp_priv.pem'; q=k/'temp_pub.pem'; \
	    (p.exists() or s.generate_keypair(p,q))"
	@missing=0; for d in $(DOMAINS); do \
	    if [ ! -f out/$$d/adapter_model.safetensors ]; then \
	        echo "[sign-all] Missing adapter for domain $$d (run 'make train-all' first)"; missing=1; \
	    fi; \
	done; [ $$missing -eq 0 ] || exit 1
	@for d in $(DOMAINS); do \
		$(POETRY) run python -m scripts.sign_plasmid \
		    --adapter-dir out/$$d --private-key keys/temp_priv.pem ; \
	done

offer-all: sign-all
	$(POETRY) run python -m scripts.offer_server --root out

# ---------------------------------------------------------------------------
# Fetch all plasmids from running server into ./fetched/<domain>
# ---------------------------------------------------------------------------

fetch-all:
	@for d in $(DOMAINS); do \
		$(POETRY) run python -m scripts.fetch_client \
		    --domain $$d \
		    --dest fetched/$$d \
		    --public-key keys/temp_pub.pem ; \
	done

# Value-add experiment (small smoke run)
value-add-smoke:
	$(POETRY) run python -m scripts.run_lora_value_add \
	  --domains "$(DOMAINS_CSV)" \
	  --latency-budget-ms $${LAT_BUDGET_MS:-250} \
	  --ignore-latency-guard \
	  --no-resume || true

# Full value-add experiment (longer, deeper grid)
value-add-full:
	$(POETRY) run python -m scripts.run_lora_value_add \
	  --domains "$(DOMAINS_CSV)" \
	  --seeds $$($(POETRY) run python -m plora.config value_add.seeds) \
	  --latency-budget-ms $${LAT_BUDGET_MS:-250} \
	  --ignore-latency-guard \
	  --no-resume

# Build value_add.jsonl from artifacts (full evaluation)
value-add-build-full:
	PLORA_FORCE_CPU=$(CPU) \
	$(POETRY) run python -m scripts.build_value_add_jsonl \
	  --artifacts-dir results/value_add \
	  --output results/value_add/value_add.jsonl \
	  --domains "$(DOMAINS_CSV)" \
	  --dev-size $(shell $(POETRY) run python -m plora.config value_add.dev_size) \
	  --base-model $(shell $(POETRY) run python -m plora.config base_model) \
	  --seeds $(shell $(POETRY) run python -m plora.config value_add.seeds | tr -d '[] ') \
	  --overwrite

# Build with very low resource usage (skip placebos & cross, smaller dev and length)
value-add-build-lowmem:
	$(POETRY) run python -m scripts.build_value_add_jsonl \
	  --artifacts-dir results/value_add \
	  --output results/value_add/value_add.jsonl \
	  --domains "$(DOMAINS_CSV)" \
	  --dev-size $(shell $(POETRY) run python -m plora.config value_add.dev_size) \
	  --max-length 256 \
	  --base-model $(shell $(POETRY) run python -m plora.config base_model) \
	  --seeds $(shell $(POETRY) run python -m plora.config value_add.seeds | tr -d '[] ') \
	  --skip-placebos \
	  --skip-cross \
	  --overwrite

# ---------------------------------------------------------------------------
# Swarm simulation
# ---------------------------------------------------------------------------

swarm-sim:
	$(POETRY) run python -m swarm.sim_entry --topology line --agents 5 --mode sim --max_rounds 50 --seed 42

# Swarm Sim v2 (push–pull) – security on, short dry-run
swarm-v2-smoke:
	$(POETRY) run python -m swarm.sim_v2_entry --agents 6 --rounds 5 --graph_p $$($(POETRY) run python -m plora.config graph.p) --security on --trojan_rate $$($(POETRY) run python -m plora.config swarm.trojan_rate) --history-alias results/history.json --adapters_dir out

# Summarise v2 (and v1 graph) reports into a compact JSON
swarm-v2-eval:
	$(POETRY) run python -m scripts.evaluate_v2 --reports results --out results/summary_v2.json

.PHONY: prepare-data
prepare-data:
	$(POETRY) run python -m scripts.preload_datasets

.PHONY: figures
figures:
	$(POETRY) run python -m scripts.plot_figures --summary results/summary_v2.json --out results/figures

.PHONY: validate-bounds
validate-bounds:
	$(POETRY) run python -m scripts.validate_bounds --ns 20,40,80,160 --p $$($(POETRY) run python -m plora.config graph.p) --seed 42 --out results/bounds_validation.json

.PHONY: calibrate-c calibrate-c-er calibrate-c-ws calibrate-c-ba
calibrate-c: calibrate-c-er calibrate-c-ws calibrate-c-ba

calibrate-c-er:
	$(POETRY) run python -m scripts.calibrate_c --topology er --ns 20,40,80,160 --p $$($(POETRY) run python -m plora.config graph.p) --rounds 20 --seed 42 --out results/c_calib_er.json

calibrate-c-ws:
	$(POETRY) run python -m scripts.calibrate_c --topology ws --ns 20,40,80,160 --p $$($(POETRY) run python -m plora.config graph.p) --rounds 20 --seed 42 --out results/c_calib_ws.json

calibrate-c-ba:
	$(POETRY) run python -m scripts.calibrate_c --topology ba --ns 20,40,80,160 --p $$($(POETRY) run python -m plora.config graph.p) --rounds 20 --seed 42 --out results/c_calib_ba.json

.PHONY: mine-calib
mine-calib:
	$(POETRY) run python -m scripts.mine_calibrate --rho 0.8 --n 2000 --out results/mine_calib.json

.PHONY: audit-verify
audit-verify:
	$(POETRY) run python -m scripts.audit_verify --audit results/audit/gate_audit.jsonl

.PHONY: consensus-smoke
consensus-smoke:
	@python -c "from swarm.consensus import ConsensusEngine, Vote; c=ConsensusEngine(quorum=2); assert c.vote(Vote(0,1,'A')) is None; assert c.vote(Vote(1,1,'A'))=='A'; assert c.committed(1)=='A'; print('Consensus smoke OK')"

.PHONY: probes-calib
probes-calib:
	$(POETRY) run python -m scripts.probes_calibrate --target_fp $$($(POETRY) run python -m plora.config probes.target_fp) --target_fn $$($(POETRY) run python -m plora.config probes.target_fn) --out results/probes_calib.json

.PHONY: net-it
net-it:
	$(POETRY) run python -m scripts.net_it_metrics --history results/history.json --out results/net_it_metrics.json

.PHONY: math-export
math-export:
	@echo "Exporting math notebook to PDF (requires nbconvert/pandoc)" && \
	$(POETRY) run jupyter nbconvert --to pdf notebooks/math_foundations.ipynb || echo "Install nbconvert/pandoc to enable export."

.PHONY: thesis-sweep thesis-sweep-full
thesis-sweep:
	$(POETRY) run python -m scripts.sweep.main --topos er,ws,ba --ns 20,40,80,160 --seeds $$($(POETRY) run python -m plora.config value_add.seeds | tr -d '[] ') --p $$($(POETRY) run python -m plora.config graph.p) --rounds 6 --trojan_rates 0.0,0.3 --out results/thesis_sweep.jsonl

# Full thesis sweep: 3 topologies × 4 sizes × 3 seeds × 2 trojan_rates = 72 experiments
# Use this for final analysis; includes per-round coverage, entropy, and accepted counts
# 6 rounds is sufficient: max t_obs=2, max t_pred=4, MI→0 by round 4
thesis-sweep-full:
	$(POETRY) run python -m scripts.sweep.main --topos er,ws,ba --ns 20,40,80,160 --seeds 41,42,43 --p $$($(POETRY) run python -m plora.config graph.p) --rounds 6 --trojan_rates 0.0,0.3 --out results/thesis_sweep.jsonl
	@echo "Generated 72 experiments (3 topos × 4 sizes × 3 seeds × 2 trojan_rates)"
	@wc -l results/thesis_sweep.jsonl

# Monolithic baseline – tiny training loop over 3 domains at rank 4
monolithic-r4:
	$(POETRY) run python -m scripts.monolithic_train --domains "$(DOMAINS_CSV)" --epochs 1 --samples $$($(POETRY) run python -m plora.config samples) --rank 4 --output out/monolithic_r4

# Rank sweep runner – writes rank-scoped outputs under results/value_add
value-add-rank-sweep:
	$(POETRY) run python -m scripts.run_lora_value_add \
	  --domains "$(DOMAINS_CSV)" \
	  --latency-budget-ms $${LAT_BUDGET_MS:-250} \
	  --ignore-latency-guard || true
# ---------------------------------------------------------------------------
# Minimal dry run:
# ---------------------------------------------------------------------------
.PHONY: config-use-full config-use-dry
config-use-full:
	$(POETRY) run python -m plora.config use config/plora.full.yml

config-use-dry:
	$(POETRY) run python -m plora.config use config/plora.dry.yml

# ---------------------------------------------------------------------------
# Dry-run lite (19 steps) - same structure as full-experiment but with dry config
# Same structure as full-experiment but with dry config
# Use `make dry-run-reset` to start fresh
# Use `make dry-run-status` to see current progress
# ---------------------------------------------------------------------------
DRYRUN_STATE := results/.dryrun_state

.PHONY: dry-run-lite dry-run-reset dry-run-status
dry-run-reset:
	@rm -f $(DRYRUN_STATE)
	@echo "Dry-run state cleared. Next run will start from step 1."

dry-run-status:
	@if [ -f $(DRYRUN_STATE) ]; then \
		completed=$$(cat $(DRYRUN_STATE)); \
		echo "Completed steps: $$completed/$(TOTAL_STEPS)"; \
		if [ $$completed -lt $(TOTAL_STEPS) ]; then \
			echo "Next step: $$((completed + 1))"; \
		else \
			echo "Dry-run complete!"; \
		fi; \
	else \
		echo "No dry-run in progress (will start from step 1)"; \
	fi

dry-run-lite: config-use-dry
	@mkdir -p logs results
	@export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0; \
	echo "MPS enabled (PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0)"; \
	ts=$$(date +%Y%m%d-%H%M%S); \
	log_file=logs/dry-run-lite-$$ts.log; \
	STATE=$(DRYRUN_STATE); \
	get_done() { [ -f "$$STATE" ] && cat "$$STATE" || echo 0; }; \
	mark_done() { echo $$1 > "$$STATE"; }; \
	run_step() { \
		step=$$1; shift; desc="$$1"; shift; \
		done=$$(get_done); \
		if [ $$done -ge $$step ]; then \
			echo "[$$step/$(TOTAL_STEPS)] $$desc - SKIPPED (already done)"; \
			return 0; \
		fi; \
		echo "[$$step/$(TOTAL_STEPS)] $$desc"; \
		if "$$@"; then \
			mark_done $$step; \
			return 0; \
		else \
			echo "FAILED at step $$step. Re-run 'make dry-run-lite' to resume."; \
			return 1; \
		fi; \
	}; \
	{ \
		run_step 1 "Running unit tests" $(POETRY) run pytest -q && \
		run_step 2 "Preloading datasets" $(MAKE) prepare-data && \
		run_step 3 "Calibrating spectral constant C (all topologies)" $(MAKE) calibrate-c && \
		run_step 4 "Validating spectral/Cheeger bounds" $(MAKE) validate-bounds && \
		run_step 5 "Calibrating probes" $(MAKE) probes-calib && \
		run_step 6 "Training per-domain adapters" $(MAKE) train-all && \
		run_step 7 "Signing adapters" $(MAKE) sign-all && \
		run_step 8 "Swarm v2 simulation (security on)" $(MAKE) swarm-v2-smoke && \
		run_step 9 "Summarising v2 reports" $(MAKE) swarm-v2-eval && \
		run_step 10 "Training monolithic baseline (rank 4)" $(MAKE) monolithic-r4 && \
		run_step 11 "Value-add rank sweep" $(MAKE) value-add-rank-sweep && \
		run_step 12 "Thesis-scale sweep (ER/WS/BA)" $(MAKE) thesis-sweep && \
		run_step 13 "Ablations (rank/scheme)" $(MAKE) ablation && \
		run_step 14 "Alternating train-merge stability" $(MAKE) alt-train-merge && \
		run_step 15 "Value-add JSONL verification" sh -c 'test -f results/value_add/value_add.jsonl && echo "  ✓ value_add.jsonl has $$(wc -l < results/value_add/value_add.jsonl | tr -d " ") records"' && \
		run_step 16 "Consensus-enabled v2 smoke" $(POETRY) run python -m swarm.sim_v2_entry --agents 4 --rounds 2 --graph er --graph_p $$($(POETRY) run python -m plora.config graph.p) --seed 9 --security on --trojan_rate $$($(POETRY) run python -m plora.config swarm.trojan_rate) --consensus on --quorum $$($(POETRY) run python -m plora.config swarm.quorum) --report_dir results --adapters_dir out && \
		run_step 17 "gRPC offer/fetch demo" sh -c '( $(POETRY) run python -m scripts.offer_server --root out & OFFER_PID=$$!; sleep 2; $(POETRY) run python -m scripts.fetch_client --domain legal --dest fetched --public-key keys/temp_pub.pem || true; kill $$OFFER_PID 2>/dev/null || true )' && \
		( \
			done=$$(get_done); \
			if [ $$done -ge 18 ]; then \
				echo "[18/$(TOTAL_STEPS)] Dump effective security policy - SKIPPED (already done)"; \
			else \
				echo "[18/$(TOTAL_STEPS)] Dump effective security policy"; \
				RANKS_STR=$$($(POETRY) run python -m plora.config allowed_ranks | tr -d '[] '); \
				$(MAKE) dump-policy RANKS="$$RANKS_STR" && mark_done 18; \
			fi \
		) && \
		run_step 19 "Net IT metrics" $(MAKE) net-it && \
		echo "" && \
		echo "════════════════════════════════════════════════════════" && \
		echo "  ✅ Dry-run lite complete! See results/ and out." && \
		echo "════════════════════════════════════════════════════════"; \
	} 2>&1 | tee $$log_file

# ---------------------------------------------------------------------------
# Full experiment (19 steps):
# - Tests, calibrations, and validations (steps 1-5)
# - Train & sign per-domain adapters (steps 6-7)
# - Swarm v2 (security on) + evaluation (steps 8-9)
# - Monolithic baseline + value-add rank sweep (steps 10-11)
# - Thesis sweep + ablations + convergence (steps 12-14)
# - Build value_add.jsonl with full placebos (step 15) - RQ1 statistical tests
# - Consensus, gRPC demo, policy dump, IT metrics (steps 16-19)
#
# Supports resuming: tracks completed steps in results/.experiment_state
# Use `make full-experiment-reset` to start fresh
# Use `make full-experiment-status` to see current progress
# ---------------------------------------------------------------------------
EXPERIMENT_STATE := results/.experiment_state
TOTAL_STEPS := 19

.PHONY: full-experiment full-experiment-reset full-experiment-status
full-experiment-reset:
	@rm -f $(EXPERIMENT_STATE)
	@echo "Experiment state cleared. Next run will start from step 1."

full-experiment-status:
	@if [ -f $(EXPERIMENT_STATE) ]; then \
		completed=$$(cat $(EXPERIMENT_STATE)); \
		echo "Completed steps: $$completed/$(TOTAL_STEPS)"; \
		if [ $$completed -lt $(TOTAL_STEPS) ]; then \
			echo "Next step: $$((completed + 1))"; \
		else \
			echo "Experiment complete!"; \
		fi; \
	else \
		echo "No experiment in progress (will start from step 1)"; \
	fi

full-experiment: config-use-full
	@mkdir -p logs results
	@export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0; \
	echo "MPS enabled (PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0)"; \
	ts=$$(date +%Y%m%d-%H%M%S); \
	log_file=logs/full-experiment-$$ts.log; \
	STATE=$(EXPERIMENT_STATE); \
	get_done() { [ -f "$$STATE" ] && cat "$$STATE" || echo 0; }; \
	mark_done() { echo $$1 > "$$STATE"; }; \
	run_step() { \
		step=$$1; shift; desc="$$1"; shift; \
		done=$$(get_done); \
		if [ $$done -ge $$step ]; then \
			echo "[$$step/$(TOTAL_STEPS)] $$desc - SKIPPED (already done)"; \
			return 0; \
		fi; \
		echo "[$$step/$(TOTAL_STEPS)] $$desc"; \
		if "$$@"; then \
			mark_done $$step; \
			return 0; \
		else \
			echo "FAILED at step $$step. Re-run 'make full-experiment' to resume."; \
			return 1; \
		fi; \
	}; \
	{ \
		run_step 1 "Running unit tests" $(POETRY) run pytest -q && \
		run_step 2 "Preloading datasets" $(MAKE) prepare-data && \
		run_step 3 "Calibrating spectral constant C (all topologies)" $(MAKE) calibrate-c && \
		run_step 4 "Validating spectral/Cheeger bounds" $(MAKE) validate-bounds && \
		run_step 5 "Calibrating probes" $(MAKE) probes-calib && \
		run_step 6 "Training per-domain adapters" $(MAKE) train-all && \
		run_step 7 "Signing adapters" $(MAKE) sign-all && \
		run_step 8 "Swarm v2 simulation (security on)" $(MAKE) swarm-v2-smoke && \
		run_step 9 "Summarising v2 reports" $(MAKE) swarm-v2-eval && \
		run_step 10 "Training monolithic baseline (rank 4)" $(MAKE) monolithic-r4 && \
		run_step 11 "Value-add rank sweep" $(MAKE) value-add-rank-sweep && \
		run_step 12 "Thesis-scale sweep (ER/WS/BA, 72 experiments)" $(MAKE) thesis-sweep-full && \
		run_step 13 "Ablations (rank/scheme)" $(MAKE) ablation && \
		run_step 14 "Alternating train-merge stability" $(MAKE) alt-train-merge && \
		run_step 15 "Value-add JSONL verification" sh -c 'test -f results/value_add/value_add.jsonl && echo "  ✓ value_add.jsonl has $$(wc -l < results/value_add/value_add.jsonl | tr -d " ") records"' && \
		run_step 16 "Consensus-enabled v2 smoke" $(POETRY) run python -m swarm.sim_v2_entry --agents 6 --rounds 3 --graph er --graph_p $$($(POETRY) run python -m plora.config graph.p) --seed 11 --security on --trojan_rate $$($(POETRY) run python -m plora.config swarm.trojan_rate) --consensus on --quorum $$($(POETRY) run python -m plora.config swarm.quorum) --report_dir results --adapters_dir out && \
		run_step 17 "gRPC offer/fetch demo" sh -c '( $(POETRY) run python -m scripts.offer_server --root out & OFFER_PID=$$!; sleep 2; $(POETRY) run python -m scripts.fetch_client --domain legal --dest fetched --public-key keys/temp_pub.pem || true; kill $$OFFER_PID 2>/dev/null || true )' && \
		( \
			done=$$(get_done); \
			if [ $$done -ge 18 ]; then \
				echo "[18/$(TOTAL_STEPS)] Dump effective security policy - SKIPPED (already done)"; \
			else \
				echo "[18/$(TOTAL_STEPS)] Dump effective security policy"; \
				RANKS_STR=$$($(POETRY) run python -m plora.config allowed_ranks | tr -d '[] '); \
				$(MAKE) dump-policy RANKS="$$RANKS_STR" && mark_done 18; \
			fi \
		) && \
		run_step 19 "Net IT metrics" $(MAKE) net-it && \
		echo "" && \
		echo "════════════════════════════════════════════════════════" && \
		echo "  ✅ Full experiment complete! See results/ and out." && \
		echo "════════════════════════════════════════════════════════"; \
	} 2>&1 | tee $$log_file

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
TARGETS ?=
RANKS ?=
SIG ?= off

dump-policy:
	$(POETRY) run python -m scripts.dump_policy \
		$(if $(POLICY),--policy_file $(POLICY)) \
		$(if $(TARGETS),--allowed_targets_file $(TARGETS)) \
		$(if $(RANKS),--allowed_ranks $(RANKS)) \
		--signatures $(SIG)

# ---------------------------------------------------------------------------
# Docker
# Pass HF_TOKEN for gated models: make docker-test HF_TOKEN=hf_xxxxx
#
# Memory: training loads a 1B-param model twice (train + baseline PPL) which
# peaks at ~10-12 GB RSS.
#
#   macOS / Colima : VM has NO swap → OOM kill (exit 137).
#                    Run `make docker-setup-swap` once after each `colima start`.
#   Linux (native) : host swap is used automatically; ensure ≥8 GB swap
#                    (`free -h`; add with `sudo fallocate …` if needed).
#   Windows / WSL2 : set memory+swap in %USERPROFILE%\.wslconfig:
#                      [wsl2]
#                      memory=10GB
#                      swap=8GB
#
# The host-side HF cache is mounted so model weights are downloaded once.
# Override HF_CACHE to point elsewhere:
#   make docker-dry-run HF_TOKEN=hf_xxx HF_CACHE=/tmp/hf
# ---------------------------------------------------------------------------
DOCKER_IMG  := plora-swarm
DOCKER_HF   := $(if $(HF_TOKEN),-e HF_TOKEN=$(HF_TOKEN))
HF_CACHE    ?= $(HOME)/.cache/huggingface
DOCKER_VOL  := -v $$(pwd)/results:/app/results -v $$(pwd)/out:/app/out -v $(HF_CACHE):/app/.hf_cache
SWAP_SIZE   ?= 16384
# Force offline mode after initial download so PEFT/transformers never retry
# network requests (DNS failures cause memory-wasting retry loops).
# MALLOC settings force glibc to return freed memory to the OS immediately.
DOCKER_ENV  := -e HF_HUB_OFFLINE=1 \
               -e TRANSFORMERS_OFFLINE=1 \
               -e MALLOC_TRIM_THRESHOLD_=0 \
               -e MALLOC_ARENA_MAX=2

# macOS only: create disk-backed swap inside the Colima VM.
# Safe to skip on Linux (native swap) and WSL2 (configured via .wslconfig).
# Replaces any existing swapfile so you can resize with SWAP_SIZE=<MB>.
.PHONY: docker-setup-swap
docker-setup-swap:
	@echo "Setting up $(SWAP_SIZE) MB swap inside Colima VM..."
	@colima ssh -- sh -c '\
		if swapon --show 2>/dev/null | grep -q /swapfile; then \
			cur=$$(stat -c%s /swapfile 2>/dev/null || echo 0); \
			want=$$(($(SWAP_SIZE) * 1048576)); \
			if [ "$$cur" -eq "$$want" ]; then \
				echo "Swap already active ($(SWAP_SIZE) MB):"; swapon --show; exit 0; \
			fi; \
			echo "Resizing swap from $$((cur/1048576)) MB to $(SWAP_SIZE) MB..."; \
			sudo swapoff /swapfile; \
		fi; \
		sudo dd if=/dev/zero of=/swapfile bs=1M count=$(SWAP_SIZE) status=progress && \
		sudo chmod 600 /swapfile && \
		sudo mkswap /swapfile && \
		sudo swapon /swapfile && \
		echo "$(SWAP_SIZE) MB swap enabled"'

docker-build:
	docker build -t $(DOCKER_IMG) .

# Pre-download the base model into the mounted HF cache (online, one-time).
# Subsequent docker-dry-run / docker-run use TRANSFORMERS_OFFLINE=1 so no network needed.
.PHONY: docker-prefetch
docker-prefetch:
	docker run --rm $(DOCKER_HF) -v $(HF_CACHE):/app/.hf_cache $(DOCKER_IMG) \
		python -c "from transformers import AutoModelForCausalLM,AutoTokenizer; \
		AutoTokenizer.from_pretrained('google/gemma-3-1b-it'); \
		AutoModelForCausalLM.from_pretrained('google/gemma-3-1b-it'); \
		print('Model cached')"

docker-run:
	docker run --rm -it $(DOCKER_HF) $(DOCKER_ENV) $(DOCKER_VOL) $(DOCKER_IMG)

docker-test:
	docker run --rm $(DOCKER_HF) $(DOCKER_ENV) -v $(HF_CACHE):/app/.hf_cache $(DOCKER_IMG) make test

docker-dry-run:
	docker run --rm $(DOCKER_HF) $(DOCKER_ENV) $(DOCKER_VOL) $(DOCKER_IMG) make dry-run-lite
