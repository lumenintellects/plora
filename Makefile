.PHONY: setup poetry-env test train-legal sign-legal offer fetch value-add-smoke value-add-full swarm-sim \
	swarm-v2-smoke swarm-v2-eval monolithic-r4 value-add-rank-sweep dump-policy train-all sign-all \
	dry-run-lite dry-run-reset dry-run-status full-experiment full-experiment-reset full-experiment-status

# Dynamic domains (from current active YAML config via plora.config)
DOMAINS_CSV := $(shell poetry run python -c 'from plora.config import get; print(",".join(get("domains", [])))')
DOMAINS := $(shell poetry run python -c 'from plora.config import get; print(" ".join(get("domains", [])))')

.PHONY: alt-train-merge
alt-train-merge:
	poetry run python -m scripts.alternating_train_merge \
		--domains $(DOMAINS_CSV) \
		--cycles $$(poetry run python -m plora.config alt_train_merge.cycles) \
		--samples $$(poetry run python -m plora.config alt_train_merge.samples) \
		--rank $$(poetry run python -m plora.config alt_train_merge.rank) \
		--out results/alt_train_merge

.PHONY: ablation
ablation:
	poetry run python -m scripts.ablation_runner --domains $(DOMAINS_CSV) --ranks $$(poetry run python -m plora.config value_add.ranks | tr -d '[] ') --schemes $$(poetry run python -m plora.config value_add.schemes | tr -d "[]'\" ") --samples $$(poetry run python -m plora.config samples) --epochs 1 --out results/ablation.jsonl

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
	poetry run python -m scripts.train_task --domain legal --epochs 1 --output out/legal

sign-legal:
	poetry run python -c "import pathlib, plora.signer as s, sys; k=pathlib.Path('keys'); k.mkdir(exist_ok=True); priv=k/'temp_priv.pem'; pub=k/'temp_pub.pem';\
	  (priv.exists() or s.generate_keypair(priv, pub))" && \
	poetry run python -m scripts.sign_plasmid --adapter-dir out/legal --private-key keys/temp_priv.pem

offer:
	poetry run python -m scripts.offer_server --root out &
	@echo "Offer server running"

fetch:
	poetry run python -m scripts.fetch_client --domain legal --dest fetched --public-key keys/temp_pub.pem

# Replace static domain list with dynamic config-driven list
# (already defined above as DOMAINS / DOMAINS_CSV)

train-all:
	@rank=$$(poetry run python -m plora.config 'value_add.ranks[0]'); \
	samples=$$(poetry run python -m plora.config samples); \
	scheme=$$(poetry run python -m plora.config 'value_add.schemes[0]' | tr -d '"'); \
	for d in $(DOMAINS); do \
		poetry run python -m scripts.train_task \
		  --domain $$d --epochs 1 --samples $$samples --rank $$rank --scheme $$scheme --output out/$$d ; \
	done

sign-all:
	@poetry run python -c "import pathlib,plora.signer as s; \
	    k=pathlib.Path('keys'); k.mkdir(exist_ok=True); \
	    p=k/'temp_priv.pem'; q=k/'temp_pub.pem'; \
	    (p.exists() or s.generate_keypair(p,q))"
	@missing=0; for d in $(DOMAINS); do \
	    if [ ! -f out/$$d/adapter_model.safetensors ]; then \
	        echo "[sign-all] Missing adapter for domain $$d (run 'make train-all' first)"; missing=1; \
	    fi; \
	done; [ $$missing -eq 0 ] || exit 1
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
	poetry run python -m scripts.run_lora_value_add \
	  --domains "$(DOMAINS_CSV)" \
	  --latency-budget-ms $${LAT_BUDGET_MS:-250} \
	  --ignore-latency-guard \
	  --no-resume || true

# Full value-add experiment (longer, deeper grid)
value-add-full:
	poetry run python -m scripts.run_lora_value_add \
	  --domains "$(DOMAINS_CSV)" \
	  --seeds $$(poetry run python -m plora.config value_add.seeds) \
	  --latency-budget-ms $${LAT_BUDGET_MS:-250} \
	  --ignore-latency-guard \
	  --no-resume

# Build value_add.jsonl from artifacts (full evaluation)
value-add-build-full:
	PLORA_FORCE_CPU=$(CPU) \
	poetry run python -m scripts.build_value_add_jsonl \
	  --artifacts-dir results/value_add \
	  --output results/value_add/value_add.jsonl \
	  --domains "$(DOMAINS_CSV)" \
	  --dev-size $(shell poetry run python -m plora.config value_add.dev_size) \
	  --base-model $(shell poetry run python -m plora.config base_model) \
	  --seeds $(shell poetry run python -m plora.config value_add.seeds | tr -d '[] ') \
	  --overwrite

# Build with very low resource usage (skip placebos & cross, smaller dev and length)
value-add-build-lowmem:
	poetry run python -m scripts.build_value_add_jsonl \
	  --artifacts-dir results/value_add \
	  --output results/value_add/value_add.jsonl \
	  --domains "$(DOMAINS_CSV)" \
	  --dev-size $(shell poetry run python -m plora.config value_add.dev_size) \
	  --max-length 256 \
	  --base-model $(shell poetry run python -m plora.config base_model) \
	  --seeds $(shell poetry run python -m plora.config value_add.seeds | tr -d '[] ') \
	  --skip-placebos \
	  --skip-cross \
	  --overwrite

# ---------------------------------------------------------------------------
# Swarm simulation
# ---------------------------------------------------------------------------

swarm-sim:
	poetry run python -m swarm.sim_entry --topology line --agents 5 --mode sim --max_rounds 50 --seed 42

# Swarm Sim v2 (push–pull) – security on, short dry-run
swarm-v2-smoke:
	poetry run python -m swarm.sim_v2_entry --agents 6 --rounds 5 --graph_p $$(poetry run python -m plora.config graph.p) --security on --trojan_rate $$(poetry run python -m plora.config swarm.trojan_rate) --history-alias results/history.json --adapters_dir out

# Summarise v2 (and v1 graph) reports into a compact JSON
swarm-v2-eval:
	poetry run python -m scripts.evaluate_v2 --reports results --out results/summary_v2.json

.PHONY: prepare-data
prepare-data:
	poetry run python -m scripts.preload_datasets

.PHONY: figures
figures:
	poetry run python -m scripts.plot_figures --summary results/summary_v2.json --out results/figures

.PHONY: validate-bounds
validate-bounds:
	poetry run python -m scripts.validate_bounds --ns 20,40,80,160 --p $$(poetry run python -m plora.config graph.p) --seed 42 --out results/bounds_validation.json

.PHONY: calibrate-c calibrate-c-er calibrate-c-ws calibrate-c-ba
calibrate-c: calibrate-c-er calibrate-c-ws calibrate-c-ba

calibrate-c-er:
	poetry run python -m scripts.calibrate_c --topology er --ns 20,40,80,160 --p $$(poetry run python -m plora.config graph.p) --rounds 20 --seed 42 --out results/c_calib_er.json

calibrate-c-ws:
	poetry run python -m scripts.calibrate_c --topology ws --ns 20,40,80,160 --p $$(poetry run python -m plora.config graph.p) --rounds 20 --seed 42 --out results/c_calib_ws.json

calibrate-c-ba:
	poetry run python -m scripts.calibrate_c --topology ba --ns 20,40,80,160 --p $$(poetry run python -m plora.config graph.p) --rounds 20 --seed 42 --out results/c_calib_ba.json

.PHONY: mine-calib
mine-calib:
	poetry run python -m scripts.mine_calibrate --rho 0.8 --n 2000 --out results/mine_calib.json

.PHONY: audit-verify
audit-verify:
	poetry run python -m scripts.audit_verify --audit results/audit/gate_audit.jsonl

.PHONY: consensus-smoke
consensus-smoke:
	@python -c "from swarm.consensus import ConsensusEngine, Vote; c=ConsensusEngine(quorum=2); assert c.vote(Vote(0,1,'A')) is None; assert c.vote(Vote(1,1,'A'))=='A'; assert c.committed(1)=='A'; print('Consensus smoke OK')"

.PHONY: probes-calib
probes-calib:
	poetry run python -m scripts.probes_calibrate --target_fp $$(poetry run python -m plora.config probes.target_fp) --target_fn $$(poetry run python -m plora.config probes.target_fn) --out results/probes_calib.json

.PHONY: net-it
net-it:
	poetry run python -m scripts.net_it_metrics --history results/history.json --out results/net_it_metrics.json

.PHONY: math-export
math-export:
	@echo "Exporting math notebook to PDF (requires nbconvert/pandoc)" && \
	poetry run jupyter nbconvert --to pdf notebooks/math_foundations.ipynb || echo "Install nbconvert/pandoc to enable export."

.PHONY: thesis-sweep
thesis-sweep:
	poetry run python -m scripts.sweep.main --topos er,ws,ba --ns 20,40,80,160 --seeds $$(poetry run python -m plora.config value_add.seeds | tr -d '[] ') --p $$(poetry run python -m plora.config graph.p) --rounds 15 --trojan_rates 0.0,0.3 --out results/thesis_sweep.jsonl

# Monolithic baseline – tiny training loop over 3 domains at rank 4
monolithic-r4:
	poetry run python -m scripts.monolithic_train --domains "$(DOMAINS_CSV)" --epochs 1 --samples $$(poetry run python -m plora.config samples) --rank 4 --output out/monolithic_r4

# Rank sweep runner – writes rank-scoped outputs under results/value_add
value-add-rank-sweep:
	poetry run python -m scripts.run_lora_value_add \
	  --domains "$(DOMAINS_CSV)" \
	  --latency-budget-ms $${LAT_BUDGET_MS:-250} \
	  --ignore-latency-guard || true
# ---------------------------------------------------------------------------
# Minimal dry run:
# ---------------------------------------------------------------------------
.PHONY: config-use-full config-use-dry
config-use-full:
	poetry run python -m plora.config use config/plora.full.yml

config-use-dry:
	poetry run python -m plora.config use config/plora.dry.yml

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
		run_step 1 "Running unit tests" poetry run pytest -q && \
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
		run_step 16 "Consensus-enabled v2 smoke" poetry run python -m swarm.sim_v2_entry --agents 4 --rounds 2 --graph er --graph_p $$(poetry run python -m plora.config graph.p) --seed 9 --security on --trojan_rate $$(poetry run python -m plora.config swarm.trojan_rate) --consensus on --quorum $$(poetry run python -m plora.config swarm.quorum) --report_dir results --adapters_dir out && \
		run_step 17 "gRPC offer/fetch demo" sh -c '( poetry run python -m scripts.offer_server --root out & OFFER_PID=$$!; sleep 2; poetry run python -m scripts.fetch_client --domain legal --dest fetched --public-key keys/temp_pub.pem || true; kill $$OFFER_PID 2>/dev/null || true )' && \
		( \
			done=$$(get_done); \
			if [ $$done -ge 18 ]; then \
				echo "[18/$(TOTAL_STEPS)] Dump effective security policy - SKIPPED (already done)"; \
			else \
				echo "[18/$(TOTAL_STEPS)] Dump effective security policy"; \
				RANKS_STR=$$(poetry run python -m plora.config allowed_ranks | tr -d '[] '); \
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
		run_step 1 "Running unit tests" poetry run pytest -q && \
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
		run_step 16 "Consensus-enabled v2 smoke" poetry run python -m swarm.sim_v2_entry --agents 6 --rounds 3 --graph er --graph_p $$(poetry run python -m plora.config graph.p) --seed 11 --security on --trojan_rate $$(poetry run python -m plora.config swarm.trojan_rate) --consensus on --quorum $$(poetry run python -m plora.config swarm.quorum) --report_dir results --adapters_dir out && \
		run_step 17 "gRPC offer/fetch demo" sh -c '( poetry run python -m scripts.offer_server --root out & OFFER_PID=$$!; sleep 2; poetry run python -m scripts.fetch_client --domain legal --dest fetched --public-key keys/temp_pub.pem || true; kill $$OFFER_PID 2>/dev/null || true )' && \
		( \
			done=$$(get_done); \
			if [ $$done -ge 18 ]; then \
				echo "[18/$(TOTAL_STEPS)] Dump effective security policy - SKIPPED (already done)"; \
			else \
				echo "[18/$(TOTAL_STEPS)] Dump effective security policy"; \
				RANKS_STR=$$(poetry run python -m plora.config allowed_ranks | tr -d '[] '); \
				$(MAKE) dump-policy RANKS="$$RANKS_STR" && mark_done 18; \
			fi \
		) && \
		run_step 19 "Net IT metrics" $(MAKE) net-it && \
		echo "" && \
		echo "════════════════════════════════════════════════════════" && \
		echo "  ✅ Full experiment complete! See results/ and out." && \
		echo "════════════════════════════════════════════════════════"; \
	} 2>&1 | tee $$log_file

# Artefacts check – verify presence of key outputs for analysis
.PHONY: artefacts-check
artefacts-check:
	@poetry run python - <<'PY'
	from pathlib import Path
	from plora.config import get as cfg
	root = Path('.')
	domains = cfg('domains', ['arithmetic','legal','medical'])
	checks = [
	    ("Swarm v2 summary", Path("results/summary_v2.json")),
	    ("Figures dir", Path("results/figures")),
	    ("Value-add JSONL", Path("results/value_add/value_add.jsonl")),
	    ("Thesis sweep", Path("results/thesis_sweep.jsonl")),
	    ("Calibration C (ER)", Path("results/c_calib_er.json")),
	    ("Bounds validation", Path("results/bounds_validation.json")),
	    ("Net IT metrics", Path("results/net_it_metrics.json")),
	    ("Monolithic baseline dir", Path("out/monolithic_r4")),
	]
	missing = []
	for name, p in checks:
	    if not p.exists():
	        missing.append(f"{name}: {p}")
	sv2 = list(Path('results').glob('swarm_v2_report_*.json'))
	if not sv2:
	    missing.append("Swarm v2 raw report: results/swarm_v2_report_*.json")
	for d in domains:
	    adir = Path('out')/d
	    if not adir.exists():
	        missing.append(f"Adapter dir missing: {adir}")
	        continue
	    if not (adir/"plora.yml").exists():
	        missing.append(f"Manifest missing: {(adir/'plora.yml')}")
	    if not (adir/"adapter_model.safetensors").exists():
	        missing.append(f"Weights missing: {(adir/'adapter_model.safetensors')}")
	fetched_any = Path('fetched').exists()
	if missing:
	    print("[artefacts-check] MISSING artefacts:")
	    for m in missing:
	        print(" -", m)
	    if not fetched_any:
	        print("[artefacts-check] Note: fetched/ not found (gRPC demo optional)")
	    raise SystemExit(1)
	else:
	    print("[artefacts-check] All key artefacts present.")
	    if fetched_any:
	        print("[artefacts-check] gRPC fetched/ present (demo OK)")
	PY

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
	poetry run python -m scripts.dump_policy \
		$(if $(POLICY),--policy_file $(POLICY)) \
		$(if $(TARGETS),--allowed_targets_file $(TARGETS)) \
		$(if $(RANKS),--allowed_ranks $(RANKS)) \
		--signatures $(SIG)
