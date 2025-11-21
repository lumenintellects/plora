.PHONY: setup poetry-env test train-legal sign-legal offer fetch value-add-smoke value-add-full swarm-sim \
	swarm-v2-smoke swarm-v2-eval monolithic-r4 value-add-rank-sweep dry-run-lite dump-policy full-experiment train-all sign-all

# Dynamic domains (from current active YAML config via plora.config)
DOMAINS_CSV := $(shell poetry run python -c 'from plora.config import get; print(",".join(get("domains", [])))')
DOMAINS := $(shell poetry run python -c 'from plora.config import get; print(" ".join(get("domains", [])))')

.PHONY: alt-train-merge
alt-train-merge:
	poetry run python -m scripts.alternating_train_merge --domains $(DOMAINS_CSV) --cycles 2 --samples 32 --rank 4 --out results/alt_train_merge

.PHONY: ablation
ablation:
	poetry run python -m scripts.ablation_runner --domains $(DOMAINS_CSV) --ranks 2,4,8 --schemes attention,all --samples 64 --epochs 1 --out results/ablation.jsonl

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
	@for d in $(DOMAINS); do \
		poetry run python -m scripts.train_task \
		  --domain $$d --epochs 1 --output out/$$d ; \
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

# Swarm Sim v2 (push–pull, in-process) – security on, short dry-run
swarm-v2-smoke:
	poetry run python -m swarm.sim_v2_entry --agents 6 --rounds 5 --graph_p $$(poetry run python -m plora.config graph.p) --security on --trojan_rate 0.3 --history-alias results/history.json

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

.PHONY: calibrate-c
calibrate-c:
	poetry run python -m scripts.calibrate_c --topology er --ns 20,40,80,160 --p $$(poetry run python -m plora.config graph.p) --rounds 20 --seed 42 --out results/c_calib_er.json

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
	poetry run python -m scripts.probes_calibrate --target_fp 0.05 --target_fn 0.1 --out results/probes_calib.json

.PHONY: net-it
net-it:
	poetry run python -m scripts.net_it_metrics --history results/history.json --out results/net_it_metrics.json

.PHONY: math-export
math-export:
	@echo "Exporting math notebook to PDF (requires nbconvert/pandoc)" && \
	poetry run jupyter nbconvert --to pdf notebooks/math_foundations.ipynb || echo "Install nbconvert/pandoc to enable export."

.PHONY: thesis-sweep
thesis-sweep:
	poetry run python -m scripts.sweep.main --topos er,ws,ba --ns 20,40,80,160 --seeds 41,42 --p 0.25 --rounds 10 --trojan_rates 0.0,0.3 --out results/thesis_sweep.jsonl

# Monolithic baseline – tiny training loop over 3 domains at rank 4
monolithic-r4:
	poetry run python -m scripts.monolithic_train --domains "$(DOMAINS_CSV)" --epochs 1 --samples $$(poetry run python -m plora.config samples) --rank 4 --output out/monolithic_r4

# Rank sweep runner – writes rank-scoped outputs under results/value_add
value-add-rank-sweep:
	poetry run python -m scripts.run_lora_value_add \
	  --domains "$(DOMAINS_CSV)" \
	  --latency-budget-ms $${LAT_BUDGET_MS:-250} \
	  --ignore-latency-guard \
	  --no-resume || true
# ---------------------------------------------------------------------------
# Minimal dry run:
# ---------------------------------------------------------------------------
.PHONY: config-use-full config-use-dry
config-use-full:
	poetry run python -m plora.config use config/plora.full.yml

config-use-dry:
	poetry run python -m plora.config use config/plora.dry.yml

.PHONY: dry-run-lite
dry-run-lite: config-use-dry
	@mkdir -p logs
	@ts=$$(date +%Y%m%d-%H%M%S); \
	log_file=logs/dry-run-lite-$$ts.log; \
	{ \
		echo "[1/20] Running unit tests" && \
		poetry run pytest -q && \
		echo "[2/20] Preloading datasets" && \
		$(MAKE) prepare-data && \
		echo "[3/20] Calibrating spectral constant C (ER)" && \
		$(MAKE) calibrate-c && \
		echo "[4/20] Validating spectral/Cheeger bounds" && \
		$(MAKE) validate-bounds && \
		echo "[5/20] Calibrating probes" && \
		$(MAKE) probes-calib && \
		echo "[6/20] Training per-domain adapters" && \
		$(MAKE) train-all && \
		echo "[7/20] Signing adapters" && \
		$(MAKE) sign-all && \
		echo "[8/20] Swarm v2 simulation (security on)" && \
		$(MAKE) swarm-v2-smoke && \
		echo "[9/20] Summarising v2 reports" && \
		$(MAKE) swarm-v2-eval && \
		echo "[10/20] Training monolithic baseline (rank 4)" && \
		$(MAKE) monolithic-r4 && \
		echo "[11/20] Value-add rank sweep (small)" && \
		$(MAKE) value-add-rank-sweep && \
		echo "[12/20] Thesis-scale sweep (compact)" && \
		$(MAKE) thesis-sweep && \
		echo "[13/20] Generating figures" && \
		$(MAKE) figures && \
		echo "[14/20] Ablations (rank/scheme)" && \
		$(MAKE) ablation && \
		echo "[15/20] Alternating train-merge stability" && \
		$(MAKE) alt-train-merge && \
		echo "[16/20] Value-add JSONL build (low mem)" && \
		$(MAKE) value-add-build-lowmem && \
		echo "[17/20] Consensus-enabled v2 smoke" && \
		poetry run python -m swarm.sim_v2_entry --agents 4 --rounds 2 --graph er --graph_p $$(poetry run python -m plora.config graph.p) --seed 9 --security on --trojan_rate 0.3 --consensus on --quorum 2 --report_dir results && \
		echo "[18/20] gRPC offer/fetch demo" && \
		( poetry run python -m scripts.offer_server --root out & OFFER_PID=$$!; sleep 2; poetry run python -m scripts.fetch_client --domain legal --dest fetched --public-key keys/temp_pub.pem || true; kill $$OFFER_PID 2>/dev/null || true ) && \
		echo "[19/20] Dump effective security policy" && \
		RANKS_STR=$$(poetry run python -m plora.config allowed_ranks | tr -d '[] '); \
		$(MAKE) dump-policy RANKS="$$RANKS_STR" && \
		echo "[20/20] Net IT metrics" && \
		$(MAKE) net-it && \
		echo "Dry-run lite complete. See results/ and out."; \
	} 2>&1 | tee $$log_file

# ---------------------------------------------------------------------------
# Full experiment:
# - Tests, calibrations, and validations
# - Train & sign per-domain adapters
# - Swarm v2 (security on) + evaluation
# - Monolithic baseline + rank sweep value-add
# - Sweep across topologies/sizes + figures
# ---------------------------------------------------------------------------
.PHONY: full-experiment
full-experiment: config-use-full
	@mkdir -p logs
	@ts=$$(date +%Y%m%d-%H%M%S); \
	log_file=logs/full-experiment-$$ts.log; \
	{ \
		echo "[1/20] Running unit tests" && \
		poetry run pytest -q && \
		echo "[2/20] Preloading datasets" && \
		$(MAKE) prepare-data && \
		echo "[3/20] Calibrating spectral constant C (ER)" && \
		$(MAKE) calibrate-c && \
		echo "[4/20] Validating spectral/Cheeger bounds" && \
		$(MAKE) validate-bounds && \
		echo "[5/20] Calibrating probes" && \
		$(MAKE) probes-calib && \
		echo "[6/20] Training per-domain adapters" && \
		$(MAKE) train-all && \
		echo "[7/20] Signing adapters" && \
		$(MAKE) sign-all && \
		echo "[8/20] Swarm v2 simulation (security on)" && \
		$(MAKE) swarm-v2-smoke && \
		echo "[9/20] Summarising v2 reports" && \
		$(MAKE) swarm-v2-eval && \
		echo "[10/20] Training monolithic baseline (rank 4)" && \
		$(MAKE) monolithic-r4 && \
		echo "[11/20] Value-add rank sweep (small)" && \
		$(MAKE) value-add-rank-sweep && \
		echo "[12/20] Thesis-scale sweep (compact)" && \
		$(MAKE) thesis-sweep && \
		echo "[13/20] Generating figures" && \
		$(MAKE) figures && \
		echo "[14/20] Ablations (rank/scheme)" && \
		$(MAKE) ablation && \
		echo "[15/20] Alternating train-merge stability" && \
		$(MAKE) alt-train-merge && \
		echo "[16/20] Value-add JSONL build (low mem)" && \
		$(MAKE) value-add-build-lowmem && \
		echo "[17/20] Consensus-enabled v2 smoke" && \
		poetry run python -m swarm.sim_v2_entry --agents 6 --rounds 3 --graph er --graph_p $$(poetry run python -m plora.config graph.p) --seed 11 --security on --trojan_rate 0.3 --consensus on --quorum 2 --report_dir results && \
		echo "[18/20] gRPC offer/fetch demo (server in background)" && \
		( poetry run python -m scripts.offer_server --root out & OFFER_PID=$$!; sleep 2; poetry run python -m scripts.fetch_client --domain legal --dest fetched --public-key keys/temp_pub.pem || true; kill $$OFFER_PID 2>/dev/null || true ) && \
		echo "[19/20] Dump effective security policy" && \
		RANKS_STR=$$(poetry run python -m plora.config allowed_ranks | tr -d '[] '); \
		$(MAKE) dump-policy RANKS="$$RANKS_STR" && \
		echo "[20/20] Net IT metrics" && \
		$(MAKE) net-it && \
		echo "Full experiment complete. See results/ and out."; \
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
