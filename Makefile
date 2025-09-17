.PHONY: setup poetry-env test train-legal sign-legal offer fetch value-add-smoke value-add-full swarm-sim \
	swarm-v2-smoke swarm-v2-eval monolithic-r4 value-add-rank-sweep dry-run-lite dump-policy full-experiment
.PHONY: alt-train-merge
alt-train-merge:
	poetry run python -m scripts.alternating_train_merge --domains arithmetic,legal,medical --cycles 2 --samples 32 --rank 4 --out results/alt_train_merge

.PHONY: ablation
ablation:
	poetry run python -m scripts.ablation_runner --domains arithmetic,legal,medical --ranks 2,4,8 --schemes attention,all --samples 64 --epochs 1 --out results/ablation.jsonl

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

DOMAINS := arithmetic legal medical

train-all:
	@for d in $(DOMAINS); do \
		poetry run python -m scripts.train_task \
		  --domain $$d --epochs 1 --output out/$$d ; \
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
	poetry run python -m scripts.run_lora_value_add \
	  --domains "$$(poetry run python -c 'import json, sys; from plora.config import get; print(",".join(get("domains", [])))')"

# Full value-add experiment (longer, deeper grid)
value-add-full:
	poetry run python -m scripts.run_lora_value_add \
	  --domains "$$(poetry run python -c 'import json, sys; from plora.config import get; print(",".join(get("domains", [])))')" \
	  --seeds $$(poetry run python -m plora.config value_add.seeds) \
	  --resume

# Build value_add.jsonl from artifacts (full evaluation)
value-add-build-full:
	PLORA_FORCE_CPU=$(CPU) \
	poetry run python -m scripts.build_value_add_jsonl \
	  --artifacts-dir results/value_add \
	  --output results/value_add/value_add.jsonl \
	  --domains "$$(poetry run python -c 'import json, sys; from plora.config import get; print(",".join(get("domains", [])))')" \
	  --dev-size $$(poetry run python -m plora.config value_add.dev_size) \
	  --base-model $$(poetry run python -m plora.config base_model) \
	  --overwrite

# Build with a filter (regex) and smaller dev set for smoke or chunked runs
value-add-build-filter:
	PLORA_FORCE_CPU=$(CPU) \
	poetry run python -m scripts.build_value_add_jsonl \
	  --artifacts-dir results/value_add \
	  --output results/value_add/value_add.jsonl \
	  --domains "$$(poetry run python -c 'import json, sys; from plora.config import get; print(",".join(get("domains", [])))')" \
	  --dev-size $(DEV) \
	  --base-model $$(poetry run python -m plora.config base_model) \
	  --filter $(FILTER)

# Build with very low resource usage (skip placebos & cross, smaller dev and length)
value-add-build-lowmem:
	poetry run python -m scripts.build_value_add_jsonl \
	  --artifacts-dir results/value_add \
	  --output results/value_add/value_add.jsonl \
	  --domains "$$(poetry run python -c 'import json, sys; from plora.config import get; print(",".join(get("domains", [])))')" \
	  --dev-size $$(poetry run python -m plora.config value_add.dev_size) \
	  --max-length 256 \
	  --base-model $$(poetry run python -m plora.config base_model) \
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
	poetry run python -m swarm.sim_v2_entry --agents 6 --rounds 5 --graph_p $$(poetry run python -m plora.config graph.p) --security on --trojan_rate 0.3

# Summarise v2 (and v1 graph) reports into a compact JSON
swarm-v2-eval:
	poetry run python -m scripts.evaluate_v2 --reports results --out results/summary_v2.json

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
	poetry run python -m scripts.thesis_sweep --topos er,ws,ba --ns 20,40,80,160 --seeds 41,42 --p 0.25 --rounds 10 --trojan_rates 0.0,0.3 --out results/thesis_sweep.jsonl

# Monolithic baseline – tiny training loop over 3 domains at rank 4
monolithic-r4:
	poetry run python -m scripts.monolithic_train --domains "$$(poetry run python -c 'import json, sys; from plora.config import get; print(",".join(get("domains", [])))')" --epochs 1 --samples $$(poetry run python -m plora.config samples) --rank 4 --output out/monolithic_r4

# Rank sweep runner – writes rank-scoped outputs under results/value_add
value-add-rank-sweep:
	poetry run python -m scripts.run_lora_value_add \
	  --domains "$$(poetry run python -c 'import json, sys; from plora.config import get; print(",".join(get("domains", [])))')"

# ---------------------------------------------------------------------------
# Minimal dry run (very fast):
# - Run unit tests
# - Swarm v2 smoke (security on)
# - Summarise v2 reports
# ---------------------------------------------------------------------------
.PHONY: config-use-full config-use-dry
config-use-full:
	poetry run python -m plora.config use config/plora.full.yml

config-use-dry:
	poetry run python -m plora.config use config/plora.dry.yml

.PHONY: dry-run-lite
dry-run-lite: config-use-dry
	@echo "[1/16] Running unit tests" && \
	poetry run pytest -q && \
	echo "[2/16] Calibrating probes" && \
	$(MAKE) probes-calib && \
	echo "[3/16] Calibrating C (tiny ER)" && \
	poetry run python -m scripts.calibrate_c --topology er --ns 10,12 --p $$(poetry run python -m plora.config graph.p) --rounds 5 --seed 7 --out results/c_calib_er_lite.json && \
	echo "[4/16] Validating bounds (tiny)" && \
	poetry run python -m scripts.validate_bounds --ns 10,12 --p $$(poetry run python -m plora.config graph.p) --seed 7 --out results/bounds_validation_lite.json && \
	echo "[5/16] Training per-domain adapters (tiny)" && \
	$(MAKE) train-all && \
	echo "[6/16] Signing adapters" && \
	$(MAKE) sign-all && \
	echo "[7/16] Swarm v2 simulation (security on, fast)" && \
	$(MAKE) swarm-v2-smoke && \
	echo "[8/16] Summarising v2 reports" && \
	$(MAKE) swarm-v2-eval && \
	echo "[9/16] Monolithic baseline (fast)" && \
	$(MAKE) monolithic-r4 && \
	echo "[10/16] Value-add rank sweep (small, tiny base)" && \
	$(MAKE) value-add-rank-sweep && \
	echo "[11/16] Alternating train-merge (tiny)" && \
	poetry run python -m scripts.alternating_train_merge --domains arithmetic,legal --cycles 1 --out results/alt_train_merge_lite && \
	echo "[12/16] Value-add JSONL build (lowmem)" && \
	$(MAKE) value-add-build-lowmem && \
	echo "[13/16] Consensus-enabled v2 smoke (fast)" && \
	poetry run python -m swarm.sim_v2_entry --agents 4 --rounds 2 --graph er --graph_p $$(poetry run python -m plora.config graph.p) --seed 9 --security on --trojan_rate 0.3 --consensus on --quorum 2 --report_dir results && \
	echo "[14/16] gRPC offer/fetch demo (fast)" && \
	( poetry run python -m scripts.offer_server --root out & OFFER_PID=$$!; sleep 2; poetry run python -m scripts.fetch_client --domain legal --dest fetched --public-key keys/temp_pub.pem || true; kill $$OFFER_PID 2>/dev/null || true ) && \
	echo "[15/16] Dump effective security policy" && \
	$(MAKE) dump-policy && \
	echo "Writing tiny history for net-it metrics" && \
	python -c "import json, pathlib; hist=[{0:['a'],1:['b'],2:['c']},{0:['a','b'],1:['b'],2:['c']},{0:['a','b','c'],1:['a','b','c'],2:['a','b','c']}]; p=pathlib.Path('results/history.json'); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(hist)); print('Wrote',p)" && \
	$(MAKE) net-it && \
	echo "Dry-run lite complete. See results/ and out/."

# ---------------------------------------------------------------------------
# Full experiment:
# - Tests, calibrations, and validations
# - Train & sign per-domain adapters
# - Swarm v2 (security on) + evaluation
# - Monolithic baseline + rank sweep value-add
# - Sweep across topologies/sizes + figures
# Tip: export PLORA_SAMPLES=64 and set BASE_MODEL to a tiny model for local runs
# ---------------------------------------------------------------------------
.PHONY: full-experiment
full-experiment: config-use-full
	@echo "[1/19] Running unit tests" && \
	poetry run pytest -q && \
	echo "[2/19] Calibrating spectral constant C (ER)" && \
	$(MAKE) calibrate-c && \
	echo "[3/19] Validating spectral/Cheeger bounds" && \
	$(MAKE) validate-bounds && \
	echo "[4/19] Calibrating probes" && \
	$(MAKE) probes-calib && \
	echo "[5/19] Training per-domain adapters" && \
	$(MAKE) train-all && \
	echo "[6/19] Signing adapters" && \
	$(MAKE) sign-all && \
	echo "[7/19] Swarm v2 simulation (security on)" && \
	$(MAKE) swarm-v2-smoke && \
	echo "[8/19] Summarising v2 reports" && \
	$(MAKE) swarm-v2-eval && \
	echo "[9/19] Training monolithic baseline (rank 4)" && \
	$(MAKE) monolithic-r4 && \
	echo "[10/19] Value-add rank sweep (small)" && \
	$(MAKE) value-add-rank-sweep && \
	echo "[11/19] Thesis-scale sweep (compact)" && \
	$(MAKE) thesis-sweep && \
	echo "[12/19] Generating figures" && \
	$(MAKE) figures && \
	echo "[13/19] Ablations (rank/scheme)" && \
	$(MAKE) ablation && \
	echo "[14/19] Alternating train-merge stability" && \
	$(MAKE) alt-train-merge && \
	echo "[15/19] Value-add JSONL build (low mem)" && \
	$(MAKE) value-add-build-lowmem && \
	echo "[16/19] Consensus-enabled v2 smoke" && \
	poetry run python -m swarm.sim_v2_entry --agents 6 --rounds 3 --graph er --graph_p 0.25 --seed 11 --security on --trojan_rate 0.3 --consensus on --quorum 2 --report_dir results && \
	echo "[17/19] gRPC offer/fetch demo (server in background)" && \
	( poetry run python -m scripts.offer_server --root out & OFFER_PID=$$!; sleep 2; poetry run python -m scripts.fetch_client --domain legal --dest fetched --public-key keys/temp_pub.pem || true; kill $$OFFER_PID 2>/dev/null || true ) && \
	echo "[18/19] Dump effective security policy" && \
	$(MAKE) dump-policy && \
	echo "Writing tiny history for net-it metrics" && \
	python -c "import json, pathlib; hist=[{0:['a'],1:['b'],2:['c']},{0:['a','b'],1:['b'],2:['c']},{0:['a','b','c'],1:['a','b','c'],2:['a','b','c']}]; p=pathlib.Path('results/history.json'); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(hist)); print('Wrote',p)" && \
	$(MAKE) net-it && \
	echo "Full experiment complete. See results/ and out/."

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
TARGETS ?= assets/allowed_targets.txt
RANKS ?= 4,8,16
SIG ?= off
dump-policy:
	poetry run python -m scripts.dump_policy \
		$$(test -n "$(POLICY)" && echo --policy_file $(POLICY)) \
		--allowed_targets_file $(TARGETS) \
		--allowed_ranks $(RANKS) \
		--signatures $(SIG)
