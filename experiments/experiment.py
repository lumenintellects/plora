import hashlib
import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)


def setup_logging(log_dir: Path = Path("logs"), log_level: str = "INFO"):
    """
    Set up comprehensive logging to both file and console.

    Args:
        log_dir: Directory to store log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Create logs directory if it doesn't exist
    log_dir.mkdir(exist_ok=True)

    # Generate timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"plasmid_swarm_{timestamp}.log"

    # Remove any existing handlers to avoid duplicates
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure logging format
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler - detailed logging
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)

    # Console handler - slightly less verbose
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)

    # Configure root logger
    logging.root.setLevel(logging.DEBUG)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(console_handler)

    # Log the setup
    logging.info(f"Logging initialized - File: {log_file}")
    logging.info(f"Log level: {log_level}")

    return log_file


@dataclass
class Experiment:
    """
    Owns all mutable state for a single run of the pipeline.
    Instantiate exactly once per Python process.
    """

    base_model: str
    domains: List[str]
    seed: int = 42
    sample_cap: Optional[int] = None
    agents_dir: Path = Path("agents")
    mono_dir: Path = Path("monolithic")
    results_dir: Path = Path("results")
    logs_dir: Path = Path("logs")  # Added logs directory

    # mutable run‑time state
    real_data: Dict[str, List[Tuple[str, str]]] = field(init=False)
    total_bytes_transferred: int = 0
    transfer_events: int = 0
    _backbone_cache: Optional[
        Tuple[torch.nn.Module, AutoTokenizer, Dict[str, torch.Tensor]]
    ] = field(init=False, default=None)
    _uncompiled_backbone_cache: Optional[
        Tuple[torch.nn.Module, AutoTokenizer, Dict[str, torch.Tensor]]
    ] = field(init=False, default=None)
    tokenizer: Optional[AutoTokenizer] = field(init=False, default=None)
    log_file: Optional[Path] = field(init=False, default=None)

    def __post_init__(self):
        # Setup logging first
        self.log_file = setup_logging(self.logs_dir)

        random.seed(self.seed)
        from plasmid_swarm import RealDatasetLoader  # unchanged helper

        RealDatasetLoader.set_sample_limit(self.sample_cap)
        self.real_data = RealDatasetLoader.build_real_data()
        log.info(
            "Loaded datasets (%s domains, cap=%s)", len(self.real_data), self.sample_cap
        )

        # Log experiment configuration
        log.info("=" * 60)
        log.info("EXPERIMENT CONFIGURATION")
        log.info("=" * 60)
        log.info(f"Base model: {self.base_model}")
        log.info(f"Domains: {self.domains}")
        log.info(f"Seed: {self.seed}")
        log.info(f"Sample cap: {self.sample_cap}")
        log.info(f"Agents dir: {self.agents_dir}")
        log.info(f"Results dir: {self.results_dir}")
        log.info(f"Logs dir: {self.logs_dir}")

        # Log dataset sizes
        log.info("Dataset sizes:")
        for domain, data in self.real_data.items():
            log.info(f"  {domain}: {len(data)} samples")

        self.tokenizer = self.backbone()[1]

    def train_agents(self, epochs: int = 1, force: bool = False):
        from plasmid_swarm import train_agent  # refactored pure helper

        log.info("=" * 60)
        log.info("STARTING AGENT TRAINING")
        log.info("=" * 60)
        log.info(f"Training {len(self.domains)} agents for {epochs} epochs")
        log.info(f"Force retrain: {force}")

        for i, dom in enumerate(self.domains):
            agent_dir = self.agents_dir / f"agent_{i}"

            # Robust check for existing trained agent
            if self._is_agent_trained(agent_dir, dom) and not force:
                log.info("Agent_%d (%s) already trained – skipping", i, dom)
                continue

            log.info("Training Agent_%d on %s", i, dom)
            start_time = datetime.now()

            try:
                train_agent(agent_dir, dom, self, epochs)  # pass self
                elapsed = datetime.now() - start_time
                log.info(f"Agent_{i} ({dom}) training completed in {elapsed}")
            except Exception as e:
                log.error(f"Agent_{i} ({dom}) training failed: {e}", exc_info=True)
                raise

        log.info("All agent training completed")

    def train_monolithic(self, epochs: int = 1, force: bool = False):
        from plasmid_swarm import train_monolithic

        scores_file = self.mono_dir / "scores.json"
        if scores_file.exists() and not force:
            log.info("Monolithic model already trained – skipping")
            return
        log.info("Training monolithic model")
        train_monolithic(self.mono_dir, self, epochs)

    async def exchange(self, n_agents: int):
        from plasmid_swarm import load_agents, exchange_phase

        agents = load_agents(n_agents, self)  # pass self
        await exchange_phase(agents, self)  # pass self

    def evaluate(self, n_agents: int):
        from plasmid_swarm import (
            load_agents,
            evaluate_swarm,
            baseline_scores,
            generate_report,
        )

        agents = load_agents(n_agents, self)
        swarm_res = evaluate_swarm(agents, self)
        baseline = baseline_scores(self)
        mono_scores = (
            json.loads((self.mono_dir / "scores.json").read_text())
            if (self.mono_dir / "scores.json").exists()
            else {}
        )
        generate_report(agents, swarm_res, mono_scores, baseline, self)

    # shared resources helpers

    def backbone(self, compile_model: bool = True):
        """Get (model, tokenizer, pristine_state) cached once per Experiment."""
        if compile_model:
            if self._backbone_cache is None:
                self._backbone_cache = self._load_backbone(compile_model=True)
            return self._backbone_cache
        else:
            if self._uncompiled_backbone_cache is None:
                self._uncompiled_backbone_cache = self._load_backbone(
                    compile_model=False
                )
            return self._uncompiled_backbone_cache

    def _load_backbone(self, compile_model: bool = True):
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        # Use bfloat16 for better MPS performance on M3
        dtype = torch.bfloat16 if torch.backends.mps.is_available() else torch.float16

        model = (
            AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                device_map={"": device},
                attn_implementation="eager",
                trust_remote_code=True,
            )
            .eval()
            .requires_grad_(False)
        )
        tok = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        # Set tokenizer max length to match generation context
        tok.model_max_length = 256  # Was 224, updated to match evaluation settings

        # Store pristine state dict BEFORE torch.compile to avoid _orig_mod prefixes
        pristine = {k: v.to("cpu", copy=True) for k, v in model.state_dict().items()}

        # Apply torch.compile once for performance optimization
        if hasattr(torch, 'compile') and compile_model:
            try:
                model = torch.compile(model, mode="max-autotune")
                model._is_compiled = True  # Mark as compiled
                log.info("✓ torch.compile applied to backbone model")
            except Exception as e:
                log.warning("torch.compile failed on backbone: %s", e)
        elif compile_model:
            log.warning("torch.compile not available, proceeding without.")

        return model, tok, pristine

    def get_training_model(self, compiled: bool = True):
        """Get a fresh model instance for training, preserving torch.compile optimization."""
        import copy
        base, tok, pristine = self.backbone(compile_model=compiled)[:3]

        # Create a fresh model instance by copying the base model
        # For compiled models, deepcopy works. For uncompiled, we reload from pristine state.
        if compiled and hasattr(base, '_is_compiled'):
             training_model = copy.deepcopy(base)
        else:
             # For uncompiled models, or if compilation failed,
             # create a fresh instance and load the pristine state dict.
             uncompiled_base, tok, _ = self._load_backbone(compile_model=False)
             training_model = uncompiled_base

        training_model.train()

        return training_model, tok

    # utility for accounting – called by Agent.accept()
    def log_transfer(self, num_bytes: int):
        self.total_bytes_transferred += num_bytes
        self.transfer_events += 1

    def _is_agent_trained(self, agent_dir: Path, expected_domain: str) -> bool:
        """
        Comprehensive check if an agent is already properly trained.
        Validates manifest, adapter files, and domain consistency.
        Also checks for incomplete training that can be resumed.
        """
        try:
            # Check if directory exists
            if not agent_dir.exists():
                return False

            # Check for final manifest file (indicates completed training)
            manifest_path = agent_dir / "manifest.json"
            if manifest_path.exists():
                # Load and validate manifest
                from plasmid_swarm import EnhancedManifest
                manifest_data = json.loads(manifest_path.read_text())
                manifest = EnhancedManifest.model_validate(manifest_data)

                # Verify domain matches expected
                if manifest.domain != expected_domain:
                    log.warning(
                        "Domain mismatch in %s: expected %s, found %s",
                        agent_dir.name, expected_domain, manifest.domain
                    )
                    return False

                # Check for adapter model file
                adapter_files = list(agent_dir.glob("adapter_model.*"))
                if not adapter_files:
                    log.debug("No adapter model files found in %s", agent_dir.name)
                    return False

                # Verify adapter file integrity using SHA256
                adapter_path = adapter_files[0]  # Should be adapter_model.safetensors
                if not adapter_path.exists():
                    log.debug("Adapter file %s does not exist", adapter_path)
                    return False

                # Validate SHA256 hash matches manifest
                actual_sha = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
                if actual_sha != manifest.sha256:
                    log.warning(
                        "SHA256 mismatch for %s: expected %s, got %s",
                        adapter_path, manifest.sha256[:8], actual_sha[:8]
                    )
                    return False

                # Check for adapter config
                config_path = agent_dir / "adapter_config.json"
                if not config_path.exists():
                    log.debug("No adapter config found in %s", agent_dir.name)
                    return False

                # Validate base model matches
                if manifest.base_model != self.base_model:
                    log.warning(
                        "Base model mismatch in %s: expected %s, found %s",
                        agent_dir.name, self.base_model, manifest.base_model
                    )
                    return False

                # All checks passed - training is complete
                log.debug("Agent %s validation passed (complete)", agent_dir.name)
                return True

            # No manifest found - check if there's resumable training in progress
            tmp_dir = agent_dir / "tmp"
            if tmp_dir.exists():
                # Check for resumable checkpoints
                from plasmid_swarm import _find_latest_checkpoint
                latest_checkpoint = _find_latest_checkpoint(tmp_dir)
                if latest_checkpoint:
                    log.info(
                        "Agent %s has resumable checkpoint: %s",
                        agent_dir.name, latest_checkpoint.name
                    )
                    # Training can be resumed, so consider it "not trained"
                    return False

            # No complete training and no resumable checkpoints
            return False

        except Exception as e:
            log.warning("Failed to validate agent %s: %s", agent_dir.name, e)
            return False
