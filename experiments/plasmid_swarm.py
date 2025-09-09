import asyncio
import hashlib
import json
import logging
import random
import shutil
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache  # memorises tokenizer calls
from inspect import signature
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import click
import numpy as np
import torch
from datasets import load_dataset
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
)
from pydantic import BaseModel
from sacrebleu.metrics import CHRF  # token/character F1 proxy
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from experiment import Experiment

# ---------------------------------------------------------------------------
# Helper – map scheme -> target module suffixes
# ---------------------------------------------------------------------------


def select_target_modules(model, scheme: str) -> List[str]:  # noqa: D401
    """Return a sorted list of target module names for a given *scheme*.

    Parameters
    ----------
    model : torch.nn.Module
        HuggingFace model whose sub-module names are inspected.
    scheme : {"attention", "mlp", "all"}
        • "attention" – common attention projection matrices.
        • "mlp"       – feed-forward (gate/up/down) projections.
        • "all"       – union of the two.

    Notes
    -----
    The function is heuristic yet robust: it collects *suffix* matches across
    ``model.named_modules()``.  If nothing matches (e.g. tiny GPT-2 uses
    "c_attn"/"c_proj"), we fall back to a minimal sensible default per scheme.
    """

    scheme = scheme.lower()
    if scheme not in {"attention", "mlp", "all"}:
        raise ValueError("scheme must be one of attention|mlp|all, got %s" % scheme)

    attn_suffixes = ["q_proj", "k_proj", "v_proj", "o_proj", "c_attn", "c_proj"]
    mlp_suffixes = ["gate_proj", "up_proj", "down_proj", "mlp", "fc_in", "fc_out"]

    if scheme == "attention":
        wanted = attn_suffixes
    elif scheme == "mlp":
        wanted = mlp_suffixes
    else:  # all
        wanted = attn_suffixes + mlp_suffixes

    found: Set[str] = set()
    for name, _ in model.named_modules():
        for suff in wanted:
            if name.endswith(suff):
                found.add(suff)

    # Fallbacks if model uses fused projections
    if not found and scheme in ("attention", "all"):
        found.update({"c_attn", "c_proj"})
    if not found and scheme in ("mlp", "all"):
        found.update({"mlp"})

    return sorted(found)


def _compat_training_args(**kwargs):
    """
    Build `TrainingArguments`.
    """
    cleaned = {}
    for k, v in kwargs.items():
        # Handle the evaluation_strategy parameter name variations
        if (
            k == "evaluation_strategy"
            and k not in signature(TrainingArguments.__init__).parameters
        ):
            # Try alternative parameter names
            if "eval_strategy" in signature(TrainingArguments.__init__).parameters:
                cleaned["eval_strategy"] = v
            else:
                cleaned[k] = v
        elif k in signature(TrainingArguments.__init__).parameters:
            cleaned[k] = v
    return TrainingArguments(**cleaned)


# constants
BASE_MODEL: str = "google/gemma-3-1b-it"
SEED: int = 42


@contextmanager
def training_context(
    *,
    model: Optional[torch.nn.Module] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    base_model_name: str = BASE_MODEL,
    experiment: Optional[object] = None,
):
    """
        Context-manager that yields *(model, tokenizer)*, then
    guarantees a meticulous clean-up of GPU/CPU memory.

        Args
        ----
        model : torch.nn.Module | None
            - An already-loaded model to use. If provided, the manager
              will skip loading and just yield this model.
        tokenizer : AutoTokenizer | None
            - An already-loaded tokenizer.
        device : torch.device | str | None
            - "cuda", "mps", "cpu" or explicit torch.device()
            - None -> auto-detect best available device.
        dtype  : torch.dtype | None
            - Override precision. None -> heuristic in `_suggest_dtype()`.
        base_model_name : str
            - Which HF checkpoint to pull. Keeps the wrapper generic.
    """
    # If a model is passed in, use it directly.
    if model is not None and tokenizer is not None:
        try:
            yield model, tokenizer
        finally:
            # The caller is responsible for the model's lifecycle,
            # but we can still help with cleanup.
            del model, tokenizer
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc

            gc.collect()
        return

    if experiment is not None:
        # Use cached backbone from experiment for much faster loading
        model, tok = experiment.get_training_model()
        try:
            yield model, tok
        finally:
            del model, tok
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc

            gc.collect()
        return

    if device is None:  # auto‑select once
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    if dtype is None:
        dtype = _suggest_dtype()

    model = tok = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map={"": device},
            attn_implementation="eager",
            trust_remote_code=True,
        )
        tok = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        yield model, tok

    finally:
        del model, tok
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc

        gc.collect()


BASE_MODEL = "google/gemma-3-1b-it"
SCRIPT_DIR = Path(__file__).resolve().parent
AGENTS_DIR = SCRIPT_DIR / "agents"
MONO_DIR = SCRIPT_DIR / "monolithic"
RESULTS_DIR = SCRIPT_DIR / "results"

DOMAINS = [
    "arithmetic",  # GSM8K
    "legal",  # EUR‑Lex
    "medical",  # PubMedQA
    "coding",  # CodeSearchNet
    "science",  # SciQ
    "history",  # ChroniclingAmericaQA
    "geography",  # GeoQuestions1089
    "literature",  # NarrativeQA
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class RealDatasetLoader:
    """Domain‑specific dataset helpers that obey an optional global sample cap."""

    SAMPLE_LIMIT: Optional[int] = None  # None → load everything

    # helper methods
    @classmethod
    def set_sample_limit(cls, k: Optional[int]):
        """Configure the maximum number of rows to return per dataset."""
        cls.SAMPLE_LIMIT = k

    @classmethod
    def _hf_slice(cls, name: str, subset: Optional[str] = None):
        """Download *train* split with streaming for memory efficiency, deterministically shuffle, hard‑cap to SAMPLE_LIMIT."""
        try:
            # Use streaming for large datasets to avoid loading everything into RAM
            ds = load_dataset(name, subset, split="train", streaming=True)
            ds = ds.shuffle(
                buffer_size=10_000, seed=SEED
            )  # Use buffer for streaming shuffle

            # Convert to list with sample limit
            data = []
            for i, example in enumerate(ds):
                if cls.SAMPLE_LIMIT is not None and i >= cls.SAMPLE_LIMIT:
                    break
                data.append(example)

            # Convert back to dataset for consistency
            from datasets import Dataset as HFDataset

            return HFDataset.from_list(data) if data else None

        except Exception as e:
            logger.warning(
                "HF dataset %s failed: %s – using fallback mini‑set", name, e
            )
            return None

    @classmethod
    def load_arithmetic_data(cls):
        ds = cls._hf_slice("gsm8k", "main")
        if ds is None:
            return [("What is 15 + 27?", "42"), ("Calculate 8 × 7", "56")] * 200
        return [(ex["question"], ex["answer"].split("####")[-1].strip()) for ex in ds]

    @classmethod
    def load_legal_data(cls):
        ds = cls._hf_slice("lex_glue", "eurlex")
        if ds is None:
            return [
                (
                    "What is habeas corpus?",
                    "A legal principle requiring a person to be brought before a court",
                )
            ] * 200
        qas = []
        for ex in ds:
            snippet = ex["text"][:100].replace("\n", " ")
            qas.append(
                (f"Which legal domain best fits: {snippet}?", "Legal classification")
            )
        return qas

    @classmethod
    def load_medical_data(cls):
        ds = cls._hf_slice("pubmed_qa", "pqa_labeled")
        if ds is None:
            return [("What is the normal human body temperature?", "37°C")] * 200
        return [(ex["question"], ex["final_decision"]) for ex in ds]

    @classmethod
    def load_coding_data(cls):
        """
        Load Python snippets from CodeSearchNet (Python subset).
        Falls back to a tiny toy set if the HF load fails.
        """
        try:
            ds = load_dataset(
                "code-search-net/code_search_net",
                split="train",
                streaming=True,
            )
            data = []
            for ex in ds:
                code = (
                    ex.get("text") or ex.get("data", {}).get("text") or ex.get("code")
                )
                if not code:
                    continue
                snippet = code[:120].replace("\n", " ")
                data.append(
                    (
                        f"Explain this snippet: {snippet}",
                        "Python code explanation",
                    )
                )
                if cls.SAMPLE_LIMIT is not None and len(data) >= cls.SAMPLE_LIMIT:
                    break
            return data
        except Exception as e:
            logger.warning("CodeSearchNet load failed: %s", e)
            return [("How do you create a list in Python?", "my_list = []")] * 200

    @classmethod
    def load_science_data(cls):
        ds = cls._hf_slice("allenai/sciq")
        if ds is None:
            return [("What is the chemical formula for water?", "H2O")] * 200
        return [(ex["question"], ex["correct_answer"]) for ex in ds]

    @classmethod
    def load_history_data(cls):
        ds = cls._hf_slice("Bhawna/ChroniclingAmericaQA")
        if ds is None:
            return [
                (
                    "Who was the first President of the United States?",
                    "George Washington",
                )
            ] * 200
        return [(ex["question"], ex["answer"]) for ex in ds]

    @classmethod
    def load_geography_data(cls):
        ds = cls._hf_slice("AI-team-UoA/GeoQuestions1089")
        if ds is None:
            return [("What is the capital of France?", "Paris")] * 200
        return [(ex["Question"], str(ex["Answer"]).strip()) for ex in ds]

    @classmethod
    def load_literature_data(cls):
        ds = cls._hf_slice("deepmind/narrativeqa")
        if ds is None:
            return [("Who wrote 'Romeo and Juliet'?", "William Shakespeare")] * 200
        qa_pairs: List[Tuple[str, str]] = []
        for ex in ds:
            q = ex.get("question")
            if isinstance(q, dict):
                q = q.get("text", "")
            answers = ex.get("answers", [])
            first_ans = ""
            if answers:
                first_ans = (
                    answers[0]["text"] if isinstance(answers[0], dict) else answers[0]
                )
            qa_pairs.append((q, first_ans))
            if cls.SAMPLE_LIMIT is not None and len(qa_pairs) >= cls.SAMPLE_LIMIT:
                break
        return qa_pairs

    @classmethod
    def build_real_data(cls) -> Dict[str, List[Tuple[str, str]]]:
        return {
            "arithmetic": cls.load_arithmetic_data(),
            "legal": cls.load_legal_data(),
            "medical": cls.load_medical_data(),
            "coding": cls.load_coding_data(),
            "science": cls.load_science_data(),
            "history": cls.load_history_data(),
            "geography": cls.load_geography_data(),
            "literature": cls.load_literature_data(),
        }


# TODO: uncomment for smoke test:
# for key in REAL_DATA:
#     REAL_DATA[key] = REAL_DATA[key][:4]


# lora helpers
_ALLOWED_LORA_TARGETS: set[str] = {
    "q_proj",  # Query   projection in self‑attention
    "k_proj",  # Key     projection in self‑attention
    "v_proj",  # Value   projection in self‑attention
    "o_proj",  # Output  projection in self‑attention
    "gate_proj",  # Gated   feed‑forward column
    "up_proj",  # Up‑proj feed‑forward column
}


@lru_cache(maxsize=None)  # unlimited because architectures are few
def _cached_targets(arch_key: str) -> list[str]:
    """
    Internal helper.  Walk the module tree *once* for each
    architecture family and remember the result indefinitely.
    """
    # We need a throw-away model instance only to inspect the graph once.
    # NB: we instantiate on CPU to avoid any GPU side-effects.
    probe = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, device_map={"": "cpu"}, trust_remote_code=True
    )
    hits = {
        name.split(".")[-1]
        for name, mod in probe.named_modules()
        if isinstance(mod, torch.nn.Linear)
        and name.split(".")[-1] in _ALLOWED_LORA_TARGETS
    }
    if not hits:
        raise RuntimeError(
            f"No eligible LoRA targets found for architecture '{arch_key}'."
        )
    return sorted(hits)


def find_lora_target_modules(model) -> list[str]:
    """
    Cached version — O(1) after the first call per architecture family.
    """
    arch_list = getattr(model.config, "architectures", None)
    arch_key = arch_list[0] if arch_list else model.__class__.__name__
    return _cached_targets(arch_key)


def _suggest_dtype():
    """Heuristic for the best floating‑point precision."""
    if torch.backends.mps.is_available():
        return torch.bfloat16  # Better MPS performance on M3
    elif torch.cuda.is_available():
        return torch.float16
    else:
        return torch.float32


def load_base_model():
    # Remove unnecessary prepare_model_for_kbit_training since we never quantize
    if torch is None:
        raise RuntimeError("PyTorch unavailable – install torch to run training")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple‑Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=_suggest_dtype(),
        low_cpu_mem_usage=True,
        device_map={"": device},
        attn_implementation="eager",
        trust_remote_code=True,
    )
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Removed prepare_model_for_kbit_training() - unnecessary without quantization
    return model, tok


if torch is not None:

    class QADataset(Dataset):
        def __init__(self, qa_pairs, tokenizer):
            self.examples = []
            tokenizer.pad_token = tokenizer.eos_token
            for q, a in qa_pairs:
                prompt = f"Question: {q}\nAnswer:"
                full = f"{prompt} {a}"
                enc = tokenizer(
                    full,
                    truncation=True,
                    max_length=256,
                    padding="max_length",
                )
                # mask labels up to answer:
                ans_off = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
                # Manually pad labels
                labels = [-100] * ans_off + enc["input_ids"][ans_off:]
                labels += [-100] * (256 - len(labels))
                enc["labels"] = labels
                self.examples.append(enc)

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            # The collator will handle conversion to tensors and padding
            return self.examples[idx]


@lru_cache(maxsize=None)
def _norm(text: str) -> str:
    """Lower-case, strip punctuation + articles, squeeze whitespace."""
    import re, string

    text = re.sub(r"\b(a|an|the)\b", " ", text.lower())
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def exact_match(pred: str, gold: str) -> int:
    """1 if normalised strings match exactly else 0."""
    return int(_norm(pred) == _norm(gold))


# Instantiate once. CHRF is deterministic and stateless
_chrF = CHRF(word_order=2)  # n‑gram up to bigrams


def corpus_chrF(preds: List[str], refs: List[str]) -> float:
    """
    SacreBLEU CHRF returns 0-100; convert to 0-1 for parity with old F1.
    Works on lists of strings; sacrebleu expects list‑of‑lists for refs.
    """
    return _chrF.corpus_score(preds, [refs]).score / 100.0


def evaluate_model(
    model,
    tok,
    qa_pairs,
    max_samples: int = 40,
    n_votes: int = 3,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 32,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Evaluate with EM + CHRF-F1 using *n_votes* stochastic samples
    (temperature/top-p).  Majority vote picks the final answer.

    Returns:
        (exact_match, chrf_f1) averaged over the sample subset.
    """
    model.eval()
    subset = qa_pairs[:max_samples] if max_samples else qa_pairs
    prompts = [f"Question: {q}\nAnswer:" for q, _ in subset]

    # Seed generation for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Build one big batch and ask the model for n_votes variants per prompt
    with torch.no_grad():
        ins = tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(model.device)

        outs = model.generate(
            **ins,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=n_votes,
            pad_token_id=tok.eos_token_id,
        )

    # `outs` has len(prompts) * n_votes rows, grouped by original prompt
    decoded = tok.batch_decode(outs, skip_special_tokens=True)
    preds_grouped: List[List[str]] = [
        decoded[i * n_votes : (i + 1) * n_votes] for i in range(len(prompts))
    ]

    # Majority vote inside each group
    final_preds: List[str] = []
    for group in preds_grouped:
        answers = [g.split("Answer:")[-1].strip() for g in group]
        vote = Counter(_norm(a) for a in answers)
        maj_norm, _ = vote.most_common(1)[0]
        # pick the first generated variant that matches the majority‑normed form
        chosen = next(a for a in answers if _norm(a) == maj_norm)
        final_preds.append(chosen)

    refs = [gt for _, gt in subset]
    em = sum(exact_match(p, r) for p, r in zip(final_preds, refs)) / len(refs)
    f1 = corpus_chrF(final_preds, refs)

    logger.info(
        "Eval — EM: %.3f, CHRF-F1: %.3f  (T=%.1f, top-p=%.2f, votes=%d)",
        em,
        f1,
        temperature,
        top_p,
        n_votes,
    )
    return em, f1


class EnhancedManifest(BaseModel):
    sha256: str
    base_model: str
    domain: str
    f1: float
    exact_match: float  # ← NEW
    base_f1: float
    toxicity: float
    timestamp: int
    data_size: int
    training_time: float


@dataclass
class AdapterInfo:
    path: Path
    manifest: EnhancedManifest
    size_bytes: int


def _find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    """
    Find the latest checkpoint directory for resuming training.
    Returns None if no valid checkpoints found.
    """
    if not output_dir.exists():
        return None

    checkpoint_dirs = []
    for item in output_dir.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            try:
                # Extract checkpoint number for sorting
                checkpoint_num = int(item.name.split("-")[1])
                checkpoint_dirs.append((checkpoint_num, item))
            except (ValueError, IndexError):
                continue

    if not checkpoint_dirs:
        return None

    # Sort by checkpoint number and return the latest
    checkpoint_dirs.sort(key=lambda x: x[0])
    latest_checkpoint = checkpoint_dirs[-1][1]

    # Verify the checkpoint is complete (has required files)
    required_files = ["trainer_state.json", "optimizer.pt", "scheduler.pt"]
    if all((latest_checkpoint / f).exists() for f in required_files):
        return latest_checkpoint
    else:
        logger.warning(
            "Latest checkpoint %s is incomplete, skipping", latest_checkpoint.name
        )
        return None


def train_agent(agent_dir: Path, domain: str, exp, epochs=1):
    # Updated data split logic
    full = exp.real_data[domain][:]
    random.Random(exp.seed).shuffle(full)
    val_size = max(40, int(0.05 * len(full)))  # 5% hold-out, ≥40
    eval_data, train_data = full[:val_size], full[val_size:]

    logger.info(f"Training agent for {domain} domain")
    logger.info(f"Dataset split: {len(train_data)} train, {len(eval_data)} eval")
    logger.info(f"Training for {epochs} epochs")

    with training_context(experiment=exp) as (model, tok):
        logger.info("Evaluating baseline performance...")
        base_em, base_f1 = evaluate_model(model, tok, eval_data, seed=exp.seed)
        logger.info(f"Baseline scores - EM: {base_em:.3f}, F1: {base_f1:.3f}")

        # Use the new, stricter LoRA target finder
        target_modules = find_lora_target_modules(model)
        logger.info(f"LoRA target modules: {target_modules}")

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        logger.info(f"LoRA config: r={lora_config.r}, alpha={lora_config.lora_alpha}")

        model = get_peft_model(model, lora_config)

        model.train()

        # Log trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Trainable parameters: {trainable_params:,} / {total_params:,} "
            f"({100 * trainable_params / total_params:.2f}%)"
        )

        # Use dynamic padding for better memory efficiency
        train_ds = QADataset(train_data, tok)  # Reduced from 512 to 256
        val_ds = QADataset(eval_data, tok)

        # Use a language modeling collator to handle labels padding correctly
        data_collator = DataCollatorForLanguageModeling(tok, mlm=False)

        output_dir = agent_dir / "tmp"

        # Check for existing checkpoints to resume from
        resume_checkpoint = _find_latest_checkpoint(output_dir)
        if resume_checkpoint:
            logger.info("Resuming training from checkpoint: %s", resume_checkpoint.name)

        # Calculate steps per epoch for proper eval/save frequency
        effective_batch_size = (
            12 * 4
        )  # per_device_batch_size * gradient_accumulation_steps
        steps_per_epoch = len(train_data) // effective_batch_size
        total_steps = steps_per_epoch * epochs

        # Adjust eval/save steps to be within epoch boundaries to ensure evaluation happens
        eval_save_steps = (
            min(500, max(50, steps_per_epoch // 4))
            if steps_per_epoch >= 200
            else max(10, steps_per_epoch // 2)
        )

        logger.info(
            f"Training plan: {steps_per_epoch} steps/epoch × {epochs} epochs = {total_steps} total steps"
        )
        logger.info(f"Eval/save frequency: every {eval_save_steps} steps")

        tr_args = _compat_training_args(
            output_dir=str(output_dir),
            per_device_train_batch_size=12,  # Increased from 4 to 12
            gradient_accumulation_steps=4,  # Reduced from 8 to 4 (total effective batch size stays 48)
            learning_rate=2e-4,
            num_train_epochs=epochs,
            warmup_steps=100,
            max_grad_norm=1.0,
            save_strategy="steps",  # Changed from "epoch" to "steps" to match eval_strategy
            save_steps=eval_save_steps,  # Use calculated steps
            evaluation_strategy="steps",  # Changed from "epoch" to "steps"
            eval_steps=eval_save_steps,  # Match save_steps
            logging_strategy="steps",
            logging_steps=25,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",  # Specify which metric to use for best model
            greater_is_better=False,  # Lower loss is better
            report_to=None,
            remove_unused_columns=False,
            dataloader_pin_memory=False if torch.backends.mps.is_available() else True,
            save_total_limit=3,  # Keep only last 3 checkpoints to save disk space
        )

        trainer = Trainer(
            model=model,
            args=tr_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
        )

        # torch.compile on MPS is unstable for training, disable it for now.
        # if hasattr(torch, 'compile') and torch.backends.mps.is_available():
        #     try:
        #         logger.info("Applying torch.compile for speedup...")
        #         trainer.model = torch.compile(trainer.model, mode="max-autotune")
        #         logger.info("✓ torch.compile applied successfully")
        #     except Exception as e:
        #         logger.warning("torch.compile failed, continuing without: %s", e)

        start = time.time()
        logger.info("Starting training...")

        # Resume from checkpoint if available
        if resume_checkpoint:
            trainer.train(resume_from_checkpoint=str(resume_checkpoint))
        else:
            trainer.train()

        ttime = time.time() - start
        logger.info(
            f"Training completed in {ttime:.1f} seconds ({ttime / 3600:.2f} hours)"
        )

        logger.info("Evaluating trained model...")
        trained_em, trained_f1 = evaluate_model(model, tok, eval_data, seed=exp.seed)
        logger.info(f"Final scores - EM: {trained_em:.3f}, F1: {trained_f1:.3f}")
        logger.info(
            f"Improvement - EM: {trained_em - base_em:+.3f}, F1: {trained_f1 - base_f1:+.3f}"
        )

        logger.info("Saving model...")
        agent_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(agent_dir, safe_serialization=True)

    # Clean up immediately after training
    import gc

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Memory cleanup completed")

    # Clean up temporary training directory
    if output_dir.exists():
        try:
            shutil.rmtree(output_dir)
            logger.info("Cleaned up temporary training directory: %s", output_dir)
        except OSError as e:
            logger.warning(
                "Failed to clean up temporary directory %s: %s", output_dir, e
            )

    ad_path = next(p for p in agent_dir.iterdir() if p.name.startswith("adapter_model"))
    sha = hashlib.sha256(ad_path.read_bytes()).hexdigest()

    manifest = EnhancedManifest(
        sha256=sha,
        base_model=exp.base_model,
        domain=domain,
        f1=trained_f1,
        exact_match=trained_em,
        base_f1=base_f1,
        toxicity=0.0,
        timestamp=int(time.time()),
        data_size=len(train_data),
        training_time=ttime,
    )

    (agent_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    logger.info(f"{domain} agent training completed successfully")

    return AdapterInfo(ad_path, manifest, ad_path.stat().st_size)


def train_monolithic(output_dir: Path, exp, epochs=1):
    # The training context is now used just for memory management.
    # We get a fresh, uncompiled model to avoid issues with torch.compile and resuming
    model, tok = exp.get_training_model(compiled=False)

    with training_context(model=model, tokenizer=tok) as (model, tok):
        data_collator = DataCollatorForLanguageModeling(tok, mlm=False)

        # Use the new, stricter LoRA target finder
        target_modules = find_lora_target_modules(model)

        logger.info("Detected LoRA target modules: %s", target_modules)

        lora_cfg = LoraConfig(
            r=8,  # Reduced from 16 to 8 for consistency
            lora_alpha=16,  # Adjusted proportionally
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        # We apply PEFT to the model we got from the context
        model = get_peft_model(model, lora_cfg)

        model.train()

        # Combine data from all domains
        all_pairs = []
        for dom, pairs in exp.real_data.items():
            all_pairs.extend(pairs)  # Use all pairs, no truncation
        random.shuffle(all_pairs)

        # Split data for training and evaluation
        val_size = max(40, int(0.05 * len(all_pairs)))
        eval_pairs, train_pairs = all_pairs[:val_size], all_pairs[val_size:]

        train_ds = QADataset(train_pairs, tok)  # Use dynamic padding
        eval_ds = QADataset(eval_pairs, tok)

        # Calculate steps per epoch for proper eval/save frequency
        effective_batch_size = 12 * 4
        steps_per_epoch = len(train_pairs) // effective_batch_size

        # Adjust eval/save steps to be within epoch boundaries
        eval_save_steps = min(500, max(50, steps_per_epoch // 4))

        tmp_output_dir = output_dir / "tmp"
        resume_checkpoint = _find_latest_checkpoint(tmp_output_dir)
        if resume_checkpoint:
            logger.info(
                "Resuming monolithic training from checkpoint: %s",
                resume_checkpoint.name,
            )

        tr_args = _compat_training_args(
            output_dir=str(tmp_output_dir),
            per_device_train_batch_size=12,  # Increased batch size
            gradient_accumulation_steps=4,  # Reduced grad accumulation
            learning_rate=2e-4,
            num_train_epochs=epochs,
            warmup_steps=100,
            max_grad_norm=1.0,
            save_strategy="steps",
            save_steps=eval_save_steps,
            evaluation_strategy="steps",  # Changed to steps
            eval_steps=eval_save_steps,  # Match save_steps
            logging_strategy="steps",
            logging_steps=25,
            load_best_model_at_end=True,
            report_to=None,
            remove_unused_columns=False,
            dataloader_pin_memory=False if torch.backends.mps.is_available() else True,
        )

        trainer = Trainer(
            model=model,
            args=tr_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=data_collator,
        )

        if resume_checkpoint:
            trainer.train(resume_from_checkpoint=str(resume_checkpoint))
        else:
            trainer.train()

        # Evaluate performance on all domains
        scores = {
            dom: evaluate_model(model, tok, exp.real_data[dom][:40], seed=exp.seed)
            for dom in exp.domains
        }

        output_dir.mkdir(exist_ok=True, parents=True)
        model.save_pretrained(output_dir, safe_serialization=True)

    # Clean up temporary training directory
    if tmp_output_dir.exists():
        try:
            shutil.rmtree(tmp_output_dir)
            logger.info(
                "Cleaned up monolithic temporary training directory: %s", tmp_output_dir
            )
        except OSError as e:
            logger.warning(
                "Failed to clean up monolithic temporary directory %s: %s",
                tmp_output_dir,
                e,
            )

    # Clean up immediately after training
    import gc

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(
        "Monolithic scores: %s",
        {k: (round(em, 3), round(f1, 3)) for k, (em, f1) in scores.items()},
    )
    (output_dir / "scores.json").write_text(json.dumps(scores, indent=2))
    return scores


@dataclass
class Agent:
    agent_id: int
    domain: str
    adapter: AdapterInfo
    exp: Any  # Reference to Experiment
    knowledge: Set[str] = field(default_factory=set)
    f1_scores: Dict[str, float] = field(default_factory=dict)
    accepted: int = 0
    offered: int = 0
    rejected_safety: int = 0
    rejected_hash: int = 0
    capacity: int = 8
    cache: List[str] = field(default_factory=list)
    # we store adapters copied from peers
    received_adapters: Dict[str, Path] = field(default_factory=dict)

    _STATE_FILE = "state.json"  # written in agent_N/ alongside manifest
    _RECV_DIR = "received"  # sub‑dir containing copied LoRA files
    _COPY_LOCK: asyncio.Lock = asyncio.Lock()  # guards all on‑disk writes

    @classmethod
    def from_dir(
        cls, agent_id: int, domain: str, adapter: AdapterInfo, agent_dir: Path, exp: Any
    ) -> "Agent":
        """
        Factory that loads any previously persisted state; otherwise creates a
        fresh Agent exactly as before.
        """
        state_path = agent_dir / cls._STATE_FILE
        if state_path.exists():
            data = json.loads(state_path.read_text())
            agent = cls(
                agent_id=agent_id,
                domain=domain,
                adapter=adapter,
                exp=exp,
                knowledge=set(data.get("knowledge", [domain])),
                f1_scores=data.get("f1_scores", {domain: adapter.manifest.f1}),
                accepted=data.get("accepted", 0),
                offered=data.get("offered", 0),
                rejected_safety=data.get("rejected_safety", 0),
                rejected_hash=data.get("rejected_hash", 0),
                cache=data.get("cache", [domain]),
                received_adapters={
                    k: agent_dir / v
                    for k, v in data.get("received_adapters", {}).items()
                },
            )
            return agent
        else:
            agent = cls(agent_id=agent_id, domain=domain, adapter=adapter, exp=exp)
            agent._save_state(agent_dir)
            return agent

    def _save_state(self, agent_dir: Path):
        """Serialise the in-memory state to a human-readable JSON file."""
        payload = {
            "knowledge": sorted(self.knowledge),
            "f1_scores": self.f1_scores,
            "accepted": self.accepted,
            "offered": self.offered,
            "rejected_safety": self.rejected_safety,
            "rejected_hash": self.rejected_hash,
            "cache": self.cache,
            # Persist paths relative to the agent directory for portability
            "received_adapters": {
                dom: str(p.relative_to(agent_dir))
                for dom, p in self.received_adapters.items()
            },
        }
        (agent_dir / self._STATE_FILE).write_text(json.dumps(payload, indent=2))

    def shareable_adapters(self) -> Dict[str, AdapterInfo]:
        """
        Return *all* adapters this agent can currently donate
        Keys are the domains they serve; values are full AdapterInfo objects.

        The pool always contains:
          - the agent’s original specialist LoRA
          - every LoRA previously accepted from peers
        """
        pool: Dict[str, AdapterInfo] = {self.domain: self.adapter}

        # Build AdapterInfo on the fly for received LoRAs
        for dom, path in self.received_adapters.items():
            try:
                man_path = path.parent / "manifest.json"
                manifest = EnhancedManifest.model_validate_json(man_path.read_text())
                pool[dom] = AdapterInfo(path, manifest, path.stat().st_size)
            except Exception as exc:
                logger.warning("Skipping %s – manifest load failed: %s", dom, exc)
        return pool

    async def run(self, agents: List["Agent"]):
        """
        One peer‑to‑peer sharing round.

        - For every other agent, pick a random adapter from our shareable pool
          (specialist *or* any we’ve already learned) and offer it.
        - Update counters and persist state so progress survives crashes.
        """
        for peer in agents:
            if peer is self:
                continue

            don_dom, don_adapter = random.choice(
                list(self.shareable_adapters().items())
            )

            self.offered += 1
            await peer.accept(don_adapter, don_dom)

        self._save_state(self.adapter.path.parent)

    # override accept() so that we copy the adapter locally and save state
    async def accept(self, adapter: AdapterInfo, dom: str) -> bool:
        if not adapter.path.exists():
            return False
        if (
            hashlib.sha256(adapter.path.read_bytes()).hexdigest()
            != adapter.manifest.sha256
        ):
            self.rejected_hash += 1
            return False

        # physically copy the LoRA checkpoint and its manifest
        agent_dir = self.adapter.path.parent
        recv_domdir = agent_dir / self._RECV_DIR / dom

        # critical section: one coroutine at a time
        async with self._COPY_LOCK:
            # mkdir is also a syscall, keep it inside the lock
            recv_domdir.mkdir(parents=True, exist_ok=True)

            src_dir = adapter.path.parent

            # Copy only essential files, not the whole directory tree
            files_to_copy = [
                "adapter_model.safetensors",
                "adapter_config.json",
                "manifest.json",
            ]
            for filename in files_to_copy:
                src_file = src_dir / filename
                dst_file = recv_domdir / filename
                if src_file.exists():
                    await asyncio.to_thread(shutil.copy, src_file, dst_file)
                else:
                    # manifest is not always in the same dir for received adapters
                    if filename == "manifest.json":
                        try:
                            # The manifest for a received adapter is in its parent's parent
                            manifest_path = src_dir.parent / "manifest.json"
                            if manifest_path.exists():
                                await asyncio.to_thread(
                                    shutil.copy, manifest_path, dst_file
                                )
                            else:  # check one level higher
                                manifest_path = src_dir.parent.parent / "manifest.json"
                                if manifest_path.exists():
                                    await asyncio.to_thread(
                                        shutil.copy, manifest_path, dst_file
                                    )
                        except Exception:
                            logger.warning(
                                "Could not find or copy manifest for %s", dom
                            )
                    else:
                        logger.warning(
                            "File %s not found in %s, skipping copy.", filename, src_dir
                        )

        # bookkeeping (pure Python, no I/O)
        self.accepted += 1
        self.knowledge.add(dom)
        self.cache.append(dom)
        self.f1_scores[dom] = adapter.manifest.f1
        self.received_adapters[dom] = recv_domdir / adapter.path.name
        self.exp.log_transfer(adapter.size_bytes)

        # FIFO eviction of RAM‑only cache
        if len(self.cache) > self.capacity:
            evict = self.cache.pop(0)
            self.knowledge.discard(evict)
            self.f1_scores.pop(evict, None)
            self.received_adapters.pop(evict, None)

        # persist immediately so a crash does not lose state
        self._save_state(agent_dir)
        return True

    def summary(self):
        return {
            "id": self.agent_id,
            "orig_domain": self.domain,
            "domains": sorted(self.knowledge),
            "accepted": self.accepted,
            "offered": self.offered,
            "rej_hash": self.rejected_hash,
            "rej_safety": self.rejected_safety,
        }


def load_agents(n_agents: int, exp):
    agents = []
    for i in range(n_agents):
        dom = exp.domains[i]
        a_dir = exp.agents_dir / f"agent_{i}"
        # original adapter (specialist for dom)
        raw_json = (a_dir / "manifest.json").read_text()
        manifest = EnhancedManifest.model_validate_json(raw_json)
        ad_path = next(p for p in a_dir.iterdir() if p.name.startswith("adapter_model"))

        adapter = AdapterInfo(ad_path, manifest, ad_path.stat().st_size)
        # load persisted state if there is one
        agent = Agent.from_dir(i, dom, adapter, a_dir, exp)
        agents.append(agent)
    return agents


async def exchange_phase(agents: list, exp):
    await asyncio.gather(*(a.run(agents) for a in agents))


def load_frozen_backbone(device=None):
    """Load Gemma once, put in eval mode, gradient-free."""
    if device is None:  # pick sensible default
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

    model = (
        AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=_suggest_dtype(),
            low_cpu_mem_usage=True,
            device_map={"": device},
            attn_implementation="eager",
            trust_remote_code=True,
        )
        .eval()
        .requires_grad_(False)
    )

    # Keep an untouched copy of the state dict on CPU for fast restores
    pristine_state = {k: v.to("cpu", copy=True) for k, v in model.state_dict().items()}
    return model, pristine_state


def eval_with_adapter(
    model: torch.nn.Module,
    pristine_state: Dict[str, torch.Tensor],
    adapter_dir: Path,  # directory that contains adapter_model.bin
    exp: Experiment,
    domain: str,
) -> Tuple[float, float]:
    """
    Thread‑safe adapter evaluation.  Loads adapter into a fresh copy of the
    pristine base model, runs `evaluate_model`, then unloads the adapter.
    """
    # Restore the model to its original state before applying the LoRA adapter
    fresh_model = model.from_pretrained(
        exp.base_model,
        torch_dtype=model.dtype,
        low_cpu_mem_usage=True,
        device_map={"": model.device},
        attn_implementation="eager",
        trust_remote_code=True,
    )
    fresh_model.load_state_dict(pristine_state, assign=True)

    # Load the LoRA adapter from the specified directory
    peft_model = PeftModel.from_pretrained(
        fresh_model,
        str(adapter_dir),  # <-- cast Path → str
        is_trainable=False,
    )
    peft_model.to(model.device)

    # Evaluate the performance of the adapted model
    em, f1 = evaluate_model(
        peft_model, exp.tokenizer, exp.real_data[domain], seed=exp.seed
    )

    del fresh_model, peft_model
    return em, f1


def evaluate_swarm(agents: List[Agent], exp: Experiment):
    """
    For each agent, evaluate its performance on all domains it has knowledge of.
    Returns a nested dictionary: {agent_id: {domain: (EM, F1)}}
    """
    model, _, pristine_state = exp.backbone()
    results = {}

    for agent in agents:
        logger.info(
            "Evaluating Agent %d (specialist in %s)", agent.agent_id, agent.domain
        )
        agent_scores = {}

        # Build a map of all adapters the agent has, both original and received
        dom2path = {agent.domain: str(agent.adapter.path.parent)}
        dom2path.update({d: str(p.parent) for d, p in agent.received_adapters.items()})

        for domain, adapter_path_str in dom2path.items():
            adapter_path = Path(adapter_path_str)
            if adapter_path.exists():
                em, f1 = eval_with_adapter(
                    model, pristine_state, adapter_path, exp, domain
                )
                agent_scores[domain] = (em, f1)
            else:
                logger.warning(
                    "Adapter path for domain '%s' not found: %s", domain, adapter_path
                )
        results[agent.agent_id] = agent_scores
    return results


# Update baseline_scores to use the cached backbone
def baseline_scores(exp: Experiment) -> Dict[str, Tuple[float, float]]:
    backbone, tok, _ = exp.backbone()
    return {
        d: evaluate_model(backbone, tok, exp.real_data[d][:40]) for d in exp.domains
    }


def generate_report(
    agents: List[Agent],
    swarm_res: Dict[str, Any],
    mono: Dict[str, float],
    base: Dict[str, float],
    exp,
):
    pre = {a.domain: a.adapter.manifest.f1 for a in agents}
    post_f1 = {}
    for d in exp.domains:
        scores = [
            res[d][1] for res in swarm_res.values() if d in res
        ]  # F1 is the 2nd element
        if scores:
            post_f1[d] = np.mean(scores)

    post_em = {}
    for d in exp.domains:
        scores = [
            res[d][0] for res in swarm_res.values() if d in res
        ]  # EM is the 1st element
        if scores:
            post_em[d] = np.mean(scores)

    improvement_f1 = {
        d: round(post_f1[d] - base[d][1], 3)
        for d in exp.domains
        if d in post_f1 and d in base
    }
    report = {
        "meta": {
            "agents": len(agents),
            "domains": exp.domains,
            "total_transfer_events": exp.transfer_events,
            "total_mb_transferred": round(exp.total_bytes_transferred / 1_048_576, 2),
        },
        "baseline_f1": {k: v[1] for k, v in base.items()},
        "pre_sharing_specialist_f1": pre,
        "monolithic_f1": {k: v[1] for k, v in mono.items()},
        "post_sharing_avg_f1": post_f1,
        "post_sharing_avg_em": post_em,
        "baseline_to_post_improvement_f1": improvement_f1,
        "agent_summaries": [a.summary() for a in agents],
    }
    exp.results_dir.mkdir(exist_ok=True, parents=True)
    (exp.results_dir / "swarm_report.json").write_text(json.dumps(report, indent=2))
    logger.info("Report written to %s", exp.results_dir / "swarm_report.json")


@click.command()
@click.argument(
    "phase", type=click.Choice(["train", "monolithic", "exchange", "evaluate", "full"])
)
@click.option(
    "--n_agents", default=8, show_default=True, help="Number of agents (=domains)"
)
@click.option("--epochs", default=1, show_default=True, help="Training epochs per LoRA")
@click.option(
    "--samples",
    type=int,
    default=None,
    help="Hard cap on examples per domain (None = use full dataset)",
)
@click.option(
    "--force", is_flag=True, help="Force retraining agents even if manifest exists"
)
def main(phase: str, n_agents: int, epochs: int, force: bool, samples: int):
    exp = Experiment(
        base_model=BASE_MODEL,
        domains=DOMAINS[:n_agents],
        sample_cap=samples,
        agents_dir=AGENTS_DIR,
        mono_dir=MONO_DIR,
        results_dir=RESULTS_DIR,
        seed=SEED,
    )

    if phase in {"train", "full"}:
        exp.train_agents(epochs, force)
    if phase in {"monolithic", "full"}:
        exp.train_monolithic(epochs, force)
    if phase in {"exchange", "evaluate", "full"}:
        asyncio.run(exp.exchange(n_agents))
    if phase in {"evaluate", "full"}:
        exp.evaluate(n_agents)


if __name__ == "__main__":
    main()
