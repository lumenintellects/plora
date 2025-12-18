from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Tuple

import pytest

from plora import dataset_loader


@dataclass
class _DummyStream:
    records: List[Tuple[str, str]]

    def shuffle(
        self, *, seed: int, buffer_size: int | None = None
    ) -> "_DummyStream":
        return self

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        yield from self.records

    def select(self, indices) -> "_DummyStream":
        if isinstance(indices, range):
            indices = list(indices)
        selected = [self.records[i] for i in indices]
        return _DummyStream(selected)


def test_all_domains_available(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[Tuple[str, str]] = []

    domain_payloads = {
        "deepmind/aqua_rat": [
            {"question": "Q1", "rationale": "A1"},
            {"question": "Q2", "rationale": "A2"},
        ],
        "lex_glue": [
            # case_hold structure: context with <HOLDING> placeholder, 5 possible holdings, correct label
            {
                "context": "Legal case context with <HOLDING> placeholder for the holding.",
                "endings": ["holding one", "holding two", "holding three", "holding four", "holding five"],
                "label": 0,
            },
            {
                "context": "Another legal case context requiring <HOLDING> determination.",
                "endings": ["first option", "second option", "third option", "fourth option", "fifth option"],
                "label": 2,
            },
        ],
        "openlifescienceai/medmcqa": [
            {"question": "Medical question", "exp": "Medical explanation"}
        ],
    }

    def _fake_load_dataset(
        name: str,
        subset: str | None,
        *,
        split: str,
        streaming: bool = False,
        download_mode: str | None = None,
    ) -> _DummyStream:
        key = name
        payload = domain_payloads.get(key)
        if payload is None:
            raise RuntimeError(f"Unexpected dataset name {name}")
        calls.append((name, split))
        return _DummyStream(payload)

    monkeypatch.setattr(dataset_loader, "load_dataset", _fake_load_dataset)

    out_by_domain = {
        dom: dataset_loader.get_dataset(dom, max_samples=2, split="train")
        for dom in ("arithmetic", "legal", "medical")
    }

    assert set(out_by_domain.keys()) == {"arithmetic", "legal", "medical"}
    for dom, examples in out_by_domain.items():
        assert examples, f"Expected non-empty dataset for domain {dom}"

    seen_datasets = {name for name, _ in calls}
    assert seen_datasets == set(domain_payloads.keys())


def test_dataset_retry_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _failing_loader(*args: Iterable[object], **kwargs: object) -> None:
        raise RuntimeError("network down")

    monkeypatch.setattr(dataset_loader, "load_dataset", _failing_loader)
    monkeypatch.setattr(dataset_loader, "_RETRIES_DEFAULT", 2)
    monkeypatch.setattr(dataset_loader, "_BACKOFF_DEFAULT", 0.0)
    monkeypatch.setattr(dataset_loader.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError):
        dataset_loader.get_dataset("arithmetic", max_samples=1, split="train")


@pytest.mark.slow
def test_real_datasets_available() -> None:
    domains = ("arithmetic", "legal", "medical")
    # Limit sample size to keep the test light while still verifying availability.
    dataset_loader.RealDatasetLoader.set_sample_limit(2)
    try:
        for domain in domains:
            examples = dataset_loader.get_dataset(domain, max_samples=2, split="train")
            assert (
                examples and len(examples) > 0
            ), f"Expected HuggingFace data for domain {domain}"
    finally:
        dataset_loader.RealDatasetLoader.set_sample_limit(None)
