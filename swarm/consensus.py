from __future__ import annotations

"""Lightweight in-process consensus for artefact acceptance (proposal/vote/commit).

Safety: A slot commits at most one artefact (by SHA) when quorum votes observed.
Liveness (non-adversarial): With honest majority repeatedly voting, a commit is reached.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class Proposal:
    slot: int
    sha256: str


@dataclass(frozen=True)
class Vote:
    agent_id: int
    slot: int
    sha256: str


class ConsensusEngine:
    def __init__(self, quorum: int) -> None:
        self.quorum = int(quorum)
        self._votes: Dict[int, Dict[str, set[int]]] = {}
        self._commit: Dict[int, str] = {}

    def vote(self, v: Vote) -> Optional[str]:
        if v.slot in self._commit:
            return self._commit[v.slot]
        slot_votes = self._votes.setdefault(v.slot, {})
        voters = slot_votes.setdefault(v.sha256, set())
        voters.add(v.agent_id)
        if len(voters) >= self.quorum:
            # commit winner and discard others
            self._commit[v.slot] = v.sha256
            return v.sha256
        return None

    def committed(self, slot: int) -> Optional[str]:
        return self._commit.get(slot)
