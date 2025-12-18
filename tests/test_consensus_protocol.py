from __future__ import annotations

from swarm.consensus import ConsensusEngine, Vote


def test_consensus_commits_single_artefact_per_slot():
    cons = ConsensusEngine(quorum=2)
    # two votes for shaA at slot 1 -> commit
    assert cons.vote(Vote(agent_id=0, slot=1, sha256="A")) is None
    c = cons.vote(Vote(agent_id=1, slot=1, sha256="A"))
    assert c == "A"
    # further conflicting votes do not change commit
    cons.vote(Vote(agent_id=2, slot=1, sha256="B"))
    assert cons.committed(1) == "A"
