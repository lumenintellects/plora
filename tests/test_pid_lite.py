from __future__ import annotations

from swarm.metrics import cooccurrence_excess, pid_lite_summary


def test_cooccurrence_and_pid_lite():
    # Construct knowledge where d0 and d1 co-occur more than chance
    know = {
        0: {"d0", "d1"},
        1: {"d0", "d1"},
        2: {"d0"},
        3: {"d1"},
        4: {"d2"},
    }
    ex = cooccurrence_excess(know, "d0", "d1")
    assert ex > 0.0
    summ = pid_lite_summary(know)
    assert summ["synergy_mean"] >= 0.0 and summ["redundancy_mean"] >= 0.0
