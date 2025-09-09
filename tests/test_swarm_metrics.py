import math
from swarm.metrics import coverage, entropy_avg, mutual_information


def _make_knowledge_unique_domains(N: int):
    # agent i owns domain_i only
    return {i: {f"d{i}"} for i in range(N)}


def _make_knowledge_full(N: int, D: int):
    doms = [f"d{i}" for i in range(D)]
    return {i: set(doms) for i in range(N)}


def test_entropy_and_mi_initial():
    N = 5
    know = _make_knowledge_unique_domains(N)
    cov = coverage(know)
    # Each domain present in exactly 1/5 agents
    assert all(math.isclose(p, 1 / N) for p in cov.values())

    H = entropy_avg(coverage_map=cov)
    # Check >0
    assert H > 0

    I0 = mutual_information(know)
    assert math.isclose(I0, math.log2(N), rel_tol=1e-6)


def test_entropy_and_mi_full_diffusion():
    N = 5
    D = 5
    know = _make_knowledge_full(N, D)
    cov = coverage(know)
    # p_d ==1
    assert all(p == 1.0 for p in cov.values())
    H = entropy_avg(coverage_map=cov)
    assert H == 0.0
    I = mutual_information(know)
    assert I == 0.0
