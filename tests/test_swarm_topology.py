from swarm.graph_engine import build_topology


def test_line_topology_connected():
    topo = build_topology("line", 5)
    # each node has <=2 neighbours, connected ends have 1
    assert topo[0] == [1]
    assert topo[4] == [3]
    # internal have 2 neighbours
    assert topo[2] == [1, 3]


def test_mesh_topology():
    topo = build_topology("mesh", 4)
    for i in range(4):
        assert sorted(topo[i]) == [j for j in range(4) if j != i]
