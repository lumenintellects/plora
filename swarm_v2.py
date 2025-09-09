from __future__ import annotations

"""Thin CLI wrapper to run Swarm Sim v2.

This forwards to swarm.sim_v2_entry.main so you can run:

  python -m swarm_v2 --agents 6 --rounds 5 --graph_p 0.25 --security on
"""

from swarm.sim_v2_entry import main


if __name__ == "__main__":
    main()
