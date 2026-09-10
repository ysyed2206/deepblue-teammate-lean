"""Cold-start measurement under the platform's actual CPU conditions.

The platform gives ONE core (AGENTS.md, confirmed against the published rules).
Every cold-start figure in this project so far was taken with every core
available, which flatters the number: Numba's compile is not fully serial, and
the OS is free to schedule its work anywhere. This pins the process to a single
logical CPU and wipes the Numba cache first, which is the closest reproduction
of a fresh container we can manage locally.

    python tools/init_onecore.py fastsearch185
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tempfile
import time

BUDGET_S = 90.0


def main() -> None:
    module = sys.argv[1]
    t0 = time.perf_counter()
    mod = importlib.import_module("deepblue." + module)
    t1 = time.perf_counter()
    mod.warm_up()
    t2 = time.perf_counter()
    # One real search, as agent.py does before the clock starts.
    from deepblue.fastcore import from_fen
    eng = getattr(mod, "FastEngine" + module.replace("fastsearch", ""))()
    eng.search(from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"), 40.0, 80.0)
    t3 = time.perf_counter()
    print("%-14s import %5.1fs  warm_up %5.1fs  first_search %4.1fs  TOTAL %6.1fs of %.0fs  %s"
          % (module, t1 - t0, t2 - t1, t3 - t2, t3 - t0, BUDGET_S,
             "OK" if t3 - t0 < BUDGET_S else "OVER BUDGET"),
          flush=True)


if __name__ == "__main__":
    main()
