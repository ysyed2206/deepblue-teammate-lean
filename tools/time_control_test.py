"""Prove the compiled search can actually be stopped.

A jitted search that cannot be interrupted until a depth finishes is a flag
waiting to happen, and a flag is an instant loss. This measures the thing that
matters: how far past its hard budget the search actually runs, including on
positions where node growth explodes.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepblue import fastsearch as FS  # noqa: E402
from deepblue.fastcore import from_fen  # noqa: E402

POSITIONS = [
    ("opening", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("kiwipete (wide)", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("tactical (explosive)", "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1"),
    ("promotion race", "8/P1P5/8/8/8/8/5p1p/4K2k w - - 0 1"),
    ("endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
]
BUDGETS_MS = [20, 50, 100, 250, 500, 1000, 2000]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=2)
    arguments = parser.parse_args()
    FS.warm_up()
    engine = FS.FastEngine()

    print(f"{'position':<22}{'budget':>8}{'actual':>9}{'overshoot':>11}{'depth':>7}{'nodes':>11}")
    print("-" * 68)
    worst = 0.0
    worst_case = ""
    for name, fen in POSITIONS:
        for budget in BUDGETS_MS:
            for _ in range(arguments.repeats):
                started = time.monotonic()
                _, _, depth, nodes, elapsed = engine.search(from_fen(fen), budget, budget)
                wall = (time.monotonic() - started) * 1000.0
                overshoot = wall - budget
                if overshoot > worst:
                    worst = overshoot
                    worst_case = f"{name} at {budget} ms"
            print(f"{name:<22}{budget:>8}{wall:>9.1f}{overshoot:>+11.1f}{depth:>7}{nodes:>11,}")
    print("-" * 68)
    print(f"worst overshoot: {worst:+.1f} ms  ({worst_case})")
    print("A flag needs the overshoot to exceed the whole remaining clock, so what")
    print("matters is that this stays a small constant rather than growing with depth.")


if __name__ == "__main__":
    main()
