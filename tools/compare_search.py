"""S0 vs S1-search0 on identical positions: the architecture measurement.

Node-count convention, stated so the numbers are not misread: BOTH engines
count one node on entry to a negamax node and one more on entry to a
quiescence node, so a leaf transition is counted twice in both. The convention
is identical, which makes the two directly comparable to each other, and it is
not comparable to any other engine's published NPS.

This reports search NPS. It is not perft NPS and the two are never quoted for
each other.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepblue import fastsearch as FS  # noqa: E402
from deepblue import fastsearch1 as FS1  # noqa: E402
from deepblue.fastcore import from_fen  # noqa: E402
from deepblue.reference import Engine  # noqa: E402

SUITE: list[tuple[str, str]] = [
    ("opening", chess.STARTING_FEN),
    ("open middlegame", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("closed middlegame", "r1bq1rk1/pp2nppp/2n1p3/2ppP3/3P4/P1PB1N2/2P2PPP/R1BQK2R w KQ - 0 1"),
    ("tactical", "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1"),
    ("endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--movetime-ms", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=2)
    arguments = parser.parse_args()

    FS.warm_up()
    FS1.warm_up()
    engine0 = FS.FastEngine()
    engine1 = FS1.FastEngine1()

    print(f"{'position':<20}{'engine':>9}{'nps':>11}{'depth':>7}{'nodes':>11}"
          f"{'score':>8}{'move':>7}{'ms':>7}")
    print("-" * 80)
    depth_gain = []
    node_ratio = []
    for name, fen in SUITE:
        rows = []
        for _ in range(arguments.repeats):
            m0, s0, d0, n0, e0 = engine0.search(
                from_fen(fen), arguments.movetime_ms, arguments.movetime_ms * 1.2
            )
            m1, s1, d1, n1, e1 = engine1.search(
                from_fen(fen), arguments.movetime_ms, arguments.movetime_ms * 1.2
            )
            rows.append(((m0, s0, d0, n0, e0), (m1, s1, d1, n1, e1)))
        (m0, s0, d0, n0, e0) = rows[-1][0]
        (m1, s1, d1, n1, e1) = rows[-1][1]
        print(f"{name:<20}{'search0':>9}{n0 / max(e0, 1) * 1000:>11,.0f}{d0:>7}"
              f"{n0:>11,}{s0:>8}{str(m0):>7}{e0:>7.0f}")
        print(f"{'':<20}{'search1':>9}{n1 / max(e1, 1) * 1000:>11,.0f}{d1:>7}"
              f"{n1:>11,}{s1:>8}{str(m1):>7}{e1:>7.0f}")
        depth_gain.append(d1 - d0)
        node_ratio.append(n1 / max(n0, 1))
    print("-" * 80)
    print(f"median depth gain search1 over search0: {statistics.median(depth_gain):+.0f} ply")
    print(f"median node ratio  search1 / search0  : {statistics.median(node_ratio):.2f}x")


def _unused() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--movetime-ms", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=3)
    arguments = parser.parse_args()

    FS.warm_up()
    engine = FS.FastEngine()

    header = f"{'position':<20}{'S0 nps':>11}{'S0 d':>5}{'S1 nps':>12}{'S1 d':>5}{'speedup':>9}{'+depth':>8}"
    print(header)
    print("-" * len(header))
    ratios: list[float] = []
    depth_gains: list[int] = []

    for name, fen in SUITE:
        reference_samples = []
        fast_samples = []
        for _ in range(arguments.repeats):
            slow = Engine()
            board = chess.Board(fen)
            slow.record_game_position(board)
            result = slow.search(board, arguments.movetime_ms, arguments.movetime_ms * 1.2)
            reference_samples.append(
                (result.nodes / max(result.elapsed_ms, 1.0) * 1000.0, result.depth)
            )
            _, _, depth, nodes, elapsed = engine.search(
                from_fen(fen), arguments.movetime_ms, arguments.movetime_ms * 1.2
            )
            fast_samples.append((nodes / max(elapsed, 1.0) * 1000.0, depth))

        s0_nps = statistics.median(s[0] for s in reference_samples)
        s0_depth = int(statistics.median(s[1] for s in reference_samples))
        s1_nps = statistics.median(s[0] for s in fast_samples)
        s1_depth = int(statistics.median(s[1] for s in fast_samples))
        ratios.append(s1_nps / s0_nps)
        depth_gains.append(s1_depth - s0_depth)
        print(
            f"{name:<20}{s0_nps:>11,.0f}{s0_depth:>5}{s1_nps:>12,.0f}{s1_depth:>5}"
            f"{s1_nps / s0_nps:>8.1f}x{s1_depth - s0_depth:>+8d}"
        )
    print("-" * len(header))
    print(f"{'median':<20}{'':>11}{'':>5}{'':>12}{'':>5}"
          f"{statistics.median(ratios):>8.1f}x{statistics.median(depth_gains):>+8.0f}")


if __name__ == "__main__":
    main()
