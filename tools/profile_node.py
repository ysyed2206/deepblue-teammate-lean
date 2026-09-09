"""Where does node time actually go?

WHY THIS EXISTS. One earlier attempt concluded "eval is 80% of runtime". It
was timing evaluate() from PYTHON, so it measured Numba's dispatch overhead
per call rather than the work inside -- and the lazy-eval change built on that
answer gained nothing. The lesson is that a jitted function's cost cannot be
measured from outside the jit.

So each primitive is timed inside its OWN @njit loop, where the call is
compiled in and the dispatch cost is gone. Accumulating the result into a sink
stops LLVM deleting the loop as dead code. Per-call costs are then multiplied
by the engine's own per-search counters to give each primitive's real share of
a search.

This is the measurement that decides whether speed work or more techniques is
the better use of the remaining time: doubling node rate is worth far more
Elo than any single technique left on the list, but only if there is a hotspot
to fix.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import chess
import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepblue.fastcore import (                                   # noqa: E402
    from_fen, generate_pseudo_legal, in_check, make_move, unmake_move,
)
from deepblue.fastsearch118 import (                              # noqa: E402
    CHECK, EG_TABLE, EVAL, FastEngine118, GEN, MAKE, MG_TABLE, NODES,
    ORDER, PHASE_TABLE, QNODES, evaluate,
)

REPS = 200_000


@njit(cache=False, nogil=True)
def _time_eval(bb, st, mg, eg, ph, reps):
    sink = np.int64(0)
    for _ in range(reps):
        sink += evaluate(bb, st, mg, eg, ph)
    return sink


@njit(cache=False, nogil=True)
def _time_incheck(bb, occ, st, reps):
    sink = 0
    for _ in range(reps):
        if in_check(bb, occ, st, st[0]):
            sink += 1
    return sink


@njit(cache=False, nogil=True)
def _time_gen(bb, occ, mail, st, buf, reps):
    sink = 0
    for _ in range(reps):
        sink += generate_pseudo_legal(bb, occ, mail, st, buf)
    return sink


def bench(label, fn, args, reps=REPS):
    fn(*args, 1000)                                   # compile + warm
    start = time.perf_counter()
    fn(*args, reps)
    return label, (time.perf_counter() - start) / reps * 1e9      # ns/call


def main() -> None:
    fen = "r2q1rk1/pp2bppp/2n1bn2/2pp4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 11"
    bb, occ, mail, st = from_fen(fen)
    buf = np.zeros(256, dtype=np.uint32)

    rows = [
        bench("evaluate", _time_eval, (bb, st, MG_TABLE, EG_TABLE, PHASE_TABLE)),
        bench("in_check", _time_incheck, (bb, occ, st)),
        bench("generate_pseudo_legal", _time_gen, (bb, occ, mail, st, buf)),
    ]

    engine = FastEngine118()
    engine.search(from_fen(chess.STARTING_FEN), 200, 400)
    _, _, _, nodes, elapsed_ms = engine.search(from_fen(fen), 2000, 2800)
    counts = {"evaluate": int(engine.counters[EVAL]),
              "in_check": int(engine.counters[CHECK]),
              "generate_pseudo_legal": int(engine.counters[GEN])}
    total_nodes = int(engine.counters[NODES])

    print(f"search: {total_nodes:,} nodes ({int(engine.counters[QNODES]):,} qnodes) "
          f"in {elapsed_ms:.0f}ms = {total_nodes/elapsed_ms*1000:,.0f} nps\n")
    print(f"{'primitive':<24}{'ns/call':>10}{'calls':>12}{'total ms':>10}{'share':>8}")
    accounted = 0.0
    for label, ns in rows:
        calls = counts[label]
        total = ns * calls / 1e6
        accounted += total
        print(f"{label:<24}{ns:>10.0f}{calls:>12,}{total:>10.1f}{100*total/elapsed_ms:>7.1f}%")
    print(f"{'-'*64}")
    print(f"{'accounted for':<24}{'':>10}{'':>12}{accounted:>10.1f}"
          f"{100*accounted/elapsed_ms:>7.1f}%")
    print(f"{'unaccounted (search':<24}{'':>10}{'':>12}"
          f"{elapsed_ms-accounted:>10.1f}{100*(elapsed_ms-accounted)/elapsed_ms:>7.1f}%")
    print(" logic, make/unmake, ordering, TT, recursion)")


if __name__ == "__main__":
    main()
