"""Is a timing change budget-neutral, and does it change moves?

Measured across 94 real games we have no spare clock: 3.15s per move in moves
1-15 against opponents' 3.73s, and we drop under 10s in 23 games to their 13.
So a stability-aware allocator must REDISTRIBUTE time, not add it. This runs
two builds over the same positions with the same budget and reports total
time spent and how often the chosen move differs.
"""
from __future__ import annotations

import argparse
import importlib
import multiprocessing as mp
import sys
import time
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_S = {}


def _init(module, ms):
    from deepblue.fastcore import from_fen
    mod = importlib.import_module("deepblue." + module)
    mod.warm_up()
    eng = getattr(mod, "FastEngine" + module.replace("fastsearch", ""))()
    eng.search(from_fen(chess.STARTING_FEN), 300, 600)
    _S.update(eng=eng, from_fen=from_fen, ms=ms)


def _one(fen):
    eng, from_fen, ms = _S["eng"], _S["from_fen"], _S["ms"]
    t0 = time.monotonic()
    try:
        mv, sc, d, *_ = eng.search(from_fen(fen), ms, ms * 1.4)
    except Exception:
        return None
    return (time.monotonic() - t0) * 1000.0, mv, d


def run(module, fens, ms, workers):
    with mp.Pool(workers, initializer=_init, initargs=(module, ms)) as pool:
        return [r for r in pool.map(_one, fens) if r]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="fastsearch180")
    ap.add_argument("--b", default="fastsearch181")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--ms", type=float, default=2500)
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    fens = []
    with open("scratch_moveranks.txt", encoding="utf-8") as fh:
        for line in fh:
            fens.append(line.split("\t", 1)[0])
            if len(fens) >= args.n:
                break

    ra = run(args.a, fens, args.ms, args.workers)
    rb = run(args.b, fens, args.ms, args.workers)
    n = min(len(ra), len(rb))
    ta = sum(r[0] for r in ra[:n]); tb = sum(r[0] for r in rb[:n])
    diff = sum(1 for i in range(n) if ra[i][1] != rb[i][1])
    da = sum(r[2] for r in ra[:n]) / n; db = sum(r[2] for r in rb[:n]) / n
    print("%d positions at %.0fms budget" % (n, args.ms))
    print("  %-14s total %8.0fms   mean %6.0fms   mean depth %.2f" % (args.a, ta, ta / n, da))
    print("  %-14s total %8.0fms   mean %6.0fms   mean depth %.2f" % (args.b, tb, tb / n, db))
    print("  time change %+.1f%%   depth change %+.2f ply   moves differing %d/%d (%.0f%%)"
          % (100 * (tb / ta - 1), db - da, diff, n, 100 * diff / n))


if __name__ == "__main__":
    mp.freeze_support()
    main()
