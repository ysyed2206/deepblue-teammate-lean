"""Compare search-ordering quality between instrumented Deep Blue variants.

This is a diagnostic, not an Elo test.  Both variants expose the same cutoff
counters so we can see whether a candidate actually moves beta-cutoff moves
earlier in the ordering while preserving the underlying search result.
"""
from __future__ import annotations

import argparse
import importlib
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepblue.fastcore import from_fen

SUITE = [
    ("start", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("closed", "r1bq1rk1/pp2nppp/2n1p3/2ppP3/3P4/P1PB1N2/2P2PPP/R1BQK2R w KQ - 0 1"),
    ("tactical", "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1"),
    ("endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
]


def load_variant(name: str):
    mod = importlib.import_module(f"deepblue.{name}")
    suffix = name.removeprefix("fastsearch") or ""
    cls = getattr(mod, "FastEngine" + suffix)
    required = (
        "CUT_TOTAL", "CUT_FIRST", "CUT_FIRST2", "CUT_MOVE_SUM",
        "CUT_TACTICAL", "CUT_QUIET", "QNODES", "NODES",
    )
    for attr in required:
        if not hasattr(mod, attr):
            raise RuntimeError(f"{name} is missing instrumentation counter {attr}")
    return mod, cls


def pct(num: int, den: int) -> float:
    return 100.0 * num / den if den else 0.0


def metrics(mod, engine):
    c = engine.counters
    cuts = int(c[mod.CUT_TOTAL])
    first = int(c[mod.CUT_FIRST])
    first2 = int(c[mod.CUT_FIRST2])
    move_sum = int(c[mod.CUT_MOVE_SUM])
    tactical = int(c[mod.CUT_TACTICAL])
    quiet = int(c[mod.CUT_QUIET])
    qnodes = int(c[mod.QNODES])
    nodes = int(c[mod.NODES])
    return {
        "cuts": cuts,
        "first": pct(first, cuts),
        "first2": pct(first2, cuts),
        "avg_idx": move_sum / cuts if cuts else 0.0,
        "tactical": pct(tactical, cuts),
        "quiet": pct(quiet, cuts),
        "qshare": pct(qnodes, nodes),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", nargs="?", default="fastsearch12i")
    ap.add_argument("candidate", nargs="?", default="fastsearch12")
    ap.add_argument("--movetime-ms", type=float, default=10000.0)
    ap.add_argument("--hard-mult", type=float, default=1.4)
    ap.add_argument("--max-depth", type=int, default=7)
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()

    variants = []
    for name in (args.baseline, args.candidate):
        mod, cls = load_variant(name)
        print(f"warming {name} ...", flush=True)
        mod.warm_up()
        variants.append((name, mod, cls))

    summaries = {name: [] for name, _, _ in variants}
    print()
    print(
        f"{'position':<11}{'variant':<15}{'move':>7}{'score':>8}{'d':>4}{'nodes':>11}"
        f"{'1st%':>9}{'<=2%':>9}{'avg#':>8}{'q%':>8}"
    )
    print("-" * 90)

    for posname, fen in SUITE:
        for name, mod, cls in variants:
            samples = []
            for _ in range(args.repeats):
                e = cls()
                result = e.search(
                    from_fen(fen), args.movetime_ms,
                    args.movetime_ms * args.hard_mult,
                    max_depth=args.max_depth,
                )
                samples.append((result, metrics(mod, e)))
            result, m = samples[-1]
            move, score, depth, nodes, ms = result
            print(
                f"{posname:<11}{name:<15}{str(move):>7}{score:>8}{depth:>4}{nodes:>11,}"
                f"{m['first']:>8.1f}%{m['first2']:>8.1f}%{m['avg_idx']:>8.2f}{m['qshare']:>7.1f}%"
            )
            print(
                f"{'':<26}cutoffs={m['cuts']:,}  tactical={m['tactical']:.1f}%  quiet={m['quiet']:.1f}%  {ms:.1f} ms"
            )
            summaries[name].append(m)
        print()

    print("summary (median across positions)")
    for name, _, _ in variants:
        rows = summaries[name]
        print(
            f"{name:<15} first={statistics.median(r['first'] for r in rows):.1f}%  "
            f"<=2={statistics.median(r['first2'] for r in rows):.1f}%  "
            f"avg#={statistics.median(r['avg_idx'] for r in rows):.2f}  "
            f"qshare={statistics.median(r['qshare'] for r in rows):.1f}%"
        )


if __name__ == "__main__":
    main()
