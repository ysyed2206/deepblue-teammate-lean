"""Deterministic side-by-side benchmark for Deep Blue fast-search variants.

This is intentionally independent of python-chess so it can run in any Numba
sandbox.  It is a *search engineering* benchmark, not an Elo test: paired games
on the pinned competition environment remain the promotion gate.
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
    cls_name = "FastEngine" + suffix
    cls = getattr(mod, cls_name)
    return mod, cls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="+", help="e.g. fastsearch1 fastsearch2 fastsearch4")
    ap.add_argument("--movetime-ms", type=float, default=500.0)
    ap.add_argument("--hard-mult", type=float, default=1.4)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--max-depth", type=int, default=64)
    args = ap.parse_args()

    variants = []
    for name in args.variants:
        mod, cls = load_variant(name)
        print(f"warming {name} ...", flush=True)
        mod.warm_up()
        variants.append((name, mod, cls))

    print()
    header = f"{'position':<11}{'variant':<13}{'move':>7}{'score':>8}{'d':>4}{'nodes':>11}{'ms':>9}{'nps':>11}"
    print(header)
    print("-" * len(header))

    depths = {name: [] for name, _, _ in variants}
    nodes = {name: [] for name, _, _ in variants}

    for posname, fen in SUITE:
        baseline_nodes = None
        for name, mod, cls in variants:
            samples = []
            for _ in range(args.repeats):
                e = cls()
                result = e.search(
                    from_fen(fen), args.movetime_ms,
                    args.movetime_ms * args.hard_mult,
                    max_depth=args.max_depth,
                )
                samples.append((result, e))
            # Last sample is printed; medians are used for summary only when
            # repeats > 1 to avoid hiding move/score disagreements.
            (move, score, depth, n, ms), e = samples[-1]
            nps = n / max(ms, 0.001) * 1000.0
            print(f"{posname:<11}{name:<13}{str(move):>7}{score:>8}{depth:>4}{n:>11,}{ms:>9.1f}{nps:>11,.0f}")
            depths[name].append(statistics.median(x[0][2] for x in samples))
            med_nodes = statistics.median(x[0][3] for x in samples)
            if baseline_nodes is None:
                baseline_nodes = med_nodes
            nodes[name].append(med_nodes / max(baseline_nodes, 1))
            if hasattr(mod, "RFP_TRY"):
                print(f"{'':<24}RFP {int(e.counters[mod.RFP_CUT]):,}/{int(e.counters[mod.RFP_TRY]):,} cut/try")
        print()

    print("summary")
    for name, _, _ in variants:
        print(
            f"  {name:<13} median depth={statistics.median(depths[name]):.1f}  "
            f"median node ratio vs first={statistics.median(nodes[name]):.3f}"
        )


if __name__ == "__main__":
    main()
