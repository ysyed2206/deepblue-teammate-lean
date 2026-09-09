"""Decompose our eval into its terms for every quiet Stockfish position.

evaluate() is base_evaluate plus a sum of eight independent White-relative
terms. Writing each term out per position turns "what does our eval score"
into a linear algebra problem: for any vector of per-term scales the total
is a dot product, so a fit that would need one Numba recompile per trial
becomes instant.

Saves scratch_terms.npz: X (n, 9) term values White-relative, y (n,) the
Stockfish cp, phase (n,).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepblue.fastcore import from_fen                          # noqa: E402
from deepblue.eval_terms import (                               # noqa: E402
    bishop_pair_white_relative, game_phase,
    king_attack_danger_strongq_white_relative, king_safety_white_relative,
    knight_outposts_white_relative, mobility_white_relative,
    passed_pawns_white_relative, pawn_structure_white_relative,
    rook_files_white_relative,
)
from deepblue.fastsearch150 import (                            # noqa: E402
    EG_TABLE, MG_TABLE, PHASE_TABLE, base_evaluate,
)

TP = 24
NAMES = ["base", "passed", "shield", "danger", "mobility",
         "bishop_pair", "pawn_struct", "rook_files", "outposts"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", default="scratch_sf_quiet.txt")
    ap.add_argument("--limit", type=int, default=120_000)
    ap.add_argument("--out", default="scratch_terms.npz")
    args = ap.parse_args()

    cps, fens = [], []
    with open(args.positions, encoding="utf-8") as fh:
        for line in fh:
            cp, fen = line.rstrip("\n").split("\t", 1)
            cps.append(int(cp)); fens.append(fen)
            if len(cps) >= args.limit:
                break

    n = len(fens)
    X = np.zeros((n, 9), dtype=np.float64)
    ph = np.zeros(n, dtype=np.int64)
    for i, fen in enumerate(fens):
        bb, st = from_fen(fen)[0], from_fen(fen)[1]
        p = game_phase(bb, PHASE_TABLE, TP)
        ph[i] = p
        b = int(base_evaluate(bb, st, MG_TABLE, EG_TABLE, PHASE_TABLE))
        X[i, 0] = b if st[0] == 0 else -b            # base is stm-relative
        X[i, 1] = passed_pawns_white_relative(bb)
        X[i, 2] = king_safety_white_relative(bb, p, TP)
        X[i, 3] = king_attack_danger_strongq_white_relative(bb, p, TP)
        X[i, 4] = mobility_white_relative(bb)
        X[i, 5] = bishop_pair_white_relative(bb)
        X[i, 6] = pawn_structure_white_relative(bb)
        X[i, 7] = rook_files_white_relative(bb)
        X[i, 8] = knight_outposts_white_relative(bb)
        if i % 20000 == 0:
            print("  %d/%d" % (i, n), flush=True)

    np.savez_compressed(args.out, X=X, y=np.array(cps, dtype=np.float64),
                        phase=ph, names=np.array(NAMES))
    print("wrote %s  n=%d" % (args.out, n))
    tot = X.sum(axis=1)
    print("\nterm contribution (mean |value|, cp):")
    for j, nm in enumerate(NAMES):
        print("  %-12s %7.1f" % (nm, np.mean(np.abs(X[:, j]))))
    print("  %-12s %7.1f" % ("SUM", np.mean(np.abs(tot))))


if __name__ == "__main__":
    main()
