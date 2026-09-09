"""Measure our static evaluation against deep Stockfish scores.

WHY THIS EXISTS (2026-09-09). A self-play match costs six hours and resolves
about +/-28 Elo, so every eval change this project has tried came back "flat"
and was thrown away one at a time. This scores the same change against 400k
Stockfish-labelled positions in a couple of minutes.

It does NOT replace the match. It is a screen: a change that does not reduce
eval error against Stockfish has no mechanism by which to win a match, so it
never needs to spend six hours proving that. Changes that DO reduce error
still have to win a match, because lower static error and more Elo are not
the same thing -- search interacts.

Both our eval and Lichess cp are put in the same White-relative frame.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepblue.fastcore import from_fen                    # noqa: E402
from deepblue.eval_terms import game_phase                # noqa: E402

# Texel's scaling: cp -> expected score. 1/400 is the conventional constant.
K = 1.0 / 400.0


def win_prob(cp: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.power(10.0, -K * cp))


def load(path: str, limit: int):
    cps, fens = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            cp, fen = line.rstrip("\n").split("\t", 1)
            cps.append(int(cp))
            fens.append(fen)
            if len(cps) >= limit:
                break
    return np.array(cps, dtype=np.int64), fens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="fastsearch150")
    ap.add_argument("--positions", default="scratch_sf_positions.txt")
    ap.add_argument("--limit", type=int, default=200_000)
    args = ap.parse_args()

    mod = importlib.import_module("deepblue." + args.module)
    evaluate = mod.evaluate
    MG, EG, PH = mod.MG_TABLE, mod.EG_TABLE, mod.PHASE_TABLE

    sf, fens = load(args.positions, args.limit)
    ours = np.empty(len(fens), dtype=np.int64)
    phases = np.empty(len(fens), dtype=np.int64)
    bad = 0
    for i, fen in enumerate(fens):
        try:
            bb, st = from_fen(fen)[0], from_fen(fen)[1]
            v = int(evaluate(bb, st, MG, EG, PH))
            ours[i] = v if st[0] == 0 else -v          # -> White-relative
            phases[i] = game_phase(bb, PH, 24)
        except Exception:
            ours[i] = 0
            phases[i] = -1
            bad += 1
    ok = phases >= 0
    ours, sf, phases = ours[ok], sf[ok], phases[ok]

    err = ours - sf
    mse = float(np.mean((win_prob(ours.astype(float)) - win_prob(sf.astype(float))) ** 2))
    print("module %s   positions %d   (%d unparsable)" % (args.module, len(sf), bad))
    print("  MAE            %7.1f cp" % np.mean(np.abs(err)))
    print("  median |err|   %7.1f cp" % np.median(np.abs(err)))
    print("  bias (ours-sf) %+7.1f cp" % np.mean(err))
    print("  corr            %7.4f" % np.corrcoef(ours, sf)[0, 1])
    print("  TEXEL MSE       %7.6f   <- the screening number, lower is better" % mse)

    print("\n  by phase (24=opening material, 0=bare kings)")
    for lo, hi, name in ((18, 24, "opening  "), (10, 17, "middle   "),
                         (4, 9, "late     "), (0, 3, "endgame  ")):
        m = (phases >= lo) & (phases <= hi)
        if m.sum() < 50:
            continue
        print("    %s n=%6d  MAE %6.1f  bias %+7.1f  corr %.4f"
              % (name, m.sum(), np.mean(np.abs(err[m])), np.mean(err[m]),
                 np.corrcoef(ours[m], sf[m])[0, 1]))

    print("\n  by how wrong we are about who is winning")
    flip = np.sign(ours) != np.sign(sf)
    big = np.abs(sf) > 200
    print("    sign disagreements            %5.1f%%" % (100 * flip.mean()))
    print("    sign wrong while SF says >200 %5.1f%%  (n=%d)"
          % (100 * flip[big].mean(), big.sum()))


if __name__ == "__main__":
    main()
