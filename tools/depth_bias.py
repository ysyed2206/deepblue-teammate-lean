"""Is our evaluation systematically optimistic at shallow depth?

Positions from LOST games all look worse with depth -- that is selection bias,
not a finding. This measures the same thing on a NEUTRAL sample carrying deep
Stockfish scores, so the question becomes: as our search deepens, does its
score move TOWARD Stockfish's, and is the residual biased in one direction?
"""
from __future__ import annotations
import argparse, importlib, multiprocessing as mp, sys
from pathlib import Path
import chess
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_S = {}

def _init(module, shallow, deep):
    from deepblue.fastcore import from_fen
    mod = importlib.import_module("deepblue." + module); mod.warm_up()
    eng = getattr(mod, "FastEngine" + module.replace("fastsearch", ""))()
    eng.search(from_fen(chess.STARTING_FEN), 300, 600)
    _S.update(eng=eng, ff=from_fen, sh=shallow, dp=deep)

def _one(rec):
    cp, fen = rec
    eng, ff = _S["eng"], _S["ff"]
    try:
        b = chess.Board(fen)
        white = b.turn == chess.WHITE
        _, s1, _, *_ = eng.search(ff(fen), 60000, 90000, max_depth=_S["sh"])
        _, s2, _, *_ = eng.search(ff(fen), 60000, 90000, max_depth=_S["dp"])
    except Exception:
        return None
    # to White-relative, to match the Stockfish label
    return (s1 if white else -s1, s2 if white else -s2, cp)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="fastsearch185")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--shallow", type=int, default=8)
    ap.add_argument("--deep", type=int, default=18)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    recs = []
    with open("scratch_sf_quiet.txt", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i % 37:      # spread across the file
                continue
            cp, fen = line.rstrip("\n").split("\t", 1)
            recs.append((int(cp), fen))
            if len(recs) >= a.n:
                break
    with mp.Pool(a.workers, initializer=_init, initargs=(a.module, a.shallow, a.deep)) as pool:
        out = [r for r in pool.map(_one, recs) if r]
    import statistics
    sh = [r[0] for r in out]; dp = [r[1] for r in out]; sf = [r[2] for r in out]
    dsh = [a_ - b_ for a_, b_ in zip(sh, sf)]
    ddp = [a_ - b_ for a_, b_ in zip(dp, sf)]
    drift = [b_ - a_ for a_, b_ in zip(sh, dp)]
    print("module %s   %d positions   depth %d vs depth %d" % (a.module, len(out), a.shallow, a.deep))
    print("  mean error vs Stockfish, shallow  %+7.1f cp   (|err| %6.1f)" % (statistics.mean(dsh), statistics.mean(map(abs, dsh))))
    print("  mean error vs Stockfish, deep     %+7.1f cp   (|err| %6.1f)" % (statistics.mean(ddp), statistics.mean(map(abs, ddp))))
    print("  mean drift shallow->deep          %+7.1f cp" % statistics.mean(drift))
    print("  positions where deep is LOWER than shallow: %.0f%%" % (100 * sum(1 for d in drift if d < 0) / len(drift)))
    print("\n  a negative mean ERROR means we are PESSIMISTIC vs Stockfish;")
    print("  a negative mean DRIFT means deeper search lowers our own score.")

if __name__ == "__main__":
    mp.freeze_support(); main()
