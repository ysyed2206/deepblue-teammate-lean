"""How much does our engine lose per move against a Stockfish move ranking?

WHY (2026-09-09, the user's framing). "We did not blunder, we played good
moves but not the best moves." Blunder scans miss that entirely -- they use
our own engine as referee and it agrees with itself. The Lichess database
stores several scored candidate moves per position, which is exactly a
ranked list from a deep search, so our choice can be priced directly.

Reported per phase, because the complaint is specifically about the opening
and early middlegame.

A move our engine picks that Stockfish did not list is scored at the WORST
listed move's value. Stockfish usually lists the top few, so a move outside
the list is normally worse than all of them: the number below is therefore a
LOWER BOUND on what we lose, not an overestimate.
"""
from __future__ import annotations

import argparse
import importlib
import multiprocessing as mp
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_STATE = {}


def _init(module, ms):
    from deepblue.fastcore import from_fen
    mod = importlib.import_module("deepblue." + module)
    eng = getattr(mod, "FastEngine" + module.replace("fastsearch", ""))()
    eng.search(from_fen(chess.STARTING_FEN), 200, 400)
    _STATE["eng"] = eng
    _STATE["from_fen"] = from_fen
    _STATE["ms"] = ms


def _one(rec):
    fen, moves = rec
    eng, from_fen, ms = _STATE["eng"], _STATE["from_fen"], _STATE["ms"]
    board = chess.Board(fen)
    stm_white = board.turn == chess.WHITE
    # cp is White-relative; the mover wants it high as White, low as Black.
    scored = {u: (c if stm_white else -c) for u, c in moves}
    best = max(scored.values())
    worst = min(scored.values())
    try:
        uci, _, _, *_ = eng.search(from_fen(fen), ms, int(ms * 1.4))
    except Exception:
        return None
    got = scored.get(uci, worst)
    phase = min(24, sum(1 for _ in board.piece_map()))
    return (best - got, uci in scored, board.fullmove_number, phase)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="fastsearch162")
    ap.add_argument("--positions", default="scratch_moveranks.txt")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--ms", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-fullmove", type=int, default=0,
                    help="0 = all; else keep positions at or before this move number")
    args = ap.parse_args()

    recs = []
    with open(args.positions, encoding="utf-8") as fh:
        for line in fh:
            fen, rest = line.rstrip("\n").split("\t", 1)
            if args.max_fullmove:
                try:
                    if int(fen.split()[5]) > args.max_fullmove:
                        continue
                except (IndexError, ValueError):
                    continue
            moves = []
            for tok in rest.split():
                u, c = tok.rsplit(":", 1)
                moves.append((u, int(c)))
            recs.append((fen, moves))
            if len(recs) >= args.n:
                break

    with mp.Pool(args.workers, initializer=_init,
                 initargs=(args.module, args.ms)) as pool:
        out = [r for r in pool.map(_one, recs) if r]

    loss = [r[0] for r in out]
    listed = [r[1] for r in out]
    import statistics
    print("module %s   %d positions   %dms/move" % (args.module, len(out), args.ms))
    print("  mean cp lost per move      %7.1f   <- lower bound" % statistics.mean(loss))
    print("  median cp lost             %7.1f" % statistics.median(loss))
    print("  played Stockfish's best    %6.1f%%" % (100 * sum(1 for x in loss if x == 0) / len(loss)))
    print("  move not in SF's list      %6.1f%%" % (100 * (1 - sum(listed) / len(listed))))
    for lo, hi, nm in ((0, 15, "moves 1-15 "), (16, 30, "moves 16-30"), (31, 999, "moves 31+  ")):
        sub = [r[0] for r in out if lo <= r[2] <= hi]
        if len(sub) >= 20:
            print("    %s n=%4d  mean %6.1f  best-move %5.1f%%"
                  % (nm, len(sub), statistics.mean(sub),
                     100 * sum(1 for x in sub if x == 0) / len(sub)))


if __name__ == "__main__":
    mp.freeze_support()
    main()
