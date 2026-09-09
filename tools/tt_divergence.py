"""Does the transposition table carried across a game change our moves?

WHY (2026-09-09). Four games this project has analysed -- rounds 58, 64, 71
and 73 -- contain a move the engine's own cold analysis rejects at every time
control. The hypothesis was always TT staleness and it was never tested
directly, because "cold analysis" always used a fresh engine and the real game
never does.

Round 79 made it concrete: from a won position the engine repeated into a
threefold draw. A fresh engine at the same position plays Ke2 (+195). An
engine that has played the game to that point plays Qc8, the repeating move.

This replays a game twice at the same time control -- once with one engine
instance carrying its table forward exactly as a real game does, once with a
new engine per position -- and reports every move where they disagree, with a
deeper search as referee to say which was better.
"""
from __future__ import annotations

import argparse
import glob
import importlib
import os
import sys
from pathlib import Path

import chess
import chess.pgn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepblue.fastcore import from_fen                       # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="fastsearch140")
    ap.add_argument("--glob", default=r"C:/Users/uniqu/Downloads/aichessathon-round-7*.pgn")
    ap.add_argument("--ms", type=int, default=2000)
    ap.add_argument("--referee-ms", type=int, default=8000)
    args = ap.parse_args()

    mod = importlib.import_module("deepblue." + args.module)
    cls = getattr(mod, "FastEngine" + args.module.replace("fastsearch", ""))

    ref = cls()
    ref.search(from_fen(chess.STARTING_FEN), 200, 400)

    totals = [0, 0, 0.0]
    for path in sorted(glob.glob(args.glob)):
        game = chess.pgn.read_game(open(path, encoding="utf-8"))
        if game is None:
            continue
        h = game.headers
        us = chess.WHITE if h.get("White", "").lower().startswith("yumo") else chess.BLACK

        carried = cls()
        carried.search(from_fen(chess.STARTING_FEN), 200, 400)

        board = game.board()
        fens = [board.fen()]
        carried.record_game_position(from_fen(board.fen()))
        diffs = []
        for mv in game.mainline_moves():
            if board.turn == us:
                fen = board.fen()
                a, _, _, *_ = carried.search(from_fen(fen), args.ms, int(args.ms * 1.4))
                fresh = cls()
                fresh.search(from_fen(chess.STARTING_FEN), 200, 400)
                for f in fens:
                    fresh.record_game_position(from_fen(f))
                b_, _, _, *_ = fresh.search(from_fen(fen), args.ms, int(args.ms * 1.4))
                if a != b_:
                    # referee: score the position each move reaches, negated
                    def after(u):
                        nb = board.copy(); nb.push(chess.Move.from_uci(u))
                        if nb.is_game_over():
                            return 0 if not nb.is_checkmate() else -30000
                        _, s, _, *_ = ref.search(from_fen(nb.fen()),
                                                 args.referee_ms, int(args.referee_ms * 1.4))
                        return -s
                    sa, sb = after(a), after(b_)
                    diffs.append((board.fullmove_number, board.san(chess.Move.from_uci(a)),
                                  board.san(chess.Move.from_uci(b_)), sa, sb))
            board.push(mv)
            fens.append(board.fen())
            carried.record_game_position(from_fen(board.fen()))

        n_our = sum(1 for _ in ())  # placeholder, counted below
        label = os.path.basename(path)
        loss = sum(max(0, d[4] - d[3]) for d in diffs)
        totals[0] += len(diffs)
        totals[2] += loss
        print("%-46s %2d disagreements   carried-TT cost %+d cp total"
              % (label, len(diffs), loss))
        for n, a, b_, sa, sb in diffs:
            flag = "carried WORSE by %d" % (sb - sa) if sb > sa + 20 else (
                   "carried better by %d" % (sa - sb) if sa > sb + 20 else "equal")
            print("     %3d.  carried %-7s (%+5d)   fresh %-7s (%+5d)   %s"
                  % (n, a, sa, b_, sb, flag))

    print("\nTOTAL disagreements %d   total cp lost to the carried table %+d"
          % (totals[0], totals[2]))


if __name__ == "__main__":
    main()
