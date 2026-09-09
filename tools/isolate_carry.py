"""Which piece of carried-over state changes our move?

An engine that has played a game differs from a fresh one in FOUR ways, not
one: the transposition table, the quiet-move/killer history, the continuation
history and the correction history. Round 79's draw was blamed on "the TT"
before any of them had been separated. This clears exactly one at a time,
at every position in the game, and reports which clearing restores the move a
fresh engine would play.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import chess
import chess.pgn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepblue.fastcore import from_fen                        # noqa: E402

GROUPS = {
    "tt":       ("tt_key", "tt_score", "tt_depth", "tt_bound", "tt_move"),
    "history":  ("history", "killers"),
    "conthist": ("continuation_history",),
    "corrhist": ("corr_hist",),
}


def clear(engine, names):
    for n in names:
        arr = getattr(engine, n)
        arr[:] = -1 if n == "tt_depth" else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="fastsearch140")
    ap.add_argument("--pgn", required=True)
    ap.add_argument("--ms", type=int, default=2000)
    ap.add_argument("--from-move", type=int, default=1)
    args = ap.parse_args()

    mod = importlib.import_module("deepblue." + args.module)
    cls = getattr(mod, "FastEngine" + args.module.replace("fastsearch", ""))

    game = chess.pgn.read_game(open(args.pgn, encoding="utf-8"))
    h = game.headers
    us = chess.WHITE if h.get("White", "").lower().startswith("yumo") else chess.BLACK

    variants = ["carried", "fresh"] + ["carried-no-" + k for k in GROUPS]
    engines = {}
    for v in variants:
        e = cls()
        e.search(from_fen(chess.STARTING_FEN), 200, 400)
        engines[v] = e

    board = game.board()
    fens = [board.fen()]
    for e in engines.values():
        e.record_game_position(from_fen(board.fen()))

    agree = {v: 0 for v in variants}
    tested = 0
    print("%-5s %-9s %s" % ("move", "carried", "  ".join("%-9s" % v.replace("carried-no-", "no-")
                                                          for v in variants[1:])))
    for mv in game.mainline_moves():
        if board.turn == us:
            fen = board.fen()
            picks = {}
            for v in variants:
                e = engines[v]
                if v == "fresh":
                    e = cls(); e.search(from_fen(chess.STARTING_FEN), 200, 400)
                    for f in fens:
                        e.record_game_position(from_fen(f))
                    engines[v] = e
                elif v.startswith("carried-no-"):
                    clear(e, GROUPS[v.replace("carried-no-", "")])
                m, _, _, *_ = e.search(from_fen(fen), args.ms, int(args.ms * 1.4))
                picks[v] = board.san(chess.Move.from_uci(m))
            if board.fullmove_number >= args.from_move:
                tested += 1
                for v in variants:
                    if picks[v] == picks["fresh"]:
                        agree[v] += 1
                if picks["carried"] != picks["fresh"]:
                    print("%-5d %-9s %s" % (board.fullmove_number, picks["carried"],
                          "  ".join("%-9s" % picks[v] for v in variants[1:])))
        board.push(mv)
        fens.append(board.fen())
        for e in engines.values():
            e.record_game_position(from_fen(board.fen()))

    print("\nagreement with a fresh engine, over %d positions:" % tested)
    for v in variants:
        print("  %-22s %3d/%-3d  %5.1f%%" % (v, agree[v], tested, 100 * agree[v] / max(1, tested)))


if __name__ == "__main__":
    main()
