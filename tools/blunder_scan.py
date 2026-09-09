"""Score every one of our moves across many games, in parallel.

For each of our moves: search the position (best move, best score), then search
the position our played move actually reached and negate it. The gap is what
the move cost. Games are analysed one per worker.

The referee is our own engine at a longer time control than the game allowed.
That is a real limitation -- it is blind to anything our evaluation is blind
to, so it will under-report positional errors and over-trust its own
tactical judgement. It is reliable for what it is used for here: finding moves
that lose material or walk into something concrete.
"""
from __future__ import annotations

import glob
import multiprocessing as mp
import os
import re
import sys
from pathlib import Path

import chess
import chess.pgn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MS = 1500
MODULE = "fastsearch150"


def analyse(path: str):
    from deepblue.fastcore import from_fen
    import importlib
    mod = importlib.import_module("deepblue." + MODULE)
    engine = getattr(mod, "FastEngine" + MODULE.replace("fastsearch", ""))()
    engine.search(from_fen(chess.STARTING_FEN), 200, 400)

    game = chess.pgn.read_game(open(path, encoding="utf-8"))
    if game is None:
        return None
    h = game.headers
    us = chess.WHITE if h.get("White", "").lower().startswith("yumo") else chess.BLACK
    res = h.get("Result", "*")
    ours = "1-0" if us == chess.WHITE else "0-1"
    outcome = "win" if res == ours else ("draw" if res == "1/2-1/2" else "loss")
    rnd = re.search(r"round-(\d+)", os.path.basename(path))
    label = "r" + (rnd.group(1) if rnd else "?")

    board = game.board()
    rows = []
    for mv in game.mainline_moves():
        if board.turn == us:
            san = board.san(mv)
            n_legal = board.legal_moves.count()
            best, best_sc, _, _, _ = engine.search(from_fen(board.fen()), MS, int(MS * 1.4))
            after = board.copy(); after.push(mv)
            if after.is_game_over():
                got = 30000 if after.is_checkmate() else 0
            else:
                _, opp, _, _, _ = engine.search(from_fen(after.fen()), MS, int(MS * 1.4))
                got = -opp
            cost = best_sc - got
            rows.append((board.fullmove_number, san, best, best_sc, got, cost,
                         n_legal, board.is_check()))
        board.push(mv)
    return label, outcome, ("W" if us == chess.WHITE else "B"), rows


def main() -> None:
    paths = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1
                             else r"C:/Users/uniqu/Downloads/aichessathon-round-6[1-7]-*.pgn"))
    with mp.Pool(min(7, len(paths))) as pool:
        results = [r for r in pool.map(analyse, paths) if r]
    print("%-5s %-4s %-5s %5s %7s %7s   worst moves (cost cp)" %
          ("game", "side", "res", "moves", ">=100", ">=300"))
    allrows = []
    for label, outcome, side, rows in results:
        # A "cost" is only meaningful in a position that was still playable and
        # where we had a real choice. Two filters, both learned from round 62:
        #
        #  * DECIDED POSITIONS. Cost is the gap between two independent
        #    searches. At -1200 or worse, with mate in view, those two searches
        #    disagree by hundreds of centipawns from depth and mate-distance
        #    instability alone. Round 62 charged us 546cp for move 44 -- where
        #    the engine played the move the referee itself called best, from
        #    two legal options, in check, already lost. That is noise reported
        #    as a blunder, and it invented an "endgame king wandering" failure
        #    mode that does not exist.
        #
        #  * FORCED MOVES. One or two legal replies is not a decision.
        ordinary = [r for r in rows
                    if abs(r[5]) < 5000            # not a walk into mate
                    and abs(r[3]) < 800            # position still playable
                    and r[6] >= 3                  # a real choice existed
                    and not r[7]]                  # not in check
        bad = [r for r in ordinary if r[5] >= 100]
        awful = [r for r in ordinary if r[5] >= 300]
        worst = sorted(ordinary, key=lambda r: -r[5])[:3]
        allrows.append((label, outcome, side, rows))
        print("%-5s %-4s %-5s %5d %7d %7d   %s" %
              (label, side, outcome, len(rows), len(bad), len(awful),
               ", ".join("%d.%s %d" % (w[0], w[1], w[5]) for w in worst if w[5] >= 100)))
    import pickle
    pickle.dump(allrows, open("scratch_blunders.pkl", "wb"))
    print("\nper-move detail saved to scratch_blunders.pkl")


if __name__ == "__main__":
    mp.freeze_support()
    main()
