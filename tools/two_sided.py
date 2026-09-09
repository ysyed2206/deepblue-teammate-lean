"""Score BOTH sides of a game through the same referee.

Point: if the opponent's moves score badly by our evaluation yet they win the
game, our evaluation is wrong about the kind of position they are creating --
which is far more informative than knowing our own moves were bad. An opponent
climbing the leaderboard by playing moves our engine considers unsound is a
direct measurement of our blind spot.

    python tools/two_sided.py <pgn> [move_ms]
"""
from __future__ import annotations

import sys
from pathlib import Path

import chess
import chess.pgn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepblue.fastcore import from_fen               # noqa: E402
from deepblue.fastsearch135 import FastEngine135     # noqa: E402


def main() -> None:
    path = sys.argv[1]
    ms = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    engine = FastEngine135()
    engine.search(from_fen(chess.STARTING_FEN), 200, 400)

    game = chess.pgn.read_game(open(path, encoding="utf-8"))
    h = game.headers
    names = {chess.WHITE: h.get("White", "White"), chess.BLACK: h.get("Black", "Black")}
    board = game.board()
    print(f"{names[chess.WHITE]} (W) vs {names[chess.BLACK]} (B)   {h.get('Result')}\n")
    print(f"{'mv':>4} {'side':<5} {'played':<8} {'best':<8} {'eval':>7} {'cost':>6}")
    totals = {chess.WHITE: [], chess.BLACK: []}
    for mv in game.mainline_moves():
        side = board.turn
        san = board.san(mv)
        best_uci, best_sc, _, _, _ = engine.search(from_fen(board.fen()), ms, int(ms * 1.4))
        best_san = board.san(chess.Move.from_uci(best_uci)) if best_uci else "-"
        after = board.copy(); after.push(mv)
        if after.is_game_over():
            got = 30000 if after.is_checkmate() else 0
        else:
            _, opp, _, _, _ = engine.search(from_fen(after.fen()), ms, int(ms * 1.4))
            got = -opp
        cost = best_sc - got
        if abs(cost) < 5000:
            totals[side].append(cost)
        flag = "  <--" if cost >= 100 else ""
        print(f"{board.fullmove_number:>4} {'W' if side else 'B':<5} {san:<8} {best_san:<8} "
              f"{got:>+7d} {cost:>6d}{flag}")
        board.push(mv)
    print()
    for side in (chess.WHITE, chess.BLACK):
        v = totals[side]
        if not v: continue
        print("%-22s moves %3d  mean cost %6.1f  errors>=100 %2d  worst %d"
              % (names[side], len(v), sum(v)/len(v),
                 sum(1 for c in v if c >= 100), max(v)))


if __name__ == "__main__":
    main()
