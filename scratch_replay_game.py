"""Replay a real game the way agent.py actually plays it, and compare.

WHY. Every single-position test in this session -- all the g4 diagnostics --
created an engine, searched one position, and read off the move. That is NOT
how the engine plays. agent.py builds ONE FastEngine at import and reuses it
for every move of the game, so by move 19 its transposition table, history
tables, killers and continuation history all carry state accumulated from
moves 9-18. A cold search of move 19 is a different computation.

This matters concretely: round 58's move 19 played Nc4 (cost 170cp), while
cold reruns of that exact position at 2s, 8s and 20s ALL prefer fxe3. Either
the accumulated state changes the choice, or the divergence is something else
-- but a cold test cannot tell us which.

Second flaw this fixes: reusing one engine across several time controls on the
SAME position lets the earlier, shorter search seed the table for the later
one. The 3s and 5s numbers quoted earlier today were contaminated that way.
Here each move gets exactly one search, in game order, as in a real game.

Usage:  python scratch_replay_game.py <module> <pgn> [move_ms]
"""
import sys, chess, chess.pgn
from deepblue.fastcore import from_fen

module_name = sys.argv[1] if len(sys.argv) > 1 else "fastsearch118"
pgn_path = sys.argv[2] if len(sys.argv) > 2 else \
    r"C:/Users/uniqu/Downloads/aichessathon-round-58-yumo-vs-highestelo.pgn"
MS = int(sys.argv[3]) if len(sys.argv) > 3 else 2000

import importlib
mod = importlib.import_module("deepblue." + module_name)
engine = getattr(mod, "FastEngine" + module_name.replace("fastsearch", ""))()
engine.search(from_fen(chess.STARTING_FEN), 200, 400)      # JIT only

g = chess.pgn.read_game(open(pgn_path))
board = g.board()
print("replaying %s with %s at %dms/move (one engine, state carried)\n"
      % (pgn_path.split("/")[-1], module_name, MS))
print("%-5s %-8s %-8s %8s" % ("move", "game", "replay", "score"))
diffs = 0
for mv in g.mainline_moves():
    if board.turn == chess.WHITE:
        played = board.san(mv)
        got, sc, d, n, el = engine.search(from_fen(board.fen()), MS, int(MS * 1.4))
        replay_san = board.san(chess.Move.from_uci(got)) if got else "-"
        mark = "" if replay_san == played else "   <-- differs"
        if mark: diffs += 1
        print("%-5d %-8s %-8s %+8d%s" % (board.fullmove_number, played, replay_san, sc, mark))
    engine.record_game_position(from_fen(board.fen()))
    board.push(mv)
print("\n%d of our moves differ from the game" % diffs)
