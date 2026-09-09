"""Score every one of our moves in round 58 against a deeper search.

For each White move: search the position (best move, best score), then search
the position our played move actually reached and negate it. The gap is what
the move cost. Analyst is fastsearch126 at 5s -- roughly 2.5x the thinking
time the game itself allowed, so it stands in for a stronger referee.
"""
import chess, chess.pgn
from deepblue.fastcore import from_fen
from deepblue.fastsearch127 import FastEngine127

MS = 5000
e = FastEngine127()
e.search(from_fen(chess.STARTING_FEN), 200, 400)      # warm the JIT

def look(board):
    mv, sc, d, n, el = e.search(from_fen(board.fen()), MS, int(MS * 1.4))
    return mv, sc, d

g = chess.pgn.read_game(open(r"C:/Users/uniqu/Downloads/aichessathon-round-58-yumo-vs-highestelo.pgn"))
b = g.board()
rows = []
for mv in g.mainline_moves():
    if b.turn == chess.WHITE:
        played_san = b.san(mv)
        played_uci = mv.uci()
        best_uci, best_sc, depth = look(b)
        after = b.copy(); after.push(mv)
        if after.is_game_over():
            got = -30000 if after.is_checkmate() else 0
        else:
            _, opp_sc, _ = look(after)
            got = -opp_sc
        cost = best_sc - got
        rows.append((b.fullmove_number, played_san, played_uci, best_uci, best_sc, got, cost, depth))
    b.push(mv)

print("%-5s %-8s %-8s %8s %8s %8s" % ("move", "played", "best", "best cp", "got cp", "cost"))
for num, san, puci, buci, bsc, got, cost, d in rows:
    flag = ""
    if cost >= 300: flag = "  <== BLUNDER"
    elif cost >= 100: flag = "  <-- mistake"
    same = "(same)" if puci == buci else buci
    print("%-5d %-8s %-8s %+8d %+8d %8d%s" % (num, san, same, bsc, got, cost, flag))
big = [r for r in rows if r[6] >= 100]
print("\n%d of %d moves cost 100cp or more" % (len(big), len(rows)))
