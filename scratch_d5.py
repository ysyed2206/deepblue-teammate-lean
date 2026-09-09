import chess
from deepblue.fastcore import from_fen
from deepblue.fastsearch140 import FastEngine140
F = "1r3b1r/1pNk1ppp/pB1pbn2/4p3/3n3q/NB6/PPP2PPP/2RQ1RK1 b - - 11 15"
b = chess.Board(F)
e = FastEngine140(); e.search(from_fen(chess.STARTING_FEN), 200, 400)
print("15...d5 (Stockfish: blunder).  Our king on d7, uncastled; White Nc7 + Bb6.")
print("%d legal moves\n" % b.legal_moves.count())
for ms in (2000, 6000, 15000):
    mv, sc, d, n, _ = e.search(from_fen(F), ms, int(ms*1.4))
    tag = "   <-- what we played" if mv == "d6d5" else ""
    print("  %6dms  best %-7s %+6d cp  depth %2d%s" % (ms, b.san(chess.Move.from_uci(mv)), sc, d, tag))
after = b.copy(); after.push(chess.Move.from_uci("d6d5"))
_, opp, d2, _, _ = e.search(from_fen(after.fen()), 15000, 21000)
print("\n  d5 gives  %+d cp for us (depth %d)" % (-opp, d2))
