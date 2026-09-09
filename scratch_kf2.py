import chess
from deepblue.fastcore import from_fen
from deepblue.fastsearch135 import FastEngine135
F = "r3kb1r/1p3ppp/p2p1n2/n2Pp3/P7/N1P2PPq/BP5P/R1BQK2R w KQkq - 1 16"
b = chess.Board(F)
e = FastEngine135(); e.search(from_fen(chess.STARTING_FEN), 200, 400)
mv, sc, d, n, ms = e.search(from_fen(F), 4000, 5600)
print("after 15...Qh3, White to move, 29 legal moves")
print("engine best : %-6s %+5d cp  (depth %d)" % (b.san(chess.Move.from_uci(mv)), sc, d))
after = b.copy(); after.push(chess.Move.from_uci("e1f2"))
mv2, sc2, d2, _, _ = e.search(from_fen(after.fen()), 4000, 5600)
print("16.Kf2 gives: %-6s %+5d cp  (depth %d)  their best reply %s"
      % ("Kf2", -sc2, d2, after.san(chess.Move.from_uci(mv2))))
print("cost of Kf2 : %d cp" % (sc - (-sc2)))
