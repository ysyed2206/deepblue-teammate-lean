import chess
from deepblue.fastcore import from_fen
from deepblue.fastsearch135 import FastEngine135
F = "r3kb1r/1p3ppp/p2p1n2/n2Pp3/P7/N1P2PPq/BP5P/R1BQK2R w KQkq - 1 16"
b = chess.Board(F)
e = FastEngine135(); e.search(from_fen(chess.STARTING_FEN), 200, 400)
print("position after 15...Qh3 (Stockfish: b4 best, we played Kf2)\n")
for ms in (2000, 6000, 15000):
    mv, sc, d, n, _ = e.search(from_fen(F), ms, int(ms*1.4))
    print("  our best @%5dms: %-6s %+5d cp  depth %2d" % (ms, b.san(chess.Move.from_uci(mv)), sc, d))
print()
for uci in ("e1f2", "b2b4"):
    after = b.copy(); after.push(chess.Move.from_uci(uci))
    _, opp, d2, _, _ = e.search(from_fen(after.fen()), 15000, 21000)
    print("  %-5s -> %+5d cp for us (depth %d)" % (b.san(chess.Move.from_uci(uci)), -opp, d2))
