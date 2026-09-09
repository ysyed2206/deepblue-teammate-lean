"""Depth or evaluation? Search each flagged position at increasing time.

If the engine changes its mind as depth grows, the game move was a horizon
failure and more search would have fixed it. If it keeps choosing the same
move at every depth, no amount of extra depth helps and the evaluation is
what is wrong. This is the diagnostic that settled the Kf2/b4 question.
"""
import chess
from deepblue.fastcore import from_fen
from deepblue.fastsearch135 import FastEngine135

CASES = [("23...f4",  "rr5k/p2b2qp/2pNp3/P1B1Ppp1/8/3P2P1/Q4P1P/2R3K1 b - - 0 23", "f5f4"),
         ("36...Rb3", "r7/3bB1k1/p1p1p3/P3P3/6N1/3P2P1/1r6/4R1K1 b - - 0 36",      "b2b3")]
e = FastEngine135(); e.search(from_fen(chess.STARTING_FEN), 200, 400)
for name, fen, played in CASES:
    b = chess.Board(fen)
    print("=== %s  (%d legal) ===" % (name, b.legal_moves.count()))
    for ms in (2000, 8000, 20000):
        mv, sc, d, n, _ = e.search(from_fen(fen), ms, int(ms*1.4))
        star = "  <-- the move we played" if mv == played else ""
        print("   %6dms  best %-7s %+6d cp  depth %2d%s"
              % (ms, b.san(chess.Move.from_uci(mv)), sc, d, star))
    after = b.copy(); after.push(chess.Move.from_uci(played))
    _, opp, d2, _, _ = e.search(from_fen(after.fen()), 20000, 28000)
    print("   played %-7s -> %+6d cp for us (depth %d)"
          % (b.san(chess.Move.from_uci(played)), -opp, d2))
    print()
