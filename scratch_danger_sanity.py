"""Does the extended zone stay sane in ordinary positions, and is it symmetric?

The failure mode to rule out: a term that fires on every position rather than
on real attacks distorts the whole evaluation. Quiet and symmetric positions
must still score near zero.
"""
import chess
from deepblue.fastcore import from_fen
from deepblue.eval_terms import (king_attack_danger_white_relative,
                                 king_attack_danger_v2_white_relative, game_phase)
from deepblue.fastsearch118 import PHASE_TABLE

QUIET = [
 ("start",        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
 ("italian",      "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
 ("both castled", "r4rk1/pppppppp/8/8/8/8/PPPPPPPP/R4RK1 w - - 0 1"),
 ("closed centre","r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1B3/PPP1BPPP/R2Q1RK1 w - - 0 9"),
 ("endgame R+P",  "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"),
 ("kiwipete",     "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
 ("symmetric mg", "r2q1rk1/pp2bppp/2n1bn2/2pp4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 11"),
]
print("%-14s %8s %8s   %s" % ("position", "old", "v2", "mirror sum (must be 0)"))
bad = 0
for name, f in QUIET:
    bb = from_fen(f)[0]
    ph = game_phase(bb, PHASE_TABLE, 24)
    a = int(king_attack_danger_white_relative(bb, ph, 24))
    v = int(king_attack_danger_v2_white_relative(bb, ph, 24))
    mf = chess.Board(f).mirror().fen()
    mbb = from_fen(mf)[0]
    mph = game_phase(mbb, PHASE_TABLE, 24)
    mv = int(king_attack_danger_v2_white_relative(mbb, mph, 24))
    s = v + mv
    if s != 0: bad += 1
    print("%-14s %8d %8d   %+d%s" % (name, a, v, s, "  <-- ASYMMETRIC" if s else ""))
print("\nasymmetric positions: %d of %d" % (bad, len(QUIET)))
