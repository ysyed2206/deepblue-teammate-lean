"""The fused term must be bit-identical to the two separate ones.

A speed refactor that changes results is not a speed refactor -- it is an
untested evaluation change wearing one. Checked across a wide spread of
positions including endgames, king-less-zone edge cases and heavy middlegames.
"""
import chess, random
from deepblue.fastcore import from_fen
from deepblue.eval_terms import (mobility_white_relative,
                                 king_attack_danger_strongq_white_relative,
                                 mobility_and_danger_white_relative, game_phase)
from deepblue.fastsearch import PHASE_TABLE

FENS = [
 "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
 "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
 "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
 "r2q1rk1/pp2bppp/2n1bn2/2pp4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 11",
 "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
 "2r3k1/p3rppp/Pp1q1n2/2pp1b2/Q7/2P1P2P/1P1NBPP1/R2R2K1 w - - 1 18",
 "r4rk1/1p2q2N/3p3Q/p4p2/1pP1P3/1P4P1/4PP2/1R2K2R b K - 0 23",
 "8/8/4k3/8/8/4K3/8/8 w - - 0 1",
 "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
 "1r3b1r/1pNk1ppp/pB1pbn2/4p3/3n3q/NB6/PPP2PPP/2RQ1RK1 b - - 11 15",
]
# plus random legal positions reached by random play
rng = random.Random(0)
for _ in range(140):
    b = chess.Board()
    for _ in range(rng.randint(4, 70)):
        ms = list(b.legal_moves)
        if not ms: break
        b.push(rng.choice(ms))
    if not b.is_game_over():
        FENS.append(b.fen())

bad = 0
for f in FENS:
    bb = from_fen(f)[0]
    ph = game_phase(bb, PHASE_TABLE, 24)
    m1 = int(mobility_white_relative(bb))
    d1 = int(king_attack_danger_strongq_white_relative(bb, ph, 24))
    m2, d2 = mobility_and_danger_white_relative(bb, ph, 24)
    if m1 != int(m2) or d1 != int(d2):
        bad += 1
        if bad <= 3:
            print("MISMATCH %s\n   mobility %d vs %d   danger %d vs %d" % (f, m1, int(m2), d1, int(d2)))
print("checked %d positions, %d mismatches" % (len(FENS), bad))
print("IDENTICAL" if bad == 0 else "DIFFERS -- do not use")
