"""Is pawn_structure_key the cost, or is it the correction changing the tree?

132 searches 2.7x fewer nodes per second than 131. Two candidates: the pawn
hash recomputed at every node, or the corrected eval reshaping the tree. This
times the hash directly, inside a njit loop with the position varying so
nothing can be hoisted.
"""
import numpy as np, time
from numba import njit
from deepblue.fastcore import from_fen
from deepblue.fastsearch132 import pawn_structure_key, corrected_eval, update_corr_hist
from deepblue import zobrist as Z

@njit(cache=False, nogil=True)
def t_key(bbs, keys, reps):
    s = np.uint64(0)
    for i in range(reps):
        s ^= pawn_structure_key(bbs[i & 7], keys)
    return s

fens = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "2r3k1/p3rppp/Pp1q1n2/2pp1b2/Q7/2P1P2P/1P1NBPP1/R2R2K1 w - - 1 18",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r2q1rk1/pp2bppp/2n1bn2/2pp4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 11",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r1b1r1k1/pp1n1p1p/4pbp1/6NP/2Pp4/3Q2P1/q2B1PB1/3RR1K1 w - - 0 18",
        "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"]
bbs = np.stack([from_fen(f)[0] for f in fens])
R = 2_000_000
t_key(bbs, Z.PIECE_KEYS, 1000)
s = time.perf_counter(); t_key(bbs, Z.PIECE_KEYS, R)
ns = (time.perf_counter()-s)/R*1e9
print("pawn_structure_key: %.1f ns/call" % ns)
print("evaluate() measured earlier at 540 ns/call, a node at ~2400 ns.")
print("=> hash is %.1f%% of a node" % (100*ns/2400))
