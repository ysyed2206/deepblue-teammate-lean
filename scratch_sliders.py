"""How much of a real search is spent inside sliding-attack generation?

Measures the per-call cost of the three slider functions (already done: ~17.6ns
each vs 2.2ns for a table lookup) and multiplies by the call count observed in
an actual search, by instrumenting a copy of the attack functions.

The ratio is what matters, so contention does not bias it -- but the ns/call
figures are timed on whatever the machine is doing right now.
"""
import numpy as np, time, chess
from numba import njit
from deepblue.fastcore import from_fen, bishop_attacks, rook_attacks, queen_attacks
from deepblue.fastsearch140 import FastEngine140, NODES, QNODES, EVAL, GEN, CHECK

# per-call cost, loop-invariance defeated
@njit(cache=False, nogil=True)
def t_b(occs, reps):
    s = np.uint64(0)
    for i in range(reps): s ^= bishop_attacks(np.int64(i & 63), occs[i & 255] ^ s)
    return s
@njit(cache=False, nogil=True)
def t_r(occs, reps):
    s = np.uint64(0)
    for i in range(reps): s ^= rook_attacks(np.int64(i & 63), occs[i & 255] ^ s)
    return s
@njit(cache=False, nogil=True)
def t_q(occs, reps):
    s = np.uint64(0)
    for i in range(reps): s ^= queen_attacks(np.int64(i & 63), occs[i & 255] ^ s)
    return s

rng = np.random.default_rng(0)
occs = rng.integers(0, 2**63, size=256, dtype=np.int64).astype(np.uint64)
R = 2_000_000
for fn, name in ((t_b, "bishop_attacks"), (t_r, "rook_attacks"), (t_q, "queen_attacks")):
    fn(occs, 1000)
    t = time.perf_counter(); fn(occs, R)
    print("%-16s %6.2f ns/call" % (name, (time.perf_counter() - t) / R * 1e9))

e = FastEngine140(); e.search(from_fen(chess.STARTING_FEN), 200, 400)
print()
print("%-10s %10s %10s %10s %10s" % ("position", "nodes", "evals", "movegen", "in_check"))
for name, f in [("midgame", "r2q1rk1/pp2bppp/2n1bn2/2pp4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 11"),
                ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")]:
    _, _, d, n, ms = e.search(from_fen(f), 3000, 4200)
    print("%-10s %10d %10d %10d %10d" % (name, int(e.counters[NODES]), int(e.counters[EVAL]),
                                          int(e.counters[GEN]), int(e.counters[CHECK])))
print("\nmobility calls sliders for every B/R/Q of BOTH sides on every evaluate();")
print("movegen and SEE call them too. Slider share is roughly:")
print("   (evals * ~10 sliders + movegen * ~8) * 17.6ns / elapsed")
