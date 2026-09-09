"""Sliding attack cost, with loop-invariance defeated.

The first attempt XOR-ed a constant expression, so LLVM hoisted it out of the
loop and reported 0.0 ns for the knight lookup. Here the square and occupancy
both vary with the iteration, so nothing can be hoisted.
"""
import numpy as np, time
from numba import njit
from deepblue.fastcore import from_fen, bishop_attacks, rook_attacks, KNIGHT_ATTACKS

@njit(cache=False, nogil=True)
def t_bishop(occs, reps):
    s = np.uint64(0)
    for i in range(reps):
        s ^= bishop_attacks(np.int64(i & 63), occs[i & 255] ^ s)
    return s

@njit(cache=False, nogil=True)
def t_rook(occs, reps):
    s = np.uint64(0)
    for i in range(reps):
        s ^= rook_attacks(np.int64(i & 63), occs[i & 255] ^ s)
    return s

@njit(cache=False, nogil=True)
def t_knight(occs, reps):
    s = np.uint64(0)
    for i in range(reps):
        s ^= KNIGHT_ATTACKS[(i ^ np.int64(s)) & 63]
    return s

rng = np.random.default_rng(0)
occs = rng.integers(0, 2**63, size=256, dtype=np.int64).astype(np.uint64)
R = 3_000_000
for name, fn in (("bishop_attacks (4 rays)", t_bishop),
                 ("rook_attacks   (4 rays)", t_rook),
                 ("KNIGHT_ATTACKS (1 lookup)", t_knight)):
    fn(occs, 1000)
    s = time.perf_counter(); fn(occs, R); ns = (time.perf_counter()-s)/R*1e9
    print("%-26s %7.2f ns/call" % (name, ns))
