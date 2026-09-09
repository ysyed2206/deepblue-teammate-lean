"""How expensive are our sliding-piece attacks?

Every strong engine uses magic bitboards (a single multiply-shift-index into a
precomputed table). We use classical ray walking -- four ray_attacks calls per
bishop, four per rook. Sliding attacks are used by move generation, in_check,
SEE and the mobility term, so their cost is spread across most of the search.
Timed inside a njit loop so the dispatch overhead that ruined the last
profiling attempt is compiled away.
"""
import numpy as np, time
from numba import njit
from deepblue.fastcore import from_fen, bishop_attacks, rook_attacks, KNIGHT_ATTACKS

@njit(cache=False, nogil=True)
def t_bishop(sq, occ, reps):
    s = np.uint64(0)
    for _ in range(reps): s ^= bishop_attacks(sq, occ)
    return s

@njit(cache=False, nogil=True)
def t_rook(sq, occ, reps):
    s = np.uint64(0)
    for _ in range(reps): s ^= rook_attacks(sq, occ)
    return s

@njit(cache=False, nogil=True)
def t_knight(sq, reps):
    s = np.uint64(0)
    for _ in range(reps): s ^= KNIGHT_ATTACKS[sq]
    return s

bb, occ, mail, st = from_fen("r2q1rk1/pp2bppp/2n1bn2/2pp4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 11")
occupied = np.uint64(0)
for i in range(12): occupied |= bb[i]
R = 2_000_000
for name, fn, args in (("bishop_attacks (4 rays)", t_bishop, (np.int64(28), occupied)),
                       ("rook_attacks   (4 rays)", t_rook,   (np.int64(28), occupied)),
                       ("KNIGHT_ATTACKS (1 lookup)", t_knight, (np.int64(28),))):
    fn(*args, 1000)
    s = time.perf_counter(); fn(*args, R); ns = (time.perf_counter()-s)/R*1e9
    print("%-26s %7.1f ns/call" % (name, ns))
print("\nA magic-bitboard slider is one multiply, one shift and one array read --")
print("essentially the KNIGHT_ATTACKS number above.")
