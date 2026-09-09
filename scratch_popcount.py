"""Kernighan loop vs SWAR popcount. Ratio measured back to back."""
import numpy as np, time
from numba import njit

@njit(cache=False, nogil=True)
def pc_loop(v):
    c = 0
    while v:
        v &= v - np.uint64(1)
        c += 1
    return c

M1 = np.uint64(0x5555555555555555); M2 = np.uint64(0x3333333333333333)
M4 = np.uint64(0x0f0f0f0f0f0f0f0f); H1 = np.uint64(0x0101010101010101)

@njit(cache=False, nogil=True)
def pc_swar(v):
    v = v - ((v >> np.uint64(1)) & M1)
    v = (v & M2) + ((v >> np.uint64(2)) & M2)
    v = (v + (v >> np.uint64(4))) & M4
    return np.int64((v * H1) >> np.uint64(56))

@njit(cache=False, nogil=True)
def bench_loop(vals, reps):
    s = 0
    for i in range(reps):
        s += pc_loop(vals[i & 1023])
    return s

@njit(cache=False, nogil=True)
def bench_swar(vals, reps):
    s = 0
    for i in range(reps):
        s += pc_swar(vals[i & 1023])
    return s

rng = np.random.default_rng(0)
# Typical mobility masks: 8-24 bits set, not full-board occupancy.
vals = np.zeros(1024, dtype=np.uint64)
for i in range(1024):
    bits = rng.choice(64, size=int(rng.integers(8, 24)), replace=False)
    v = 0
    for b in bits: v |= 1 << int(b)
    vals[i] = np.uint64(v)
R = 5_000_000
bench_loop(vals, 1000); bench_swar(vals, 1000)
a = bench_loop(vals, 1000); b = bench_swar(vals, 1000)
assert a == b, (a, b)
t = time.perf_counter(); bench_loop(vals, R); tl = time.perf_counter() - t
t = time.perf_counter(); bench_swar(vals, R); ts = time.perf_counter() - t
print("kernighan loop : %6.2f ns/call" % (tl / R * 1e9))
print("SWAR           : %6.2f ns/call" % (ts / R * 1e9))
print("speedup        : %.2fx   (identical results verified)" % (tl / ts))
