# Performance benchmarks

All numbers below are wall-clock, single-threaded, measured on this
development machine (see `EXPERIMENTS.md` at the repo root for its spec),
using the **real downloaded Pawnstar v12 net** (1024-wide, 8 king buckets),
not synthetic weights, unless stated otherwise. JIT compilation is excluded
(the Numba kernels use eager compilation via an explicit `@njit` signature,
so compilation happens at import time -- see "Cold load" below -- not on the
first call).

## Backend comparison (Priority 5): which one should the hot path use

The mission asks not to assume Numba wins without measuring. Measured:

| Backend | What was measured | Result |
|---|---|---:|
| **Numba njit** (`numba_kernels.py`) | Fused tail (SCReLU + dot + dequant), precomputed accumulators | **3.26 us/call** |
| Pure NumPy (`reference_eval.py`) | Same tail computation, vectorised NumPy ops | 31.57 us/call (**~9.7x slower**) |
| PyTorch (eager, 1 thread) | A *single* 1024-dim `torch.dot` call -- not even the full tail (no SCReLU, no two perspectives, no dequant) | 3.69 us/call -- **already matches Numba's entire fused tail**, before adding the 4-5 more chained ops (clamp, square x2, dot x2, dequant) a real evaluator needs |
| ONNX Runtime | Not benchmarked | See reasoning below |

**Decision: Numba is the hot-path backend.** The PyTorch number is the
concrete evidence, not an assumption: a *single* minimal op in PyTorch's
eager mode already costs as much as Numba's *entire* fused evaluator call.
A real PyTorch tail needs several chained ops (each carrying its own
Python/dispatch overhead), so a naive PyTorch implementation would land well
behind Numba's single fused, compiled function -- exactly the "framework-call
overhead dominates on tiny calls" failure mode the mission brief warned about.
ONNX Runtime was not benchmarked for the same underlying reason (its
per-inference call overhead is architecturally similar to PyTorch's C++
dispatch path for a model this small); building and exporting an ONNX graph
for a network this size, to very likely land in the same regime as the
PyTorch number above, was judged not worth the session time versus
finishing the Numba path's correctness and the incremental gate. If this
lane is picked up again, benchmarking ONNX Runtime CPU directly (rather than
inferring from the PyTorch result) would be the natural next measurement.

## Numba kernel timings (the candidate hot path)

| Operation | Time | Notes |
|---|---:|---|
| Cold load (JIT compile, all 3 kernels) | ~0.01 s | Eager-compiled via explicit `@njit` signatures -- happens once at import, not per-call; negligible against the competition's 60 s import budget |
| Full refresh, one perspective (32 active rows) | 15.98 us | Both perspectives: ~32 us |
| Quiet update (1 removed + 1 added row) | 2.665 us | |
| Capture update (2 removed + 1 added row) | 2.957 us | |
| Castling update (2 removed + 2 added rows) | 3.507 us | King + rook, both perspectives worth of columns |
| Tail only (SCReLU + dot + dequant, both perspectives) | 3.259 us | |
| Quiet update + tail (the realistic per-node cost) | 5.915 us | |

Implied throughput on one core: **~306,800 evaluations/sec** (tail only, accumulator already current) or **~169,000 evaluations/sec** (incremental update + tail, the realistic per-node figure for a quiet move).

### Why this matters for Deep Blue specifically

`ARCHITECTURE.md` (repo root) records Deep Blue's own current search rate as
**13k-25k nodes/sec** (S0) with S1 still below that at present. This NNUE
evaluator's incremental-update-plus-tail cost (~169,000 evals/sec) is **roughly
7-13x faster than Deep Blue's entire current node rate** -- meaning, if this
evaluator were ever wired into the live search, the evaluator itself would
not be the search's bottleneck; move generation, make/unmake and the legality
test (already identified as the dominant costs in `ARCHITECTURE.md`) would
remain the limiting factor. This is a significant de-risking data point for
any future integration decision: the concern "is a 1024-wide NNUE too slow
for a 1-core budget" is answered no, at least for the evaluator in isolation.

## Incremental correctness gate (also a de-facto performance/scale test)

100,011 transitions (100,000 random-game + 11 hand-built special-move cases)
processed via `incremental.py`'s Python/NumPy path (not yet the Numba
kernels above -- that integration is the natural next step, not done this
session) in under 2 minutes wall-clock, including a full-refresh cross-check
on every single transition (which is far more expensive than production use
would ever be, since production never double-checks itself). See
`DIFFERENTIAL_RESULTS.md` for the full correctness breakdown by move
category; this section is the timing note for the same run.

Reproduce the Numba benchmark:

```powershell
.venv\Scripts\python.exe nnue_lab\teammate_pawnstar\tests\test_numba_kernels.py --bench
```

## What was not benchmarked (scope notes)

- **ONNX Runtime CPU** -- reasoned out above rather than measured; flagged as
  the natural next measurement if this lane continues.
- **The Numba kernels wired into `incremental.py`'s `NNUEState`** -- the
  100k-transition correctness gate runs on the plain NumPy path (`reference_eval.py`
  functions), which is what `incremental.py` currently calls. The Numba
  kernels in `numba_kernels.py` are verified correct against that same NumPy
  path (see `tests/test_numba_kernels.py`) but are a separate, not-yet-wired
  fast path. Swapping `NNUEState`'s internals to call the Numba kernels
  instead of the NumPy reference functions is a small, low-risk follow-up
  (the row-list bookkeeping is already computed the same way for both), left
  for whoever picks up integration next -- see `INTEGRATION_CONTRACT.md`.
