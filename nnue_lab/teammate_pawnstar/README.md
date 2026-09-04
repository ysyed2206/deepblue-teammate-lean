# teammate_pawnstar

A production-lane prototype for a Pawnstar-v12-compatible NNUE evaluator
(1024-wide, 8 king buckets), built per `TEAMMATE_NNUE_MISSION.md` at the repo
root. Self-contained: nothing outside this directory was modified, and
nothing here is wired into the live engine (`agent.py`, `deepblue/`).

See `FINAL_STATUS.md` for the honest done/not-done breakdown and
`INTEGRATION_CONTRACT.md` for what a future integration into the live search
would actually need to do.

## Files

| File | What it is |
|---|---|
| `PAWNSTAR_FORMAT.md` | The donor's real architecture/format, verified against actual source (not the mission brief's placeholder numbers) |
| `features.py` | Standalone feature mapper: chess position -> donor-compatible sparse feature rows |
| `weights.py` | Loader for the donor's stamped `.bin` net format, with architecture validation |
| `reference_eval.py` | Exact int16 reference evaluator (donor's `EvaluateExact`) |
| `incremental.py` | `NNUEState`: incremental (and optional lazy) accumulator maintenance across make/unmake |
| `quantized.py` | The donor's shipped int8-output fast path (`Network::Evaluate`), and its error vs. the int16 reference |
| `numba_kernels.py` | Numba njit hot-path kernels (refresh, update, tail) -- the actual speed candidate |
| `DIFFERENTIAL_RESULTS.md` | Correctness evidence: 250/250 exact matches against the donor's own real reference evals; 100,011-transition incremental gate, 0 mismatches; int8 quantisation error stats |
| `BENCHMARKS.md` | Numba vs. NumPy vs. PyTorch timing, and what that implies for Deep Blue's current node rate |
| `INTEGRATION_CONTRACT.md` | The minimal adapter a real `fastcore.py` integration would need (not implemented) |
| `nnue_reference_250.txt` | The donor's own 250 (FEN, eval) ground-truth pairs, downloaded verbatim, used as the differential-testing oracle |
| `tests/` | All correctness tests, dependency-free (no pytest -- see below) |

## Why no pytest

The project venv (governed by `pyproject.toml`/`uv.lock`, both outside this
directory's write boundary) does not include pytest, and adding a dependency
there was judged out of scope for this lane. Every `tests/test_*.py` file is
a plain assert-based module with its own `run_all()` entry point.

```powershell
.venv\Scripts\python.exe nnue_lab\teammate_pawnstar\tests\test_features.py
.venv\Scripts\python.exe nnue_lab\teammate_pawnstar\tests\test_reference_eval.py
.venv\Scripts\python.exe nnue_lab\teammate_pawnstar\tests\test_incremental.py
.venv\Scripts\python.exe nnue_lab\teammate_pawnstar\tests\test_incremental.py --gate 100000   # slow (~2 min), full-scale gate
.venv\Scripts\python.exe nnue_lab\teammate_pawnstar\tests\test_quantized.py
.venv\Scripts\python.exe nnue_lab\teammate_pawnstar\tests\test_numba_kernels.py
.venv\Scripts\python.exe nnue_lab\teammate_pawnstar\tests\test_numba_kernels.py --bench       # timing, not a correctness check
```

## The real donor weight file

Kept **outside** the repository per the mission's own instruction (large
binary artifacts belong outside version control):

```
%LOCALAPPDATA%\Temp\deepblue-teammate-data\donor_reference\pawnstar-v12.bin
```

SHA-256 `ee551d64a06aedd4b073bea2082d0e8d6a71981270b8e31737d2b42e2616c472`,
12,589,152 bytes, downloaded directly from
`github.com/jonny-reckless/pawnstar` (GPL-family licensed -- see the licence
note in `PAWNSTAR_FORMAT.md`). Every test that needs it checks for its
presence first and skips cleanly (printing why) if it is not at that path or
at the path in the `PAWNSTAR_DONOR_NET` environment variable.

## Fine-tuning (`finetune/`)

Not attempted. The mission brief is explicit that fine-tuning pretrained
weights requires the user/organiser to first confirm the competition rule
permits it; that confirmation was not given, so per the brief's own
instruction ("If fine-tuning is NOT explicitly confirmed... do not spend
your session training from donor weights"), this lane stops at a complete,
tested, benchmarked evaluator and does not produce a trained candidate. See
`FINAL_STATUS.md`.
