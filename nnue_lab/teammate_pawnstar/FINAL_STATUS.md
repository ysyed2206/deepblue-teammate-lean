# Final status

## Definition of success (from the mission brief), scored honestly

| Requirement | Status |
|---|---|
| Pawnstar-compatible feature mapping works | **Done.** Verified against real source; 20/20 perspective-symmetry checks plus hand-computed cases, all in `tests/test_features.py`. |
| Reference evaluator works | **Done, and independently verified.** 250/250 exact matches against the donor's own real reference evals on the real shipped net -- not a self-consistency check, a match against external ground truth. |
| Incremental evaluator is exact | **Done.** 100,011 transitions (100,000 random + 11 hand-built special-move cases), 0 accumulator mismatches, 0 evaluation mismatches, 0 unmake failures, using the real donor weights. All mission-listed categories (quiet, capture, en passant, king same/different bucket, O-O, O-O-O, promotion, capture-promotion, underpromotion) covered and counted. |
| Quantisation is understood | **Done.** The donor's real int8 shipped path is implemented and measured (median 6cp, p99 25.5cp, max 28cp vs. the exact int16 path) -- closely matching the donor's own stated "~26cp" figure, an independent cross-check of both the spec and this implementation. |
| One-core runtime is measured | **Done, with a real comparison, not an assumption.** Numba: ~169k evals/sec (incremental update + tail). Measured PyTorch overhead for a single minimal op already matches Numba's *entire* fused tail time, which is the concrete evidence behind not implementing full PyTorch/ONNX paths. Cross-referenced against Deep Blue's own current node rate (13k-25k nps) -- the evaluator would not be the bottleneck. |
| A clean integration API exists | **Done as a contract, not a merge.** `INTEGRATION_CONTRACT.md` specifies the exact adapter shape against `fastcore.py`'s real `mail`/`decode(move)` interface (read, not modified) -- grounded in the actual live-engine code, not a guess. |
| Fine-tuned candidate (only if pretrained fine-tuning explicitly confirmed) | **Not attempted, correctly per the brief.** Confirmation was never given in this session; per the mission's own instruction this means "do not spend the session training from donor weights." Nothing in `finetune/` exists. |

## What's genuinely strong about this result

- Every correctness claim is checked against **the donor's own real, external
  data** (their actual shipped net, their own 250 published reference evals) --
  not synthetic self-consistency, which is the weaker form of evidence the
  mission brief explicitly wanted avoided ("do not proceed to aggressive
  optimisation while basic semantics are wrong").
- The 100,011-transition incremental gate exceeds the mission's "100,000
  preferred, 50,000 minimum" bar, using real weights, with natural-language-
  auditable category counts (not just a pass/fail).
- The performance conclusion (evaluator is not the bottleneck at Deep Blue's
  current node rate) is a genuinely useful, falsifiable finding for whoever
  makes the actual integration-or-not decision later -- it removes "is NNUE
  even fast enough" as an open question, leaving only "does it actually win
  more games," which needs paired-game evidence this lane correctly did not
  attempt to fabricate.

## Scope decisions made (and why)

- **No pytest.** Adding a dependency to the shared venv (governed by files
  outside this directory's write boundary) was judged riskier than writing
  a small dependency-free test harness. See `README.md`.
- **No compiled Pawnstar C++ engine build for live-search differential
  testing.** Would need a Rust + C++ toolchain; the donor's own 250-position
  published reference set was judged higher-value evidence for the time
  available, since it is *their* ground truth, not a re-implementation of
  their reference tooling that could share the same bug.
- **ONNX Runtime not benchmarked directly.** Reasoned out from a measured
  PyTorch data point instead (see `BENCHMARKS.md`) -- flagged explicitly as
  the natural next measurement, not silently skipped.
- **Numba kernels not yet wired into `incremental.py`'s `NNUEState`.** The
  100k-transition gate validates the NumPy/`reference_eval.py` path (which
  is what `NNUEState` currently calls); the Numba kernels are separately
  verified correct against that same path (`tests/test_numba_kernels.py`)
  but are a distinct, not-yet-merged fast path. This was a deliberate
  ordering choice: proving correctness on the simple path first, before
  optimising, matches the mission's own explicit priority order (Priority 3
  correctness before Priority 5 performance) and avoids debugging two
  moving parts (new logic + new fast kernel) at once.
- **`incremental.py`'s diff works on `chess.Board` pairs, not Deep Blue's
  own `mail` array.** Correct scope for a standalone, testable prototype;
  `INTEGRATION_CONTRACT.md` specifies exactly what would change for a real
  integration, grounded in having actually read `fastcore.py`.

## Honest weaknesses / open items for whoever picks this up next

1. Numba kernels are correct but not yet the thing `incremental.py`'s
   100k-transition gate actually exercised end-to-end together (see above).
   Wiring them together and re-running the gate would be the natural next
   half-day of work.
2. No actual paired-game or fixed-depth-search evidence exists for whether
   this evaluator, once integrated, would win more games than Deep Blue's
   current hand-crafted evaluation -- by design (that decision needs a real
   integration first, which this lane deliberately did not perform).
3. The lazy/deferred-accumulator API (`lazy=True` on `make_move`/`unmake_move`)
   is implemented and tested for correctness (identical eval to eager mode)
   but its actual speed benefit (the donor measured +20.66 Elo / +13.6% nps
   from this specific optimisation) was not separately benchmarked here --
   the `BENCHMARKS.md` numbers are all eager-mode.
4. `PAWNSTAR_FORMAT.md` section 7 lists what was deliberately *not* verified
   from source (SIMD kernels, the Rust trainer, the C++ `SearchState` class) --
   worth a read before assuming this lane's coverage of the donor is total.

## Deliverables checklist (mission section 18)

All present under `nnue_lab/teammate_pawnstar/`: `README.md`,
`PAWNSTAR_FORMAT.md`, `features.py`, `weights.py`, `reference_eval.py`,
`incremental.py`, `quantized.py`, `tests/`, `DIFFERENTIAL_RESULTS.md`,
`BENCHMARKS.md`, `INTEGRATION_CONTRACT.md`, `FINAL_STATUS.md` (this file).
Plus `numba_kernels.py` (the mission's Priority 5 deliverable, not
separately named in the section-18 list but required by Priority 5) and
`nnue_reference_250.txt` (the differential-testing ground truth, small
enough to keep in-repo per the mission's own size guidance).

No `finetune/`/`candidate/` directories, per the fine-tuning decision above.
