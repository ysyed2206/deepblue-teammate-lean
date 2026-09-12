# Deep Blue NNUE production push

Status: **Stage-1 H256 trained, quantised, development-evaluated, and 100k-transition validated.** No mainline integration or packaging has occurred. The original H128 run and baseline reports are preserved.

## Stage-1 data and training

Source: revision-pinned [Lichess chess-position-evaluations](https://huggingface.co/datasets/Lichess/chess-position-evaluations), CC0-1.0, revision `abb8f0b1251f89295a35b5ac801cb08a873812de`.

- Three million selected source FENs yielded 2,626,776 usable new unique positions after excluding 372,753 mate labels and 471 positions outside the 32-piece feature domain.
- New train 2,495,067; validation 39,790; development 39,335; sealed pristine 52,584.
- Training additionally includes the preserved prior 500,000-position component: **2,995,067 unique training examples per epoch**.
- The prior component is 400k quiet/100k nonquiet. New data retains its natural clean distribution; it is not relabelled as an 80/20 mixture.
- Old/new FEN overlap: zero. Deterministic exact-FEN hash splitting; 10,000 manual/reference encodings agreed exactly. Pristine has not been evaluated.
- Teacher cp orientation reuses the baseline's empirical proof: source White-relative, converted to STM-relative and clipped to +/-2000 cp. No game results are fabricated.

H256 is independent participant training of no-bucket Perspective Chess768, shared H=256 transformer, two SCReLU accumulators, STM-first linear scalar tail. It has **197,377 parameters**. Loss is probability-space MSE with scale 400 cp; CPU AdamW, batch 2,048, six threads, seed 20260911, learning rate 0.001. Global decay milestones 8 and 12 are reserved for staged continuation; Stage 1 ran four epochs at 0.001.

Training consumed 11,980,268 examples in 1,101.10 seconds (18.35 minutes), averaging **10,880.29 examples/s**. Validation loss decreased 0.0164038 -> 0.0133043; validation teacher MAE 190.20 -> **171.76 cp**. Best and resumable latest checkpoints are retained.

## Development comparison (39,335 identical FENs)

| Evaluator | Teacher MAE |
|---|---:|
| Accepted Deep Blue HCE | 221.591 cp |
| Preserved baseline H128 float | 209.995 cp |
| Stage-1 H256 float | 173.448 cp |
| Stage-1 H256 quantised | 173.444 cp |

H256 Q1 is 21.73% lower MAE than HCE and approximately 17.40% lower than the preserved H128 on this development set. This is a screening metric, **not an Elo claim**.

## Quantisation and correctness

Canonical Q1 uses int16 feature weights/bias, int32 accumulators, int16 output weights/bias, QA=255, and the donor's two truncation-toward-zero divisions. Output bias is quantised at QA*QB. Initial QB=64 narrowly missed the preferred quantisation targets, so an exact-int16 output-scale calibration on development data selected **QB=256**. QB64 artifacts/results are preserved separately; no retraining occurred.

| Float-versus-Q1 error | QB64 | Selected QB256 |
|---|---:|---:|
| Median | 3.447 cp | 1.938 cp |
| p95 | 11.202 cp | 5.734 cp |
| p99 | 15.469 cp | 7.659 cp |
| Maximum | 27.402 cp | 16.363 cp |
| Signed bias | +2.746 cp | +0.086 cp |
| Correlation | 0.999841 | 0.999943 |

All 39,335 development positions were cross-checked. Exporter/NumPy/Numba also agreed exactly on sampled actual-weight and random accumulator tests. The selected trained H256 candidate passed **100,000 legal move transitions with zero accumulator, evaluation, or unmake failures**, plus explicit special-move coverage. Both widths separately passed 50,000 synthetic-weight transitions. King moves never require a bucket refresh.

## Runtime (concurrent-training measurement)

Actual elapsed Numba hot-loop medians, 11 trials, 100,000 iterations/trial; JIT excluded. These should be repeated under low load before integration decisions.

| Operation | H256 |
|---|---:|
| Full refresh | 4.618 us |
| Quiet update | 0.174 us |
| Capture update | 0.215 us |
| Tail only | 1.424 us |
| Quiet update + tail | 1.598 us |
| Capture update + tail | 1.616 us |
| HCE | 0.152 us |

Full refresh is not the expected per-node cost. The incremental-plus-tail path is the relevant first runtime screen; later search integration and equal-time games remain necessary.

## Artifacts and active work

- Candidate: `runs/pawnstar_h256_cc0_stage1/candidate_q1.npz`, **457,066 bytes**, SHA256 `8e8771acedccf37aa8e875199320a1bce1fb8e2461331692423507859b634fd6`.
- Float/best and resumable latest: same run directory, `best.pt` and `latest.pt`.
- Exact config, training log/summary, quantisation report, development report/predictions, incremental gate, and benchmark are in that run directory.
- Fifteen million production-prefix FENs are acquired before filtering. Additive preprocessing and verified disjoint row-group-window acquisition are extending the corpus; this is not a 3M ceiling.
- The historical PlentyChess donor corpus was not used because its public dataset repository did not expose a reuse licence. Donor architecture/training deviations are recorded in `PAWNSTAR_PRODUCTION_NOTES.md`.

Final H256 selection, pristine evaluation, final model manifest, and GO/NO-GO recommendation are **pending**. Mainline remains untouched regardless of outcome.
