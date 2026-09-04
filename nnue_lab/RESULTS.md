# Deep Blue NNUE v0 results

Experiment completed: 2026-09-02. This is an isolated screening experiment. It made no mainline
engine change, packages nothing, and makes no Elo claim.

## Decision

**GO to a later, controlled search-integration and arena experiment with the final H=128 model.**
All pre-integration gates passed. This is not yet a recommendation to ship the model: teacher
agreement is a screening metric, and only an integrated one-core search benchmark plus games can
establish whether the added evaluation cost buys Elo.

| Gate | Result | Decision |
|---|---:|---|
| Feature and perspective tests | 7/7 pass | PASS |
| Incremental transitions | 50,000; 0 accumulator/evaluation/unmake failures | PASS |
| Quantisation error vs float | median 2.23 cp; p99 10.39 cp | PASS |
| Held-out teacher MAE | HCE 196.80; quantised NNUE 140.55 cp | PASS: 28.58% lower |
| Ordinary update + ready tail | 827.76 ns median | PASS, subject to integrated search measurement |
| Quantised model | 1,577,046 bytes | PASS: far below 50 MB |
| Runtime external dependency | none: no engine, network, or training data | PASS |

## Data

- Source: [Lichess/chess-position-evaluations](https://huggingface.co/datasets/Lichess/chess-position-evaluations),
  revision `abb8f0b1251f89295a35b5ac801cb08a873812de`, licensed **CC0-1.0**.
- Acquisition: bounded Parquet range reads from shards 0000, 0004, 0008, 0012, and 0016; 160,000
  selected FENs per shard. The five real acquisitions downloaded 393,033,077 bytes, scanned
  2,546,465 flattened rows, and selected 800,000 unique FENs. Cross-input deduplication found no
  duplicate FENs or quality replacements.
- Selection per FEN: maximum depth, then maximum `knodes`, then the first PV row within that
  analysis/MultiPV group. The acquisition audit found the first row was the side-to-move-best PV in
  99.957% to 99.992% of checked groups, depending on shard. An incomplete leading FEN group was
  discarded from each nonzero shard.
- The successful 2,000-FEN acquisition smoke test downloaded an additional 71,374,657 bytes. One
  earlier aborted smoke attempt did not expose exact transferred bytes; its hard cap makes the
  conservative total experiment-download upper bound 564,407,734 bytes, still well below the cap.
- Large data, the isolated data environment, and the orientation-only Stockfish binary are outside
  the repository under `C:\Users\mohib\AppData\Local\Temp\deepblue-nnue-data\` (and the sibling
  `deepblue-nnue-dataenv-20260901` environment). Only small manifests remain here.

After mate and invalid-position rejection, 698,573 clean unique FENs remained:

| Split | Count | Quiet | General/nonquiet |
|---|---:|---:|---:|
| Training pool | 663,682 | 440,732 | 222,950 |
| Validation | 17,450 | 11,489 | 5,961 |
| Test | 17,441 | 11,469 | 5,972 |
| Total | 698,573 | 463,690 | 234,883 |

The split is `BLAKE2b-64(FEN, key=20260901)` with 95%/2.5%/2.5% buckets. Sampling order is a
separate deterministic hash keyed by 20260902, so a duplicate FEN cannot cross splits. The final
training set contains 500,000 unique positions: exactly 400,000 quiet and 100,000 nonquiet.

Clean-data accounting:

- 95,189 mate-labelled rows rejected.
- 6,238 invalid board-status rows rejected.
- 7,864 accepted targets clipped at the symmetric +/-2,000 cp boundary.
- Quiet candidates require: not in check; legal PV first move; no capture; no promotion; and no
  checking first move. Overlapping disqualifiers were 44,110 in-check, 171,446 capture, 1,157
  promotion, and 46,176 checking-move cases.

## Target-orientation verification

The official [Lichess CloudEval schema](https://raw.githubusercontent.com/lichess-org/api/master/doc/specs/schemas/CloudEval.yaml)
defines `cp` as White-relative, but this was also checked empirically before preprocessing. A
balanced sample of 40 non-mate, high-signal FENs (20 White to move, 20 Black to move) was reanalysed
with Stockfish 18 at depth 16:

- dataset sign vs Stockfish White POV: 40/40 agreement; Pearson correlation 0.968853;
- dataset sign vs Stockfish side-to-move POV: 20/40 agreement; correlation -0.031419.

Therefore the target is `clip(cp, -2000, 2000)` for White to move and its negation for Black to
move. Mate-labelled rows are excluded. Training minimises MSE between `sigmoid(pred_cp / 400)` and
`sigmoid(target_cp / 400)`; no result blending is used.

## Architecture and pilots

The independent model has 6,144 sparse rows (`8 king buckets * 2 colours * 6 types * 64 squares`),
one shared H-wide feature transformer and bias, fixed White/Black accumulators, SCReLU, side-to-move
accumulator first, and one `2H -> 1` scalar tail at a 400 cp scale. There are no threats, pawn pairs,
Finny tables, material heads, output buckets, or deeper layers.

| Run | Training data | H | Epochs | Wall | Mean train rate | Validation loss / MAE | Test float / quant MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| General pilot | 200,000 natural-clean | 128 | 3 | 62.35 s | 10,018/s | 0.013576 / 162.10 cp | 164.56 / 164.56 cp |
| General pilot | 200,000 natural-clean | 256 | 3 | 78.94 s | 7,871/s | 0.013102 / 159.43 cp | 160.67 / 160.74 cp |
| Mixed pilot | 160k quiet + 40k nonquiet | 128 | 3 | 62.22 s | 9,982/s | 0.013582 / 160.94 cp | 162.59 / 162.71 cp |
| Final | 400k quiet + 100k nonquiet | 128 | 4 | 204.59 s | 9,943/s | 0.010186 / 139.34 cp | 140.55 / 140.55 cp |

The quiet-heavy H=128 pilot improved test float MAE by 1.20% over the general H=128 pilot, a modest
but held-out improvement, so the 80/20 mix was selected. On the same general pilot, H=256 improved
float MAE only 2.36% over H=128 while its measured ordinary-update-plus-tail path rose from about
0.87 us to 1.54 us (77.7% slower). H=128 was retained and H=512 was not justified.

Parameter and stored-model sizes:

| Width | Parameters | Float parameter bytes | Checkpoint bytes | Quantised candidate bytes |
|---|---:|---:|---:|---:|
| 128 | 786,817 | 3,147,268 | 3,149,823 | 1,577,046 |
| 256 | 1,573,633 | 6,294,532 | 6,297,087 | 3,150,678 |

The final fit used CPU PyTorch 2.13.0, six training threads, seed 20260906, batch size 1,024, and
four requested/completed epochs. Every epoch improved validation loss; epoch 4 is frozen.

## Held-out comparison

All values below use the exact same fixed 17,441-FEN test set, which was never used for gradient
training. It was, however, inspected during the H=128/H=256 and general/mixed pilot comparison, and
that evidence influenced the final choice. The final number is therefore a post-selection screening
estimate, not a pristine never-consulted test result; a fresh external set or games are needed for an
unbiased confirmation. The HCE is the accepted compiled baseline
`deepblue.fastsearch.evaluate(bb, st, MG_TABLE, EG_TABLE, PHASE_TABLE)`, which returns side-to-move
cp; its source SHA-256 at evaluation was
`0c488363957fbba35b8a79feac442ee0bef13ced6c17414028a4046ba58527fc`.

| Evaluator | MAE | Median abs. error | p95 | p99 |
|---|---:|---:|---:|---:|
| Deep Blue HCE | 196.7971 cp | 102.0 | 665.0 | 1,485.0 |
| Final float NNUE | 140.5515 cp | 77.79 | 453.30 | 1,216.71 |
| Final int16 NNUE | 140.5468 cp | 77.0 | 453.0 | 1,216.0 |

The final quantised NNUE has **28.5829% lower MAE than HCE**. By subset, HCE/quantised-NNUE MAE is
151.97/118.31 cp on 11,469 quiet positions and 282.89/183.25 cp on 5,972 nonquiet positions. This
is teacher agreement, not an Elo estimate.

## Quantisation

The self-describing NPZ uses int16 transformer weights/bias and output weights/bias, `QA=255`,
`QB=64`, and int32 accumulators. Its exact int64 tail is:

`A = clip(round(QA*b) + sum(round(QA*W)), 0, QA)`

`cp = round_symmetric(400 * (round(QB*c)*QA^2 + sum(round(QB*v)*A^2)) / (QB*QA^2))`

Cross-check over all 17,441 test FENs:

| Absolute integer-vs-float error | cp |
|---|---:|
| Median | 2.2289 |
| Mean | 2.7685 |
| p95 | 7.1718 |
| p99 | 10.3920 |
| Maximum | 21.6073 |

## Incremental correctness

Seed 20260907 exercised 50,000 legal move transitions from held-out roots. Each transition compared
incremental accumulators with a full post-move refresh exactly, compared integer evaluations, then
unmade and verified restoration. Result: **0 accumulator mismatches, 0 evaluation mismatches, 0
unmake mismatches, 0 failures**.

The run included 40,418 quiet moves, 4,984 captures, 6 en-passant moves, 359 promotions, 26 promotion
captures, 19 castlings, and 4,472 king-bucket refreshes. Explicit deterministic cases additionally
covered both colours, underpromotion, all four castlings, and rank/file king-bucket boundaries.

## One-core Numba microbenchmarks

Median of seven trials on the local Windows/Python 3.12.13 environment with Numba 0.67.0 and one
Numba thread:

| Operation | Median |
|---|---:|
| Full refresh, both perspectives | 2,549.12 ns |
| Full refresh, one perspective | 914.88 ns |
| Ordinary incremental update | 216.26 ns |
| Capture incremental update | 278.48 ns |
| King-bucket update including one refresh | 430.23 ns |
| Evaluation tail from ready accumulators | 611.49 ns |
| Accepted HCE evaluation | 84.39 ns |
| **Ordinary update + tail** | **827.76 ns** |

The key path is 9.81 times the isolated HCE call but remains below one microsecond. Full refresh is
reported separately and is not the intended ordinary-node path. The benchmark repeatedly exercises
one fixed hot position/move; it does not include integration bookkeeping, realistic cache traffic,
search-node distribution, or any NPS/depth effect.

## Candidate artifacts

- Float checkpoint: `runs/h128_mixed_final/best.pt`, 3,149,823 bytes, SHA-256
  `799bd0e83e7b2ddc0e7e8e4b77b344f79f3f0b14be44464a65e470d6fc599426`.
- Quantised candidate: `runs/h128_mixed_final/candidate_int16.npz`, 1,577,046 bytes, SHA-256
  `65b291ca2bef0e4bff6978d889c46df76879d7a4ecf1ce9ee0d679ca58232067`.
- Training record: `runs/h128_mixed_final/summary.json` and `train.jsonl` contain the exact command,
  seed, epoch curve, throughput, and checkpoint metadata.
- Evaluation evidence: `quantisation.json`, `heldout.json`, `heldout_predictions.npz`,
  `incremental_gate.json`, and `benchmark.json` in the same run directory.

The model was initialised and trained by the participant pipeline in this lab. Stockfish supplied
offline teacher labels and an orientation check only; no Stockfish executable or third-party
network is included in the candidate. See `FILE_MANIFEST.json` for every lab file and SHA-256.

## What remains unknown

The experiment establishes a strong teacher-agreement candidate that clears correctness,
quantisation, size, and isolated-runtime gates. It does not establish Elo. The later experiment in
`INTEGRATION_PLAN.md` must measure end-to-end one-core search throughput, completed depth and arena
results against the then-current champion before any shipping decision.
