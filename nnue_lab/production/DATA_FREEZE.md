# Production data freeze

Frozen at `2026-09-02T14:37:45Z` from the completed Stage-1 acquisitions only. This is a
**staging corpus, not the final acquisition ceiling**. No download or network operation is
implemented by the freeze script.

## Contents

- New unique, clean, non-mate FENs after old-data exclusion: **2,626,776**.
- Production training rows: **2,495,067 new + 500,000 prior mixed = 2,995,067**.
- Validation: **39,790**.
- Development: **39,335**.
- Pristine final test: **52,584**, sealed; no model metric or target-distribution
  statistic was computed during this freeze.
- Split membership is `BLAKE2b-64(exact FEN, key=20260910) mod 10000`: 0-199 pristine,
  200-349 validation, 350-499 development, and 500-9999 train.

## Representation and target

- Shared Chess768 rows, fixed White and Black perspectives, maximum 32 pieces, padding row 768.
- Black perspective flips ranks (`square xor 56`) and swaps relative colours.
- The source `cp` is White-relative as established by the preserved v0 orientation proof.
- Training target is side-to-move relative and clipped to +/-2000 cp; mate rows are excluded.
- The prior 500k 80/20 quiet/general component was converted exactly by mapping active bucketed
  indices modulo 768 and mapping old padding row 6144 to new padding row 768.

## External artifacts

- Canonical FEN ledger: `C:\Users\mohib\AppData\Local\Temp\deepblue-nnue-data\production_frozen\frozen_positions.sqlite3`
- Converted 500k component: `C:\Users\mohib\AppData\Local\Temp\deepblue-nnue-data\production_frozen\converted_baseline_train_500k.npz`
- New train shards: **10** files.
- Validation/development/pristine arrays and FEN sidecars live alongside those shards.

All external artifacts and exact inputs are SHA-256 pinned in `data_freeze_manifest.json`.
