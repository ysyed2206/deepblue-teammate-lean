# Isolated Deep Blue NNUE lab

This directory contains the participant-owned NNUE v0 data, training, quantisation, correctness,
and Numba-inference experiment. It deliberately does not integrate with or package the live engine.

- `RESULTS.md`: decision and exact measured results.
- `REFERENCES.md`: third-party concepts studied and the independent design.
- `INTEGRATION_PLAN.md`: minimum future integration experiment; no mainline change was made.
- `manifests/`: revision-pinned source, orientation, split, counts, and dataset hashes.
- `runs/h128_mixed_final/`: selected float checkpoint, int16 candidate, logs, predictions, and gates.
- `features.py`, `model.py`: encoder and float reference/training model.
- `download_data.py`, `verify_orientation.py`, `preprocess.py`, `train.py`: bounded data pipeline.
- `export.py`, `evaluate_models.py`: exact quantisation and held-out comparison.
- `inference_numba.py`, `test_incremental.py`, `benchmark.py`: standalone one-core prototype and gates.
- `test_features.py`: feature and symmetry unit tests.
- `FILE_MANIFEST.json`: complete file inventory with sizes and hashes.

## Frozen final training command

Run from the repository root with the already-produced external datasets:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OMP_NUM_THREADS='6'
$env:MKL_NUM_THREADS='6'
.\.venv\Scripts\python.exe -B -m nnue_lab.train `
  --train-data C:\Users\mohib\AppData\Local\Temp\deepblue-nnue-data\processed\final_mixed.npz `
  --validation-data C:\Users\mohib\AppData\Local\Temp\deepblue-nnue-data\processed\validation.npz `
  --width 128 --epochs 4 --batch-size 1024 --validation-batch-size 2048 --threads 6 `
  --seed 20260906 --patience 2 `
  --checkpoint nnue_lab\runs\h128_mixed_final\best.pt `
  --log nnue_lab\runs\h128_mixed_final\train.jsonl `
  --summary nnue_lab\runs\h128_mixed_final\summary.json
```

The exact executed command is also embedded in `summary.json`. Dataset paths and SHA-256 values are
frozen in `manifests/preprocess.json`; raw selected-file hashes are recorded there and in the five
download manifests.

## Re-run the final gates

```powershell
$data = 'C:\Users\mohib\AppData\Local\Temp\deepblue-nnue-data\processed'
$run = 'nnue_lab\runs\h128_mixed_final'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:NUMBA_CACHE_DIR=(Resolve-Path nnue_lab).Path+'\.numba_cache'
$env:NUMBA_NUM_THREADS='1'

.\.venv\Scripts\python.exe -B -m nnue_lab.export `
  --checkpoint $run\best.pt --heldout $data\test.npz `
  --output $run\candidate_int16.npz --metrics $run\quantisation.json `
  --qa 255 --qb 64 --batch-size 512 --threads 6

.\.venv\Scripts\python.exe -B -m nnue_lab.evaluate_models `
  --checkpoint $run\best.pt --quantised $run\candidate_int16.npz `
  --heldout $data\test.npz --fens $data\test_fens.jsonl `
  --report $run\heldout.json --predictions $run\heldout_predictions.npz `
  --batch-size 512 --threads 6

.\.venv\Scripts\python.exe -B -m nnue_lab.test_incremental `
  --model $run\candidate_int16.npz --fens $data\test_fens.jsonl `
  --transitions 50000 --episode-plies 128 --seed 20260907 `
  --report $run\incremental_gate.json

.\.venv\Scripts\python.exe -B -m nnue_lab.benchmark `
  --model $run\candidate_int16.npz --report $run\benchmark.json `
  --trials 7 --refresh-iterations 20000 --update-iterations 100000 `
  --tail-iterations 100000 --hce-iterations 100000
```

Use `python -B` and a lab-local `NUMBA_CACHE_DIR` whenever importing read-only Deep Blue modules so
Python/Numba cannot update caches outside this directory. Offline training may use CPU threads;
inference benchmarks intentionally use one Numba thread.

## Acquisition and preprocessing record

The source revision and exact shard URLs are in `manifests/download_0000.json` through
`download_0016.json`. Each invocation requested 160,000 unique FENs and used a 280,000,000-byte hard
cap; nonzero shards used `--discard-first-fen`. The inputs were merged with:

```powershell
.\<external-data-env>\Scripts\python.exe -B -m nnue_lab.preprocess `
  --input <shard0000-selected.jsonl> --input <shard0004-selected.jsonl> `
  --input <shard0008-selected.jsonl> --input <shard0012-selected.jsonl> `
  --input <shard0016-selected.jsonl> `
  --output-dir <external-processed-dir> `
  --orientation-proof nnue_lab\manifests\orientation_verification.json `
  --manifest nnue_lab\manifests\preprocess.json `
  --pilot-train 200000 --final-train 500000 --heldout 20000 --clip-cp 2000
```

The external data environment used PyArrow 21, requests 2.32.5, NumPy 2.3.2, and python-chess
1.11.2. It did not alter this project's dependency files. The training/runtime environment used
Python 3.12.13, CPU PyTorch 2.13.0, NumPy 2.5.2, Numba 0.67.0, and python-chess 1.11.2.
