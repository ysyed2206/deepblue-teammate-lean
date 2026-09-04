DEEP BLUE PATCH 01 — baseline correctness/performance fixes

Overwrite these files in the repo:
  deepblue/fastcore.py
  deepblue/zobrist.py
  deepblue/fastsearch1.py

What this patch does:
- adds a qsearch-only pseudo-tactical generator (captures + promotions)
- canonicalises repetition Zobrist EP state: only a LEGAL EP capture affects the key
- fixes real-root/game-history double counting in repetition logic
- removes per-qnode temporary ordering allocations
- preserves search1 as the C6 baseline; no speculative pruning feature is added

After extracting at repo root, run:
  uv run python tools/tactical_diff.py --positions 5000
  uv run python tools/hash_invariants.py --positions 5000
  uv run python tools/regression.py --engine fast
  uv run python tools/fastcore_invariants.py

Expected minimum for first two:
  tactical differential: 0 failures / 5,000 positions
  hash targeted EP: 3 passes
  incremental/full + unmake: 0 failures

Do not promote a later search variant until these are green.
