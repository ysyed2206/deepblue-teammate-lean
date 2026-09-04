# Deep Blue — Teammate NNUE Mission

## Goal

You are joining an active AI Chessathon project called **Deep Blue**.

Your job is **NNUE only**.

Do **not** duplicate the main search-engine work. Another Claude session is already improving search (PVS / TT / qsearch / SEE / history / LMR / pruning). Another worker is already training from-scratch H256/H512 models on the primary machine.

Your highest-value contribution is to build the **production NNUE compatibility + inference lane** that can accept the strongest allowed weights later, with a special focus on a Pawnstar-style H1024×8-bucket NNUE because it is strong, compact, and structurally much simpler than current Stockfish/Coda NNUEs.

The objective is:

**strong weights + exact feature mapping + fast one-core inference + incremental updates + quantization + drop-in integration contract**

Do not spend the session writing plans. Build and test.

---

# 0. What you receive

You will receive a ZIP of the current repository.

Extract it somewhere simple, for example:

```text
C:\Users\<YOU>\aichessathon-starter-teammate
```

Do not work from inside OneDrive if you can avoid it.

---

# 1. Install tools (Windows)

Open PowerShell.

Install `uv` if needed:

```powershell
winget install --id=astral-sh.uv -e
```

Install Claude Code:

```powershell
irm https://claude.ai/install.ps1 | iex
```

Verify:

```powershell
uv --version
claude --version
```

Enter the extracted repository:

```powershell
cd C:\Users\<YOU>\aichessathon-starter-teammate
```

Set up the project environment:

```powershell
uv sync
```

Verify Python:

```powershell
uv run python --version
```

The project targets Python 3.12.

Then start Claude Code:

```powershell
claude
```

Use Sonnet / high effort for the main autonomous session if available.

Give Claude only this message:

```text
Read TEAMMATE_NNUE_MISSION.md completely before doing anything else. Treat it as the operating brief for this session. Work autonomously for the full session. Do not ask me routine questions. Start by auditing the existing nnue_lab state, then execute the mission.
```

---

# 2. Hard write boundary

You may **READ the entire repository**.

You may **WRITE ONLY inside**:

```text
nnue_lab/teammate_pawnstar/
```

Create that directory if it does not exist.

Do not modify anything else.

In particular, do not modify:

```text
agent.py
deepblue/**
tools/**
tests/**
harness/**
champion/**
submission*.zip
Makefile
.gitignore
pyproject.toml
uv.lock
ARCHITECTURE.md
EXPERIMENTS.md
```

Do not modify the existing baseline NNUE files outside your teammate directory.

The primary machine is doing separate work. Your output must be easy to copy back without merge conflicts.

Large datasets/checkpoints must live **outside the repository**, for example:

```text
C:\Users\<YOU>\AppData\Local\Temp\deepblue-teammate-data\
```

Inside the repo, keep only code, small metadata, manifests, benchmarks, and final reasonably-sized model artifacts.

---

# 3. Competition deployment constraints

The deployed agent runs with approximately:

- Python 3.12
- 1 dedicated CPU core
- 2 GB RAM
- no GPU
- no network
- 120 seconds initial clock + 0.5 seconds increment per move
- 50 MB ZIP limit
- Numba / NumPy / python-chess / PyTorch CPU / ONNX Runtime preinstalled
- native binaries in the submission are not allowed
- Numba is the intended compiled-speed path

Therefore the production evaluator must be viable on **one CPU core**.

Do not design something that only looks good on a GPU.

---

# 4. Existing project state

The main project already has:

## Search

A working custom Numba chess engine with:

- custom bitboards
- legal move generation
- make/unmake
- Zobrist
- repetition / rule50 handling
- PVS
- TT
- qsearch
- SEE
- RFP
- search variants under active development

You do not need to work on any of this.

## Existing NNUE baseline

There is a completed participant-trained H128 baseline in `nnue_lab/`:

- 8 king buckets
- 6144 sparse rows
- H = 128
- SCReLU
- two perspectives
- scalar STM output
- 500k training positions
- ~1.58 MB quantized model
- 50,000 incremental transitions with 0 failures
- ~28.6% lower teacher MAE than the old HCE on its screening set

A separate worker has also:

- built no-bucket H256/H512 production infrastructure
- passed 50,000 exact incremental transitions for H256
- passed H512-compatible differential checks
- trained Stage-1 H256 on ~3M examples
- started H512 Stage-1
- acquired millions more CC0 Lichess positions

Do not duplicate that exact work.

---

# 5. Your mission: production Pawnstar-compatible NNUE lane

The main purpose of your machine is to answer:

> Can we implement a strong Pawnstar-style NNUE evaluator correctly and fast enough in Numba/NumPy/ONNX under the Chessathon one-core sandbox, and leave behind a clean path for allowed fine-tuned or participant-trained compatible weights?

This is primarily a **coding, compatibility, inference and performance mission**, not a giant training mission.

---

# 6. Pawnstar donor target

Mine the actual public Pawnstar repository and NNUE history.

Do not rely only on summaries.

Find the exact architecture / format / constants for the mature simple Pawnstar NNUE lineage, especially the H1024 × 8-king-bucket network.

Expected broad shape (VERIFY FROM SOURCE):

```text
8 king buckets
768 piece-square features per bucket
6144 sparse rows total

shared feature transform:
6144 -> H1024

two perspective accumulators:
white perspective
black perspective

SCReLU

concatenate:
STM perspective first
opponent perspective second

2048 -> scalar output
```

Expected broad integer constants from public notes (VERIFY):

```text
QA ≈ 255
QB ≈ 64
Scale ≈ 400
```

Expected payload shape (VERIFY):

```text
feature weights: 6144 x 1024
feature bias:    1024
output weights: 2048
output bias:     1
```

Do not trust these numbers blindly. Confirm them from actual source / history.

Write concise verified notes to:

```text
nnue_lab/teammate_pawnstar/PAWNSTAR_FORMAT.md
```

No essay.

---

# 7. Priority 1 — exact feature mapping

Build a standalone feature mapper that converts a chess position into the donor-compatible active feature indices.

Requirements:

- both perspectives
- exact square orientation
- exact colour convention
- exact piece-type indexing
- exact king-bucket mapping
- promotion pieces
- en passant does not create phantom pieces
- castling represented correctly after the move
- black-perspective mirroring exactly matches donor convention

Create:

```text
nnue_lab/teammate_pawnstar/features.py
```

and tests under your own directory.

Use existing `nnue_lab` code as reference where useful, but do not overwrite it.

---

# 8. Priority 2 — weight loader / reference evaluator

Implement a loader for the donor-compatible network format.

Create:

```text
nnue_lab/teammate_pawnstar/weights.py
nnue_lab/teammate_pawnstar/reference_eval.py
```

The reference evaluator should be simple and obviously correct, even if not maximally fast.

It must support:

- full refresh
- white accumulator
- black accumulator
- SCReLU
- STM-first concatenation
- scalar centipawn-ish output
- exact scaling / rounding

If publicly released donor weights are available, they may be used **for compatibility / differential / benchmarking research**.

Do not assume they are automatically shippable.

The user is separately clarifying whether pretrained-weight fine-tuning is allowed for final submission.

Your code should work regardless of the final ruling.

---

# 9. Priority 3 — differential correctness against donor reference

Before performance work, prove the implementation matches the donor semantics.

Use one or more of:

- Pawnstar's own public eval / bench interface
- a tiny independently built reference from documented tensors
- cross-check against donor feature indices
- known FEN/eval samples if available

Test thousands of positions, including:

- starting position
- random middlegames
- endgames
- king moves across bucket boundaries
- captures
- en passant
- both castling directions
- promotions
- underpromotions

Record:

- exact matches
- bounded rounding matches
- mismatch examples

Create:

```text
nnue_lab/teammate_pawnstar/DIFFERENTIAL_RESULTS.md
```

Do not proceed to aggressive optimization while basic semantics are wrong.

---

# 10. Priority 4 — production incremental accumulator

Implement the fast incremental path.

Target:

```text
nnue_lab/teammate_pawnstar/incremental.py
```

Maintain two H1024 accumulators.

For ordinary piece move:

```text
subtract old feature row
add new feature row
```

For capture:

```text
subtract mover old
subtract captured piece
add mover new
```

For promotion:

```text
subtract pawn old
add promoted piece new
subtract captured piece if capture-promotion
```

For castling:

update king and rook features.

For king moves:

- if the king remains in the same donor bucket, use incremental updates
- if the king changes bucket, refresh the affected perspective exactly as the donor architecture requires

Explicitly verify the other perspective's feature set behavior as well.

Run:

```text
>= 100,000 random legal transitions
```

preferred.

Absolute minimum:

```text
50,000
```

Compare incremental accumulator/eval against full recomputation.

Required result:

```text
0 accumulator mismatches
0 evaluation mismatches outside known rounding tolerance
0 make/unmake restoration failures
```

Include counts for:

- quiet
- capture
- en passant
- king move same bucket
- king move bucket change
- O-O
- O-O-O
- promotion
- capture-promotion
- underpromotion

---

# 11. Priority 5 — fastest one-core evaluator

Build and benchmark multiple viable execution paths.

Do not assume Numba wins without measurement.

Compare where practical:

## A — Numba integer evaluator

Likely primary target.

- contiguous integer weights
- preallocated accumulators
- no Python calls from hot `njit` path
- no dynamic allocation per eval
- efficient SCReLU
- integer output

## B — NumPy vectorized tail

Useful as benchmark/reference.

## C — ONNX Runtime CPU

Only if a tiny-call inference benchmark is competitive.

## D — PyTorch CPU

Only if actually competitive; do not waste time if framework-call overhead dominates.

The chess-search usage pattern is tiny incremental calls, not giant batches, so optimize for that.

Benchmark:

- cold load
- full refresh
- quiet incremental update
- capture update
- king same-bucket move
- king bucket-change move
- castling
- promotion
- tail only
- update + tail

Use enough iterations for stable hot-loop results.

Create:

```text
nnue_lab/teammate_pawnstar/BENCHMARKS.md
```

---

# 12. Priority 6 — quantization / integer path

Mine Pawnstar's actual quantization.

Determine exactly:

- feature-transformer weight dtype
- bias dtype
- accumulator dtype
- activation clipping
- output weight dtype
- output accumulator dtype
- scale / shifts
- rounding
- centipawn conversion

Implement a clean quantized format under:

```text
nnue_lab/teammate_pawnstar/quantized.py
```

If the donor uses int8 output weights or another cheap tail optimization, implement it as an isolated benchmarkable path.

Never change architecture and quantization simultaneously without a reference comparison.

Measure float/reference vs quantized:

- median absolute error
- p95
- p99
- max
- bias
- correlation

---

# 13. Priority 7 — lazy/deferred accumulator research

Pawnstar reportedly benefited from lazy/deferred accumulator work.

Mine the actual implementation/history.

If applicable to Deep Blue-style make/unmake search, prototype it inside your directory.

Do not modify the live search engine.

The output should be a standalone data structure / API that the main team can integrate later.

A good proposed API is something like:

```python
state = NNUEState.from_position(...)
state.make_move(move, board_delta)
score = state.evaluate(side_to_move)
state.unmake_move()
```

or an equivalent array-oriented Numba-compatible API.

Keep it simple enough to integrate.

---

# 14. Pretrained fine-tuning status

The user is currently clarifying the competition rule for pretrained initialization.

Therefore:

## If the user / organizer explicitly confirms fine-tuning pretrained weights is allowed

Then add a second lane:

```text
nnue_lab/teammate_pawnstar/finetune/
```

Use the donor-compatible architecture exactly so pretrained tensors fit.

Do NOT randomly redesign the network.

Fine-tune conservatively:

- preserve original donor checkpoint
- never overwrite it
- low learning rate
- frequent validation
- checkpoint often
- avoid catastrophic forgetting

Prefer currently available legitimate engine-annotated data.

A small-to-medium fine-tune is enough to test the idea.

The goal is not to retrain billions of positions.

## If fine-tuning is NOT explicitly confirmed

Do not spend your session training from donor weights.

Still complete:

- exact loader
- feature compatibility
- reference evaluator
- incremental evaluator
- quantization
- performance benchmarks

Those artifacts remain useful for architecture comparison, teacher/reference work, and potential participant-trained compatible models.

---

# 15. Do not duplicate the other worker

Do NOT spend hours recreating:

```text
no-bucket H256 Stage-1
no-bucket H512 Stage-1
the exact same 3M training corpus
```

unless you discover a clear bug in that pipeline.

The primary machine already owns that experiment.

Your role is the **strong-network production-runtime lane**.

---

# 16. Use subagents aggressively

Main Claude = coordinator / writer.

Keep parallel read-only subagents where useful:

### Agent A — Pawnstar source archaeologist
- exact format
- feature indexing
- king buckets
- quantization
- runtime history
- lazy accumulators
- public model files

### Agent B — correctness/red team
- feature mapping
- bucket transitions
- promotions
- en passant
- make/unmake
- overflow
- rounding

### Agent C — performance
- Numba layout
- vectorization
- integer tail
- memory access
- benchmark methodology

### Agent D — integration-contract designer
- define the smallest clean interface the primary Deep Blue search will need later
- do not edit live Deep Blue

Do not let two agents edit the same file concurrently.

---

# 17. Autonomous behavior

Do not ask the teammate routine questions.

If a test fails:

- inspect
- fix
- rerun

If performance is poor:

- profile
- try the next isolated optimization

If a donor detail is ambiguous:

- inspect actual source/history
- build a tiny differential
- decide from evidence

Do not sit idle while benchmarks run.

Do not repeatedly poll healthy processes.

Do not write giant planning documents.

The valuable output is code + tests + benchmarks.

---

# 18. Deliverables

Everything should live under:

```text
nnue_lab/teammate_pawnstar/
```

At minimum leave:

```text
README.md
PAWNSTAR_FORMAT.md
features.py
weights.py
reference_eval.py
incremental.py
quantized.py
tests/
DIFFERENTIAL_RESULTS.md
BENCHMARKS.md
INTEGRATION_CONTRACT.md
FINAL_STATUS.md
```

If fine-tuning is explicitly allowed and performed:

```text
finetune/
candidate/
```

with:

- config
- checkpoint
- quantized model
- SHA256
- training log
- validation result

---

# 19. Return to primary team

At the end:

1. Stop all teammate-only processes cleanly.
2. Do not delete data/checkpoints.
3. Create a ZIP containing ONLY:

```text
nnue_lab/teammate_pawnstar/
```

4. Give that ZIP back to the primary developer.

The primary developer will integrate it.

Do not attempt to merge into the live engine yourself.

---

# 20. Definition of success

Success is NOT "I researched Pawnstar."

Success is:

```text
Pawnstar-compatible feature mapping works
+
reference evaluator works
+
incremental evaluator is exact
+
quantization is understood
+
one-core runtime is measured
+
a clean integration API exists
```

and, if pretrained fine-tuning is explicitly permitted:

```text
+
a fine-tuned candidate is produced and benchmarked
```

This is the fastest non-duplicative way for this machine to contribute meaningful Elo potential to Deep Blue.

Start now.
