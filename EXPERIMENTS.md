# Deep Blue - experiment log

Every meaningful change is recorded here with numbers. Never "seems faster",
never "looks stronger". A change is KEEP, REJECT or INCONCLUSIVE, and
INCONCLUSIVE means the measurement was not strong enough to decide.

Three node-rate numbers exist and are never quoted for one another:
**movegen/s** (legal move generation alone), **perft NPS** (tree enumeration, no
evaluation) and **search NPS** (the real search including evaluation, ordering,
transposition probes and quiescence). Only search NPS bounds playing strength.

---

## Environment

| | |
|---|---|
| Development machine | Intel Core i7-13620H, 10 physical cores / 16 logical |
| Development OS | Windows 11; measurements run in a Linux VM on the same machine |
| Python | 3.12.13 (competition target: 3.12) |
| chess | 1.11.2 |
| numba | 0.67.0 |
| numpy | 2.5.2 |
| Competition runtime | 1 core, 2 GB RAM, no network, no GPU, 120 s + 0.5 s/move, 60 s init |
| Submission limit | 50 MB unzipped, `agent.py` at zip root |

All measurements below were taken on this machine unless stated otherwise.
Numbers quoted from any other environment are marked as priors and were
re-measured before being relied on.

---

## EXP-000 - Time zero: the untouched starter

**Date** 2026-08-31 · **Commit** `241c16b` (branch `pristine-starter`)

The starter `agent.py` is a uniformly random legal mover.

```
uv run python -m harness.arena --opponent baselines/greedy --games 20
+0 =3 -17, score 7.5%
terminations: checkmate 17, threefold repetition 1, stalemate 2
```

Packaging: `submission.zip`, 781 bytes compressed, 1,255 bytes unzipped,
`agent.py` at root.

**Conclusion** Baseline recorded. Everything after this is measured against it.

---

## EXP-001 - S0: the python-chess reference engine

**Date** 2026-08-31 · **Branch** `deep-blue`

**Hypothesis** A correct negamax engine on `python-chess` - iterative deepening,
alpha-beta, transposition table, quiescence, MVV-LVA, killers, history, tapered
evaluation, strict clock management - beats every supplied baseline decisively
and is safe to ship as the first champion.

**Change** New package `deepblue/` with `constants.py` (generated piece-square
tables), `evaluation.py` (tapered evaluation), `time_manager.py` (clock
allocation), `reference.py` (the search). `agent.py` rewritten as a wrapper with
a known-legal fallback chosen before the search starts.

### Correctness results

| Check | Result |
|---|---|
| Passed-pawn masks vs brute force, 20,000 random squares | 0 mismatches |
| Evaluation colour symmetry, 595 random positions, mobility off | 0 mismatches |
| Evaluation colour symmetry, 595 random positions, mobility on | 0 mismatches |
| Clock allocation invariants, 6,600 (clock, move) combinations | 0 violations |
| Fresh-process torture: 5 processes x 14 edge positions, from the packaged zip | 0 problems |
| Illegal moves observed across all arena games and torture runs | 0 |

### Speed results

`tools/benchmark.py --movetime-ms 2000 --repeats 3`, median of 3:

| Position | movegen/s | **search NPS** | depth reached | nodes |
|---|---:|---:|---:|---:|
| opening | 21,600 | 25,428 | 5 | 21,540 |
| open middlegame (Kiwipete) | 9,600 | 13,353 | 1 | 9,058 |
| closed middlegame | 12,000 | 17,623 | 3 | 8,992 |
| tactical | 28,000 | 12,935 | 3 | 13,522 |
| endgame | 27,200 | 20,293 | 6 | 20,708 |
| **median** | | **17,623** | | |

Evaluation cost, measured separately: 28.9 us/call at the start position,
16.2 us in an endgame, with mobility disabled. Cold import of the packaged
submission: 92-107 ms against a 60,000 ms budget.

### Game results

`tools/arena_parallel.py`, concurrency 8, 5,000 ms + 100 ms unless noted:

| Opponent | Games | Result | Score | Our failures |
|---|---:|---|---:|---|
| `baselines/greedy` | 24 | +24 =0 -0 | 100.0% | none |
| `baselines/minimax` | 24 | +24 =0 -0 | 100.0% | none |
| `baselines/numba` | 24 | +24 =0 -0 | 100.0% | none |

Note: 21 of 24 wins against `minimax` and 22 of 24 against `numba` came by the
*opponent* flagging, because neither baseline has clock management and both
exceed a 5 s budget. These results confirm Deep Blue is stronger and does not
fail; they are not a measurement of how much stronger.

Clock discipline, full self-play game at 10,000 ms + 500 ms
(`tools/clock_sim.py`): 44 plies, per move min 150 ms / mean 691 ms / max
899 ms, **no flag**, both clocks positive at the end.

### Bugs found and fixed during this experiment

1. **Repetition test inverted.** The search counted the node it had just been
   given as a prior occurrence, so every root move returned an immediate draw
   score. Symptom: depth 64 reached in 29 ms, 1,280 nodes, score 0, arbitrary
   move. Fix: the current node's own occurrence is included in the count, so the
   test is `>= 2`, not `>= 1`.
2. **Evaluation was not colour-symmetric.** Floor division on the tapered blend
   rounds toward negative infinity, so a position and its mirror differed by one
   centipawn on ~40% of positions (160 of 397). Fix: truncate toward zero.
   Verified 0 mismatches over 595 positions.
3. **Passed-pawn masks included the pawn's own rank.** An enemy pawn abreast on
   an adjacent file cannot stop a passer, but the mask counted it, so passers
   were under-detected. Fix: mask covers strictly-ahead ranks only. Verified
   against brute force, 0 mismatches.
4. **Time management overshot badly.** A pass begun just under the soft limit
   ran on to the hard limit: measured 5-9 s per move against a 3.2 s budget.
   Fix: measure iteration-to-iteration growth as the search runs and refuse to
   start a pass predicted not to finish; hard limit tightened from 3.0x soft to
   1.4x, which is safe because the last completed pass is always retained.
5. **Packaging would have shipped a broken zip.** `harness/package.py` globs
   only top-level `*.py` plus named includes, so `deepblue/` was omitted and the
   submission imported fine in the repo and would have died on the platform.
   Fix: `make zip` passes `--include deepblue`, and `tools/fresh_process_test.py`
   now builds the zip and runs from the *extracted* copy so this can never
   regress silently.
6. **Arena mis-attributed failures.** The parallel arena counted any failing
   termination as ours, so a 24-0 win reported "21 agent failures". Fix: a
   failing termination is attributed to the side that lost.

**Conclusion: KEEP.** S0 meets its gate. Packaged submission is 12,623 bytes
compressed / 38,081 bytes unzipped, well inside the 50 MB limit. Frozen as the
champion in `champion/`.

**Measured bottleneck** `python-chess` move generation. At Kiwipete the engine
generates 9,600 positions/s and searches 13,353 nodes/s, reaching depth 1 in
two seconds. This is not a search bug - it is the library's node cost, and it is
the entire justification for S1.

---

## EXP-003 - S1 core: our own bitboard representation

**Date** 2026-08-31 · **Branch** `deep-blue`

**Hypothesis** A board representation, move generator and make/unmake written
to stay inside Numba nopython mode reaches >= 1M perft NPS and is far faster
than python-chess at the same work.

**Change** New module `deepblue/fastcore.py`. Twelve `uint64` piece bitboards
plus a 64-entry mailbox, occupancy maintained incrementally, moves packed into
a single `uint32`, preallocated move and undo stacks indexed by ply. Sliding
attacks use the classical ray method - a precomputed ray per direction per
square, masked by occupancy, with the ray beyond the first blocker removed -
chosen because it is obviously correct on inspection. Legality is decided by
making each pseudo-legal move and testing the mover's king.

### Correctness results - the gate

| Check | Result |
|---|---|
| `is_attacked` vs python-chess, 400 positions x 64 squares x 2 colours | **0 mismatches / 51,200** |
| `in_check` vs python-chess, 2,000 random positions | **0 mismatches** |
| Perft, 6 standard positions to depth 5/4, vs published counts | **0 failures, 16,564,718 nodes** |
| Published counts independently re-derived with python-chess | agree on every cross-checked depth |
| Differential legal-move sets vs python-chess, incl. 14 hand-picked stress FENs | **0 mismatches / 30,000 positions** |
| make/unmake exact state restoration (bb, occ, mailbox, st) | **0 corruptions / 148,237 pairs** |
| Occupancy = union of piece bitboards, mailbox agrees with bitboards | **0 violations / 3,000 positions** |

The stress corpus deliberately includes en-passant captures that expose the
king along a rank, en-passant with a pinned pawn, castling with rights partially
lost, castling through attacked squares, promotion races on both wings and
sparse endings. The random walk is biased 45% toward captures and promotions so
it reaches thin positions rather than wandering in the middlegame.

### Speed results - and a correction

**perft NPS: 1,822,163** (median across the suite). The >= 1M target is met.

That number is **perft NPS and nothing else**. It is not search NPS and it is
not a claim about playing strength.

Like-for-like against python-chess, same positions, same depths, same machine:

| Position | Depth | Nodes | python-chess | S1 | Speedup |
|---|---:|---:|---:|---:|---:|
| startpos | 4 | 197,281 | 284,642 | 1,732,013 | **6.1x** |
| kiwipete | 3 | 97,862 | 370,929 | 1,967,930 | **5.3x** |
| endgame | 4 | 43,238 | 231,482 | 1,481,091 | **6.4x** |
| aggregate | | | | | **5.9x** |

**This contradicts the prior hypothesis.** The project plan argued for a 40-100x
core speedup. The measured figure is **5.9x**, and the plan's claim should be
treated as refuted until a further change earns it back. Two things explain the
gap: python-chess perft is much faster than python-chess *search* (285k vs the
17.6k search NPS measured in EXP-001, because perft neither evaluates, orders
nor probes a table), and our legality filter is expensive.

### Profiling - where the time actually goes

Measured inside njit loops, so Python call overhead is excluded:

| | startpos | kiwipete |
|---|---:|---:|
| `generate_pseudo_legal` | 0.43 us | 1.01 us |
| `generate_legal` (with legality filter) | 10.18 us | 24.48 us |
| make + unmake, per move | 0.323 us | 0.330 us |
| `in_check`, per call | 0.134 us | 0.132 us |
| **legality filter share of total** | **96%** | **96%** |

Pseudo-legal generation is already running at roughly 1-2M positions/s. The
entire remaining cost is the decision to prove legality by making every move,
testing the king and unmaking. That is 24x the cost of generating the moves.

**Conclusion: KEEP** the core - it passes every correctness gate. **The speed
claim is not yet earned**: 5.9x, not 40-100x, and the measured bottleneck is
named and quantified rather than guessed at.

---

## EXP-007 - Remove per-node heap allocations

**Hypothesis** `generate_legal` and `perft` each allocated a fresh
`np.empty(MAX_MOVES)` per node, putting two heap allocations on every node of
the tree. Removing them, via preallocated per-ply stacks, will materially raise
perft NPS.

**Result** perft NPS 1,807,510 before, 1,822,163 after. **Under 1%.**

**Conclusion: REJECT as a speed change.** The change is kept because per-ply
stacks are needed by the search anyway and the code is clearer, but the
hypothesis was wrong and Numba's allocator was not the bottleneck. Recorded so
the same idea is not proposed again as a speed fix.

---

## Pending experiments

Recorded now so they are not quietly forgotten or quietly assumed.

| ID | Hypothesis | Status |
|---|---|---|
| EXP-002 | Mobility in the evaluation is worth its 42% cost (21 us of a 50 us evaluation). Currently **off** by default on the argument that depth is worth more at this node rate. Needs an A/B against the champion. | NOT RUN |
| EXP-003 | S1 jitted core reaches >= 1M perft NPS. | **DONE - 1.82M perft NPS, all correctness gates passed, but only 5.9x python-chess** |
| EXP-006 | Legal move generation by pin mask and check-evasion mask, removing make/unmake from generation entirely. Profiling says this is 96% of `generate_legal`. Highest-value change available. Gated on the full perft and differential suite. | NOT RUN - NEXT |
| EXP-007 | Preallocated per-ply move stacks instead of per-node allocation. | **DONE - REJECT, under 1%** |
| EXP-008 | Sliding attacks: classical rays vs magic bitboards vs Hyperbola Quintessence, benchmarked under numba 0.67. Only after EXP-006, since rays are currently a small share of the cost. | NOT RUN |
| EXP-004 | Concurrency sweep: aggregate games/hour at 4, 6, 8 and 10 concurrent games, checking that individual game speed stays representative. | NOT RUN |
| EXP-005 | Quiescence needs delta pruning or SEE. At Kiwipete depth 1 costs 9,058 nodes, most of it quiescence. | NOT RUN |

---

## EXP-010 - Phase A: making S0 Swiss-safe

**Date** 2026-09-01 · Triggered by an external code audit, which found three
correctness bugs that all cost half points rather than centipawns.

| Bug | Symptom | Fix |
|---|---|---|
| `_last_key` suppressed repeated root positions | The de-duplication fired exactly when a position repeated - the one case that had to be counted. Repetition detection was disabled precisely when it mattered. | Removed. Every call records the position it was handed. |
| Only our own turns were recorded | The referee claims threefold on either side's turn, and we are only ever shown ours. Half the game's positions were never counted, so `ARCHITECTURE.md`'s claim to track "real game history" was false. | `agent.py` now also records the position our own move creates, including after a fallback, wrapped so bookkeeping can never cost a game. |
| Rule and heuristic conflated in one dict | A plain twofold was being scored as a FIDE threefold. | Separate `game_counts` (permanent, actual game) and `path_counts` (current search line). Threefold is `>= 3`. The twofold-in-search shortcut is retained but named `TWOFOLD_SEARCH_HEURISTIC`, documented as a heuristic, and A/B-able (EXP-009). |
| Quiescence stood pat in stalemate | A stalemated opponent scored as a material advantage, so **stalemating a lost opponent looked like winning**. | Quiescence detects no-tactical-and-no-legal-move and returns a draw. |
| Fifty-move rule checked before checkmate | A mate delivered with the halfmove clock past 100 was scored as a draw. The referee scores it as mate. | Checkmate now outranks the fifty-move rule, with move generation only in the rare high-clock case. |
| Insufficient material not modelled | K+B vs K evaluated as +335 while the referee calls it drawn. | Terminal insufficient-material test, gated behind a popcount so it costs nothing in normal positions. |
| Mate scores stored in the TT unadjusted | A mate score means "mate in N *from here*"; read back at another ply the distance was wrong. | Standard score-to-TT / score-from-TT normalisation. |
| TT key ignores the halfmove clock | Two positions identical except for being near the fifty-move boundary shared an entry. | The table is simply not used above halfmove 80. Rare enough to be free. |
| Packaging footgun | `deepblue/` was included only because the Makefile passed `--include`. This machine has no `make`, so the obvious bare command would silently build a submission that cannot import. | `deepblue` is a default include. The fresh-process test now runs the **bare** command so a regression fails the test. |

### Results

```
tools/regression.py            20/20 positions pass
tools/fresh_process_test.py    3 fresh processes x 14 probes, 0 problems
                               cold import 95-111 ms (budget 60,000)
bare `python -m harness.package`  produces a valid, importable submission
```

Two of the corpus entries were wrong when first written, and are recorded here
rather than quietly corrected: a `mated_stm` case that was merely losing rather
than forced mate, and a note claiming opposite-coloured K+B vs K+B is
insufficient material when it is same-coloured bishops that are.

**Conclusion: KEEP.** S0 is now safe to be the fallback.

---

## EXP-011 - Phase B: permanent invariant suite

`tools/fastcore_invariants.py`, deterministic seed 20260831, 4,000 positions.
Previously these numbers came from one-off commands and could not be re-run.

```
is_attacked vs python-chess        0 failures /   512,000 checks
in_check vs python-chess           0 failures /     4,000 checks
make/unmake exact restoration      0 failures /    67,254 checks
occupancy = union of bitboards     0 failures /     4,000 checks
mailbox agrees with bitboards      0 failures /     4,000 checks
```

**Conclusion: KEEP.** Every S1 change now re-runs this.

---

## EXP-012 - Phase C: the fused S1 search, and the measurement that mattered

**Hypothesis** The real search should not call `generate_legal`. Prefiltering
makes every searched move twice - once to prove it legal, once to search it.
A fused loop (generate pseudo-legal, make once, test the king, recurse,
unmake once) is both simpler and faster, and the 5.9x perft figure from EXP-003
measured the wrong architecture.

**Change** New `deepblue/fastsearch.py`: compiled negamax, alpha-beta,
iterative deepening, quiescence, compiled evaluator carrying S0's exact
material and piece-square semantics. No transposition table, no pruning, no
reductions - deliberately, so the architecture is measured unconfounded.

### Result: S0 vs S1-search0, identical positions, 2,000 ms, median of 3

Both engines count one node per negamax entry and one per quiescence entry, so
the convention is identical between them and comparable to no other engine.

| Position | S0 nps | S0 depth | S1 nps | S1 depth | Speedup | Depth |
|---|---:|---:|---:|---:|---:|---:|
| opening | 16,244 | 5 | 683,044 | 4 | **42.0x** | -1 |
| open middlegame | 9,176 | 1 | 271,006 | 4 | **29.5x** | +3 |
| closed middlegame | 12,515 | 3 | 375,502 | 5 | **30.0x** | +2 |
| tactical | 7,106 | 3 | 342,095 | 5 | **48.1x** | +2 |
| endgame | 15,921 | 6 | 589,829 | 7 | **37.0x** | +1 |
| **median** | | | | | **37.0x** | **+2** |

**37x in a real search**, against 5.9x measured on perft. The audit's call was
correct: perft's prefilter-then-re-make architecture was measuring something
the search does not do.

The depth column is the honest counterweight. S1-search0 reaches *fewer* plies
than S0 at the start position despite 42x the node rate, because S0 has a
transposition table, killers and a history heuristic and S1-search0 has none.
Raw speed is not depth, and depth is not strength.

**Conclusion: KEEP.**

---

## EXP-013 - Interruptibility: a flag bug found and fixed

**Symptom** With a 2,000 ms hard budget, the search returned after **4,112 ms**.
The timer thread fired correctly at 2,000.9 ms; the search then took a further
2,090 ms to unwind. In a real game that is a flag, and a flag is a loss.

**Cause** The stop flag was read only when `nodes & 2047 == 0`. Masking is what
makes a clock *syscall* affordable, but this flag is one byte in an array the
search already holds. Gating it meant 2,047 node entries out of every 2,048 did
their full work without ever looking.

**Fix** Read the flag on every node entry, and again inside every move loop so
a node with thirty candidates abandons the rest instead of finishing the list.

**Result** across 5 positions x 7 budgets from 20 ms to 2,000 ms, 2 repeats:

```
worst overshoot BEFORE : +2,112.0 ms
worst overshoot AFTER  :     +1.6 ms
```

**Conclusion: KEEP.** This is the single most important fix of the session.

---

## EXP-014 - Phase E: profile of the REAL search, and the EXP-006 verdict

Unit costs measured inside njit loops at Kiwipete (48 pseudo-legal moves):

```
pseudo-legal generation  0.919 us      make + unmake  0.323 us/move
in_check                 0.133 us      evaluate       0.079 us
move ordering            0.070 us
```

Modelled share of a node that examines every candidate:

| Component | us/node | Share |
|---|---:|---:|
| make + unmake | 15.522 | 67.6% |
| `in_check` legality test | 6.363 | 27.7% |
| pseudo-legal generation | 0.919 | 4.0% |
| move ordering | 0.070 | 0.3% |
| evaluation (leaves only) | 0.079 | 0.3% |

### Verdict on EXP-006: do not do it next

The EXP-003 profile said the legality filter was 96% of `generate_legal`, which
looked like an overwhelming case for pin-mask generation. In the **fused**
search that number does not transfer, because the `make` is no longer wasted -
the recursion reuses it. What pin masks would actually recover is:

* the `in_check` call after each make, **27.7%**, and
* the make/unmake of *illegal* candidates only, typically two to five of
  forty-eight in a quiet position.

So the realistic ceiling is roughly a 25-30% node-cost reduction, which at an
effective branching factor of 2-3 is about **+0.3 ply**.

Against that, S1-search0 currently reaches **fewer plies than S0** at the start
position despite 42x the node rate, purely for want of a transposition table
and move ordering. Those are worth plies, not fractions of one.

**Recommendation: EXP-006 is not worth its correctness risk yet.** Do C6
(Zobrist, array-backed TT, TT-move-first ordering, killers and history) first
and re-profile. EXP-006 should be reconsidered only once the search is
otherwise modern and the 27.7% is the largest remaining share.

---

## EXP-015 - The hardened S0 costs speed, and the fast-TC test exaggerated it

**Result** The Phase A correctness work made S0 slower: median search NPS
**17,112 -> 12,684**, about 25%. Node counts and depths on the benchmark suite
are otherwise identical, and on 24 random positions the hardened and original
engines choose the **same move with the same score in 24 of 24 cases**. So the
cost is speed, not judgement.

Head to head against the frozen champion:

| Time control | Result | Score | Elo | 95% CI |
|---|---|---:|---:|---|
| 4,000 ms + 100 ms | +0 =0 -20 | 0.0% | -800 | (before the increment fix) |
| 4,000 ms + 100 ms | +3 =1 -16 | 17.5% | -269 | [-717, -120] |
| 20,000 ms + 500 ms | +5 =0 -7 | 41.7% | **-58** | **[-319, +144]** |

The first row was largely a **testing artifact of my own making**. `agent.py`
hardcoded a 500 ms increment while the arena ran at 100 ms, so the engine
overspent roughly 375 ms every move, drained its clock and dropped into panic
mode playing unsearched moves by move eight. Fixed by observing the increment
from consecutive clock readings rather than assuming it, keeping the smallest
credible observation because budgeting for less increment than you receive is
safe and budgeting for more is a flag.

The remaining gap shrinks from -269 to -58 Elo, with an interval spanning zero,
simply by testing at a realistic time control. At 4,000 ms the engine reaches
depth 2-4, where losing 25% of the node rate frequently costs a whole ply and a
ply at that depth is worth a great deal. The audit's warning about correlated,
single-opening, fast-TC samples applies to this measurement in full.

**Conclusion: INCONCLUSIVE, and the champion does NOT change.** The hardened S0
is strictly more correct and is probably at rough parity at a real time control,
but twelve games from one starting position is not evidence of that. The
original champion stays frozen until a paired, opening-diverse test at the real
time control says otherwise. Correctness is not in question; the speed cost is
real and unexplained, and finding it is the next S0 task.

**Open** Where did the 25% go? It is not the stalemate probe (tightening its
gate from 12 pieces to 6 changed nothing measurable) and it is not
`_transposition_key`, which is now computed fewer times per node than before.

---

## EXP-016 - Audit round two: three real rule bugs in S1

**1. `insufficient_material` was far too broad.** It tested "at most one minor
each", which declares K+B vs K+N, K+N vs K+N and opposite-coloured K+B vs K+B
to be automatic draws. None of them are. Rewritten to match the referee's
actual asymmetric rule, which depends on what the *opponent* holds. Verified
against python-chess: **0 disagreements over 8 targeted cases**, including all
three the audit named.

**2. The fifty-move rule never reached quiescence.** `negamax` entered qsearch
at `depth <= 0` before any fifty-move test, and qsearch had none. Both engines
also ignored the rule at the root, where the referee claims it before asking us
to move: measured as **+556 for a position the referee scores as a draw**.
Fixed in both, with checkmate outranking it - the first version of that fix
zeroed a mate score and the corpus caught it immediately.

**3. Quiescence detected stalemate from *pseudo* move counts.** A position can
have pseudo-legal captures that are all illegal because they expose the king,
and no legal quiet move either. Replaced with a real legality probe that exits
at the first survivor, so it costs about one make/test/unmake in an ordinary
position. Two genuine instances were found by machine search and added to the
corpus: `8/8/8/5k2/8/1r6/K7/1r6 w - - 0 1` has two pseudo-captures, both
illegal, and is stalemate.

**Corpus verification caught one of my own labels.** `tools/regression.py`
now cross-checks every rule-sensitive expectation against python-chess instead
of trusting the handwritten tag. It immediately rejected a line I had labelled
stalemate that is actually checkmate.

**Move buffer.** `tools/movebuffer_fuzz.py` measured a worst case of **218
pseudo-legal moves over 30,000 probes**, against a 256 cap - 15% headroom on a
buffer that lives inside nopython code where overflow is silent corruption.
Raised to 320 (32% headroom) for a few hundred kilobytes.

**Artifacts.** `submission.zip` sometimes held the champion and sometimes the
candidate. Now `submission_champion.zip` (13,650 b, built only from
`champion/`) and `submission_candidate.zip` (28,818 b). A candidate packaging
successfully never promotes it.

---

## EXP-017 - C6: S1-search1

**Change** Zobrist hashing (`deepblue/zobrist.py`), an array-backed
transposition table with mate-score normalisation and rule-50 suppression,
TT-move-first ordering with killers and history, and repetition detection.
Nothing else - no pruning, no reductions, no aspiration, no SEE.

The hash is maintained *outside* `fastcore`, from the move's own bits and the
state either side of it, so the verified core is untouched and perft, the
differential suite and the invariants all still exercise exactly the code they
were green against.

**Hash invariants:** incremental hash equals full recompute, **0 failures /
37,523 moves**; make/unmake restores the hash exactly, **0 failures / 37,523
moves**.

### search0 vs search1, same suite, 2,000 ms

| Position | Engine | NPS | Depth | Nodes | Score | Move |
|---|---|---:|---:|---:|---:|---|
| opening | search0 | 452,419 | 4 | 213,265 | +12 | b1c3 |
| | **search1** | 366,394 | **7** | 377,149 | +8 | b1c3 |
| open middlegame | search0 | 185,930 | 4 | 213,447 | -3 | e2a6 |
| | **search1** | 183,781 | 4 | 191,502 | -3 | e2a6 |
| closed middlegame | search0 | 266,874 | 5 | 319,795 | +35 | d4c5 |
| | **search1** | 264,160 | **6** | 291,655 | +44 | d4c5 |
| tactical | search0 | 226,155 | 5 | 139,611 | -457 | c5c4 |
| | **search1** | 227,322 | **6** | 262,545 | -457 | c5c4 |
| endgame | search0 | 403,554 | 7 | 582,337 | +58 | b4f4 |
| | **search1** | 393,045 | **9** | 479,785 | +54 | b4f4 |

**Median +1 ply on 0.91x the nodes.** NPS is slightly lower, as expected -
that is the trade being bought, and it is the right direction.

### Paired playing strength, 8 openings x 2 colours, 50 ms/move

```
search1 vs search0: +6 =9 -1, paired score 65.6%
terminations: checkmate 7, threefold repetition 9
illegal moves / crashes / flags: 0
```

Sixteen games is a regression gate, not a rating, and no Elo is claimed from
it. Nine draws by repetition is itself evidence the new repetition detection
is live.

Regression corpus: **32/32 for all three engines** (reference, fast, fast1).

---

## EXP-018 - Profile of the REAL search1, and the next bottleneck

Instrumented call counts multiplied by separately measured unit costs, which
disturbs the search far less than timing inside it.

| Component | Share |
|---|---:|
| make + unmake | 46.9% |
| **pseudo-legal generation** | **39.6%** |
| `in_check` legality | 10.3% |
| evaluation | 2.8% |
| move ordering | 0.4% |

Quiescence is **49% of nodes at the start position and 96% at Kiwipete**.
TT hit rate 44% / 25% / 56% across the suite.

**The next measured bottleneck is not what it was.** In search0's model,
pseudo-legal generation was 4% of node cost. In the real search1 it is 39.6%,
because quiescence dominates the node count and every quiescence node
generates the **complete** pseudo-legal move list in order to use only the
captures - and a quiet q-node then generates it a *second* time inside the
legality probe.

So the measured candidate for the next change is **a capture-and-promotion
generator for quiescence**, plus avoiding the double generation at quiet
q-nodes. That is a contained change to move generation with a clear
measurement attached.

**EXP-006 (pin masks) remains deferred.** Its target, the `in_check` legality
test, is now 10.3% - down from the 27.7% modelled against search0, and far
below quiescence generation. It has not earned priority.

*No further feature is being started. Reporting for a decision.*

## 2026-09-06: the 60ms methodology bug, and what it invalidated

**Every paired result recorded before this date was measured at 60 ms/move**,
the default of `tools/paired_fast_variants*.py`. At 60 ms this engine
completes **depth 4**. Every search technique with a depth threshold
therefore never fired in the test that rejected it:

| technique | gate | old verdict | status |
|---|---|---|---|
| null move pruning | depth>=4, R=3 -> null searched at depth 0 | 40.6% reject | **WRONG - now shipped** |
| late move reductions | depth-gated | 43.8%, 5-6 attempts | **WRONG - now shipped** |
| ProbCut | depth>=5, 2nd path depth>=10 | 48.3% hold | retesting |
| singular extensions | depth>=8-10 | 48.1% | genuinely dead at min_depth 10 |
| LMP / razoring / SEE pruning | depth-gated | untested/rejected | see below |
| aspiration windows | needs deep ID | "neutral" | retesting |

Non-gated changes (check extension, mate distance pruning, eval terms) did at
least execute at depth 4, so their sign may hold, but their magnitudes were
calibrated at an operating point 25-50x faster than real play (competition
games average 1.7-2.9 s/move).

### New gate: tools/sprt_gate.py
Parallel workers, SPRT sequential stopping, real tournament seed positions as
openings. Self-validated: identical engines score exactly 50.0% (+4 =8 -4).
Fixed-depth mode is deterministic (kills the AC-001 "sign flip across
replays" jitter); time mode is the decisive gate for speed techniques, whose
payoff is extra depth and is invisible at fixed depth.

### Results
Depth reached at a 2-second budget, start position:

    fastsearch57 (old champion)   depth 7
    fastsearch65 (+NMP)           depth 8    4.3x fewer nodes
    fastsearch67 (+LMR)           depth 11   18x fewer nodes   <- SHIPPED
    fastsearch68 (+LMP)           depth 13   but see below

    NMP alone    59.2% over 60 games  (+64 Elo)
    NMP+LMR      56.5% over 100 games (+45 Elo)  -> shipped as fastsearch67
    LMP           5.0% over 20 games  (-511 Elo) -> REJECTED, decisively

### The LMP lesson (generalizable)
LMP scoring 5% is not noise, it is a signature. This engine tolerates
techniques that REDUCE-then-verify (LMR re-searches anything beating alpha;
NMP and ProbCut verify with a real search) but not techniques that DISCARD
moves outright (LMP). The reason is move ordering: SEE is implemented and
validated in deepblue/see.py but **was never wired into ordering**, so "late"
moves are frequently good moves here. Fix ordering before retrying any
discard-based pruning.

Razoring was implemented and measured structurally inert (0.1% node change --
its trigger essentially never fires); not worth a game test.

---

## 2026-09-06/07 overnight session

Four changes shipped, measured over 120+ paired games each. Fourteen
rejected. The pattern in what survived is the most useful result here.

### Shipped

| Change | Result | Mechanism |
|---|---|---|
| gravity history | **+67 Elo** / 120 games | history records failures as penalties and self-limits, instead of only growing until it saturates |
| SEE pruning in quiescence | **+32 Elo** / 120 games | quiescence was 59-71% of all nodes and unpruned; captures that lose material by force now skip their whole subtree |
| continuation history | **+39 Elo** / 80 games | quiet-move ordering conditioned on the opponent's previous move |
| bishop pair | **+41 Elo** / 120 games | 28cp; the base evaluation priced every bishop identically |
| single-legal-move exit | free | forced replies played at 0ms instead of consuming a full move budget |

### Rejected

**Time management -- eight distinct approaches, none positive.**

    moves-remaining schedule 28->44   neutral over two independent runs
    best-move stability               -39 Elo
    node-fraction effort              test invalid (see below)
    budget utilisation floor          -64 Elo
    INCREMENT_FRACTION 0.75->1.0      neutral (50.5% at 100 games)
    Stockfish moves-in-time formula   -70 Elo
    Stockfish sudden-death formula    -79 Elo
    complexity (legal-move count)     47.5% over 80 games

The conclusion is structural rather than a tuning failure. Total time is
fixed, and at this engine's strength roughly DOUBLING a move's time buys one
extra ply -- so shifting time between moves trades a ply here for a ply
there. Reallocation is close to zero-sum. The only time change that worked
did not reallocate anything: it stopped spending time on moves with no
alternative.

Two specific traps worth recording:

* The node-fraction candidate was tested in `--mode time`, which gives every
  move a fresh budget. Time management can only be measured in `--mode
  clock`, where spending now genuinely costs later. In fixed-time mode it
  read +107 Elo purely because it was allowed up to 2520ms against the
  baseline's 2000ms. Fixed-time testing of a clock change measures nothing.
* The Stockfish formula was first ported from the DEEPBLUE1 handoff, which
  implemented the `movestogo != 0` branch. This competition is SUDDEN DEATH
  with increment, which uses a different formula entirely. Both branches were
  then tested; both lost. Stockfish's curve is tuned for an engine reaching
  depth 20+, where thinking longer in complex positions pays; at depth 9 the
  returns flatten much sooner.

**Search techniques.**

    singular extensions   47.5%   costs +43-55% nodes for the same depth
    TT 20->22 bits        50.8%   1M -> 4M entries; cache locality offsets retention
    delta pruning         38.8%   stacked on SEE, over-prunes quiescence
    capture history       dropped +37-120% nodes at fixed depth; two bound values tried
    lazy evaluation       dropped no NPS gain

**Evaluation.**

    rook open/semi-open file   51.9% over 80 games -- discarded, borderline
                               (constants 22/11/18 were never tuned for this
                               engine; a scaled version remains untested)

### What generalises

Every accepted change improves how well the engine searches each node --
ordering quality or evaluation accuracy. Every rejected change tried to make
it search *more* nodes, or moved time between moves. On an engine that is
already heavily pruned (NMP, LMR, RFP, futility, SEE), additional pruning
mostly re-cuts what is already cut, while ordering and evaluation still have
real headroom.

### Methodology notes

* Node counts predicted outcomes reliably; wall-clock microbenchmarks did
  not. An "eval is 80% of runtime" measurement, taken by calling `evaluate()`
  from Python in a loop, was dominated by Numba dispatch overhead that does
  not exist inside the compiled search. The lazy-eval candidate built on that
  reading produced no gain, which is what exposed it. Counted quantities are
  trustworthy here; timed ones need care.
* A reading of ~53% at 80 games is genuinely ambiguous in this project:
  three separate candidates sat there and went on to 51.9% (discarded),
  50.5% (decayed to nothing) and 54.6% (shipped). Candidates in that band get
  120 games before a decision, applied uniformly.

### Late additions (same session)

    doubled/isolated pawns    +66 Elo / 80 games   SHIPPED
    knight outposts           51.2% / 80 games     discarded

Doubled/isolated pawns was the strongest single result of the session and the
only candidate to reach formal significance (interval excluding zero) at 80
games rather than needing 120.

Knight outposts is worth recording alongside rook placement, because the two
failed eval terms share something the four successful ones do not: their
WEIGHTS WERE INVENTED rather than taken from an established source. Bishop
pair used the conventional 28cp, the pawn penalties are standard values, and
gravity/continuation history inherited constants from the already-tuned
history table. Rook placement (22/11/18) and outposts (20/30/22 by rank) were
both guessed, and both landed in the 51-52% neutral band -- close enough to
suggest the CONCEPT is sound and the numbers are not. Retuning either is a
separate experiment; with a deadline close, eval terms whose weights come
from an established source are the better bet.

    steeper passed-pawn curve (1.5x)   46.9% / 80 games   discarded

This refines the "invented weights" pattern above rather than confirming it.
The passed-pawn CURVE is established and already measured positive; only the
1.5x multiplier applied to it was invented, and that was enough to lose 22
Elo. Overvaluing a passer evidently pushes the engine to advance it into
positions where it is won rather than promoted. The distinction that actually
holds across the session is narrower than "established source": the four
eval/ordering changes that worked used values AS PUBLISHED, while all three
that failed (rook placement, knight outposts, this) used values someone here
chose or scaled.

    Ethereal budget formula (time+25*inc)/20   50.6% / 80 games   discarded
    node-effort scaling (Ethereal 0.5-2.4)     built, untested

Ethereal's formula is MORE generous early than ours (6.6s vs 4.66s on move 1)
and still measured neutral, which closes the last structural question: our
allocator is not mis-shaped in either direction. Nine approaches tested, none
positive. The only clock change that ever worked did not reallocate anything
-- it stopped searching when there was exactly one legal move.

### Texel tuning of the evaluation weights (2026-09-07)

    tuned weights (14 params, fitted on 120k positions)   38.8% / 80 games   REJECTED

Fitted by coordinate descent on the logistic objective against positions
carrying depth 46-58 engine evaluations, 20% held out. Both training (+1.97%)
and holdout (+1.87%) error improved, and the resulting weights were
chess-sensible -- a passed pawn on the seventh went 95 -> 163, the doubled
penalty 12 -> 48. It still lost 80 Elo in actual games.

Two things worth keeping from this:

* THE OBJECTIVE MEASURES THE WRONG THING. Agreement with a strong engine's
  STATIC score is a proxy. Our evaluation is not used statically -- it ranks
  moves inside a search millions of nodes deep, and a weight set can agree
  better on average while ranking worse. Where a proxy objective and a paired
  game result disagree, the game result is measuring the thing we care about.
* THE WARNING WAS VISIBLE BEFORE THE TEST. The fit set BISHOP_PAIR_BONUS to 4
  when a 120-game match had measured that term at +41 Elo with 28, and the
  fitted K bottomed out at 0.05 (the floor of the search range), meaning the
  logistic had degenerated to nearly linear and stopped weighting near-equal
  positions -- which is the entire point of the method.

A worthwhile revisit would pin the weights already measured in games and free
only the untested ones, with K constrained to a sane range. Not attempted here
because a proxy that contradicts a direct measurement has already said
something about how much to trust it.

TWO BUGS found while building this, both of which would have silently produced
weights that do not transfer:

* The dataset's cp is WHITE-relative while our evaluation is SIDE-TO-MOVE
  relative. Correlation of base_evaluate against cp measured +0.343 on
  white-to-move positions and -0.296 on black-to-move ones, cancelling to
  0.023 overall -- the first fit was against pure noise, and produced weights
  saying passed pawns are worthless and mobility is harmful. After aligning
  the convention, correlation is +0.432.
* The feature extractor initially omitted the `attacks & ~own_occ` mask that
  mobility_white_relative applies, and carried the pawn-penalty sign in both
  the feature and the weight, which cancels and turns the penalty into a
  reward for doubled pawns. Verified afterwards by reconstructing evaluate()
  exactly: residuals are zero except for the deliberately-excluded quadratic
  king-danger term.

### Bundle of individually-rejected features (2026-09-07)

    rook placement + knight outposts + TT 22 bits    +61 Elo / 80 games   SHIPPED

Individually these measured 51.9%, 51.2% and 50.8% and were all discarded as
neutral. Together they measure 58.8% (+61.4 +/-61) over 80 games, positive at
every checkpoint.

The lesson is about the INSTRUMENT, not the features. This gate resolves
roughly +/-30 Elo at 80-120 games, so a change genuinely worth +10 reads ~51%
and gets thrown away. Stacking three of them clears the noise floor. Other
candidates sitting at 50-52% with a sound mechanism should therefore be
treated as UNRESOLVED rather than rejected. Candidates that read clearly
negative are a different matter -- those are not invisible-small, they are
harmful.

### Import time: every warm measurement in this file was optimistic

Numba caches compiled functions to disk, so repeated imports in one session
are far faster than the competition's fresh process. Measured with an empty
NUMBA_CACHE_DIR:

    fastsearch109   COLD 85.3s of the 90s budget   (warm readings said 39-43s)
    fastsearch114   COLD 83.8s of the 90s budget

Missing the init budget loses the game outright, so this is the tightest
constraint on the engine and it was being measured wrongly all session. The
mitigation of shipping the .nbc/.nbi cache files is not available: they are
native compiled code and the rules prohibit shipped native binaries.

Empirically the platform is managing it -- rounds 46-54 all completed on
fastsearch109 -- so real hardware is faster than this laptop's cold compile.
But the margin is single-digit seconds and any further search code must be
measured COLD before shipping.

---

## 2026-09-07: the measurement pipeline cannot resolve what we have been asking it

Prompted by a direct question -- are we actually improving? -- against the
observation that rated play keeps oscillating around 1740 while this file
records win after win.

### The contradiction

Shipped gains recorded in this file:

    NMP+LMR +45, gravity history +67, SEE qsearch +32, continuation history
    +39, bishop pair +41, doubled/isolated +66, rook+outposts+TT22 +61,
    razoring +70, futility +25, mobility +61, pawn structure +39,
    contempt 20 +20                                    TOTAL  +566 Elo

Rated performance over the same period: flat. Both cannot be true.

### What the harness can actually resolve

Standard error of a paired match, converted to Elo at the 50% slope:

    games      1 SE          95% CI
      60      +/-45 Elo     +/-88 Elo
      80      +/-39 Elo     +/-76 Elo
     120      +/-32 Elo     +/-62 Elo
     240      +/-22 Elo     +/-44 Elo

EVERY gain in the list above is smaller than the 95% interval of the test
that measured it. The 80-game standard (adopted over 60 precisely to be
more careful) resolves +/-76 Elo; the largest claimed increment is +70.

### The mechanism: selection on noise

Testing a stream of candidates that are truly worth ~0 and shipping those
reading above ~54% does not select good changes, it selects lucky ones. The
recorded "gain" is then the size of the luck, which at these sample sizes
averages +40 to +70 -- exactly the range this file is full of. Twelve
iterations of that produce "+566 Elo" and no rating movement.

Worse than useless: a change truly worth -20 Elo still reads above 50% about
one time in three at 80 games. Twelve shipped changes means several probably
ARE negative, and nothing in these records would distinguish them.

Bundling, argued for earlier the same day on the grounds that small effects
are individually unresolvable, makes this worse rather than better: it raises
the chance a group clears the bar, and the precedent cited in its favour
(three individually-rejected features measuring +61 together) is equally well
explained by three coin flips landing heads.

### What this does NOT invalidate

Two categories survive:

* Techniques whose true effect is far above the noise floor -- transposition
  table, PVS, quiescence, null move, LMR, futility, MVV-LVA/killers/history.
  Our measurements of them were noisy, but effects of 50-100 Elo are large
  enough that the SIGN is safe even if the magnitude is not.

* Bug fixes verified by MECHANISM plus a reproduced real game, not by a win
  rate. fastsearch52's mate-score fix (replayed round 20 and showed the old
  code plays the losing move) and the graded king shield (showed the shield
  mask scores a pawn on h3 identically to one on h2, and showed the round-58
  game where that mattered) are both in this class. This evidence is STRONGER
  than a 53% score over 80 games, because it explains a cause instead of
  counting outcomes.

Everything in between -- hand-picked eval constants, margin tuning, contempt,
gravity-vs-plain history -- is unproven. The contempt axis is the clearest
illustration: contempt 20 sits in the champion, contempt 35 measured 46.2%
over 80, contempt 10 was queued next. That is sampling, not tuning.

### Consequences adopted

1. Do not ship on a ">50%" reading. That rule applied to sub-resolution
   effects is a noise amplifier.
2. Prefer mechanism-backed fixes over score-backed ones. Prefer a reproduced
   game and a named cause to any win rate we can currently produce.
3. Stop adding hand-weighted evaluation terms. We cannot measure them, so we
   cannot know we are not hurting ourselves.
4. When choosing what to submit, prefer the SIMPLER engine among candidates
   that cannot be distinguished -- fewer unproven constants means fewer
   chances that a negative one slipped through.

### Open audit

fastsearch118 vs fastsearch67 (accumulated claim between them: ~+496 Elo,
which predicts ~95% for 118). Result to be recorded here whichever way it
lands. If it comes back near 50-60%, points 1-4 above become the operating
rules for the remainder of the competition.

### Audit result (2026-09-07 21:29) -- the aggregate IS real

    fastsearch67 vs fastsearch118    80 games  +5 =12 -63   13.8%
                                     -319 Elo +/-92   LLR -3.03  (terminated)

118 is +319 Elo stronger than 67. The section above speculated that the
process might have produced nothing; that is now refuted and the speculation
was an over-correction.

The honest reading, with both halves kept:

* Claimed between 67 and 118: +496 Elo. Measured: +319. The claims ARE
  inflated -- by about 55%, which is precisely the winner's-curse signature
  the section above predicts. Individual increment numbers in this file
  should be read as upper bounds, not estimates.

* But the accumulated direction and magnitude are real. The pipeline is
  inefficient and its per-change numbers are overstated; it is not fictional.
  The four operating rules above stand on the grounds that we cannot RANK
  small changes -- not on the discredited grounds that nothing has worked.

This sharpens rather than answers the original question. If +319 is real in
self-play while rated performance oscillates around 1740, either self-play
Elo is transferring poorly to a 334-engine field (expected, and largely
outside our control), or the gains are HISTORICAL and recent increments have
stalled. The remaining bake-off matches (91, 105, 109, 114, 116 against 118)
distinguish these: if the late lineage clusters at 50%, the recent versions
are the same engine and the flat recent rating is explained.

### Full lineage bake-off (2026-09-07/08) -- the gains stopped at 105

Every shipped version against the current champion, 80 games each, run
SEQUENTIALLY with all 7 workers so no match was distorted by CPU contention.

    version          score    gap to 118     significant?
    fastsearch67     13.8%    +319 +/- 92    YES, decisive
    fastsearch91     29.4%    +152 +/- 60    YES, decisive
    fastsearch105    32.5%    +127 +/- 62    YES, decisive
    fastsearch109    42.5%    + 52 +/- 63    no
    fastsearch114    46.9%    + 22 +/- 57    no
    fastsearch116    49.4%    +  4 +/- 53    no -- identical

Read together with the claimed increments, this splits the project's history
cleanly in two.

67 -> 105 is REAL: +192 Elo measured across that stretch, decisive at every
step. Null move, LMR, futility, gravity history, SEE quiescence pruning and
continuation history are worth what an engine textbook says they are worth,
and the pipeline found them.

105 -> 118 is NOT: +127 down to +4, never significant, and the four versions
shipped in that window (109, 114, 116, 118 -- marketed as 2.7, 2.8, 2.9, 3.0)
are one engine. The claimed increments over that window were rook placement
+ knight outposts + TT 22 bits (+61), contempt 20 (+20) and razoring (+70):
+151 claimed, +52 measured against 109 and not distinguishable from zero.

The sharpest single refutation: 116 -> 118 IS the razoring change, recorded
above at "+70 Elo / 80 games" and the reason 118 became champion. Head to
head it measures +4.3 +/- 53.

What separates the two eras is not effort or care -- it is that the first era
added SEARCH TECHNIQUES with true effects of 50-100+ Elo, comfortably above
this harness's +/-60 resolution, and the second added EVALUATION TERMS and
MARGIN TUNING whose true effects are in the 0-20 Elo band where the harness
returns coin flips. The pipeline did not break. It ran out of things it was
capable of measuring, and kept reporting results anyway.

Consequence for the remaining days: adding another hand-weighted evaluation
term is the one activity with a demonstrated zero return. Effort belongs on
defects that can be shown mechanically -- see the round 58 / round 60
decomposition below, which separates a SEARCH failure from an EVAL blindness
for the first time in this project.

### The graded king shield: correct mechanism, WORSE engine (2026-09-07/08)

    fastsearch127 vs fastsearch118   80 games  +16 =33 -31   40.6%
                                     -65.9 +/- 59 Elo   (CI [-125, -7], excludes zero)

Rejected. Kept here because everything about its justification was true and it
was still wrong, which is the most useful failure this project has recorded.

What was true: the flat shield mask counts rank+1 and rank+2 as an equally
intact shield, so a pawn on h3 in front of a king on g1 scores identically to
one on h2. Demonstrated in the source, and demonstrated in round 58 where the
engine charged itself 15cp for 18. g4 and NOTHING for 17. h3, then was mated
using g3 and h3 -- the two squares those pushes vacated. Third loss of that
exact shape. The graded term fixed it: h3 0 -> -10, g4 -15 -> -32, the engine
stopped playing g4 at every time control, and the regression corpus passed
33/33 including a new avoid:g2g4 entry.

What was false: that any of the above predicts strength. The term cannot
distinguish a pawn that ABANDONED its king from one that is ATTACKING. Found
by applying it to a second game (round 60) before testing: with White's king
on g1 and pawns on g3/h4 -- White's attacking plan -- the term charged White
76cp as if the king were exposed. An engine that will not storm pawns loses
more than a correctly-priced h3 gains.

Two rules follow, and they are the ones that would have prevented shipping it:

1. A mechanism story is a reason to TEST, never a reason to ship. This one had
   a named bug, a reproduced game, a fixed regression entry and a clean
   symmetry check, and it cost 66 Elo.
2. Check every new term on a game where the OTHER side is doing the thing the
   term penalises. The defect was invisible in round 58 (where we were the one
   with the weak king) and obvious in round 60 (where we were not).

Postscript on process: this candidate was packaged, sent to the user and came
within one instruction of being uploaded on a ">50% and it shows promise"
rule. It never reached 50%.

### Time management: NOT a defect (2026-09-08, closed)

Three separate claims were made tonight that the clock is being wasted. All
three were wrong. Recorded in full so the thread is not reopened.

    claim 1  "spends 2.87s/move early vs 1.50s late, 2:1 front-loaded"
             FALSE. The 1.50s included 274 forced and mate-score moves across
             70 games -- correct behaviour, not thinking time. Excluding
             moves under 0.25s: 2.87s early vs 2.24s late.

    claim 2  "the predictive stop has degenerated into a flat 50% cap, the
             search throws away half its budget"
             FALSE. Per-move attribution (tools/clock_sim.py) over round 58:
             79% of allocation spent, and TEN of 31 moves ran OVER soft into
             hard time at 126-140%. Stop reasons: predicted overrun 16, soft
             budget 10, forced 3, mate found 2.

    claim 3  "32% of the 120s clock is never spent"
             TRUE as an average, MEANINGLESS as a diagnosis. It is entirely
             explained by game length:

                 round 58   39 moves    spent 79% of allocation   36% clock left
                 round 60  149 moves    spent 81% of allocation    2% clock left

             The allocator reserves against a game length it cannot predict.
             Short games end with the reserve unspent; long games consume it
             almost entirely. That is correct risk management. An allocator
             that spent the reserve would flag when a game ran long -- which
             is the failure mode the 28/12 constants were chosen to avoid.

What survives, and it is small: on round 58 the losing move 18 got 2484ms of
a 4060ms budget (61%) and reached depth 10, and moves 15-18 all sat at 59-65%
of soft. Raising the predictive-stop floor (fastsearch128, 0.50 -> 0.80) would
recover roughly 1.5s across that stretch. That is a 0-20 Elo lever in a band
this harness cannot resolve -- worth one cheap 240-game match, not a project.

METHOD NOTE, which is the real lesson. Every one of the three false claims
came from aggregating across games and inspecting the components afterwards.
Every one died to a per-move breakdown. Averages over heterogeneous games
(different lengths, forced moves, mate scores mixed in) are how this project
manufactures phantom defects.

### fastsearch131: four standard techniques, ACCEPTED (2026-09-08)

    fastsearch131 vs fastsearch118   82 games  +42 =26 -14   67.1%
                                     +117 Elo    VERDICT: ACCEPT H1
    (screen at 1000ms/move, H1=+30; confirmation at 2000ms pending)

First formal SPRT acceptance of the session and the largest gain since the
67 -> 91 era. Contents, and where each came from:

    late move pruning          rejected here at 60ms/move -- a time control
                               where the engine reaches depth 4 and LMP is
                               depth-gated, so it never executed
    delta pruning (qsearch)    profiling showed 48.7% of all nodes are
                               quiescence; we pruned it with SEE alone
    SEE pruning, main search    Berserk and Ethereal prune losing captures in
                               the main search; we had SEE in qsearch only
    internal iterative         measured TT miss rate 72.8%. Took the
    REDUCTION                  reduction form, not classical IID: IID pays a
                               search to decide better, the same trade that
                               ProbCut (49.4%) and singular extensions
                               (47.5%) both lost in this engine

Smoke test before the match: +8 plies of depth across six positions, 3.4%
FEWER nodes, and the same move chosen in all six -- the signature of pruning
that works rather than pruning that changes decisions.

WHY THIS ONE WORKED AND THE OTHERS DID NOT. Every candidate that failed this
session came from a theory of mine about what the evaluation was missing:
the graded king shield (-66 Elo), extended king-danger zone (+44 then -57 on
two runs of the same pair). Every component of 131 came from either a
measurement of this engine (qsearch share, TT miss rate) or a gap against a
published feature list for a ~3000 Elo open-source engine. Guessing lost
three times; measuring and copying architecture won once, by +117.

ON BUNDLING. This is a four-technique bundle, argued for on exactly the
grounds set out earlier the same day when bundling was rejected for eval
terms: bundling is illegitimate when nothing fixes the sign of the components
(hand-weighted eval constants), and legitimate when literature and mechanism
both do (textbook search techniques absent from our engine). Individually
each is worth +10-20 and therefore invisible against +/-100 Elo of run-to-run
noise -- LMP alone measured 53.8% (+26 +/- 48), i.e. nothing. Together they
are unmissable. The earlier bundles failed because they bundled the wrong
KIND of change, not because bundling is wrong.

COST TO WATCH: compile time 39s -> 45s (+15%) measured on an idle machine.
Platform init budget is 90s and a miss is an instant loss; user-reported
platform init for the 2.7 build was ~44s.

### Candidates against fastsearch131 (2026-09-08)

All screened at 1000ms/move, H1=+30, against the new champion. 1000ms was
validated first: mean depth 10.3 vs 12.3 at 2000ms, and the deepest gate in
use is NMP at depth>=4, so every technique still fires -- unlike the 60ms
tests that wrongly rejected LMP, NMP, LMR and razoring at depth 4.

    fs132  correction history, LINEAR curve      100 games  51.0%   +6.9 +/-53
    fs134  TT 22->24 bits (76MB -> 304MB)        300 games  51.7%  +11.6 +/-32
    fs135  correction history, QUADRATIC curve   300 games  54.2%  +29.0 +/-31
    fs133  late move reduction for captures      260 games  48.7%   -9.4 +/-34  REJECT

CORRECTION HISTORY: the technique was fine, the curve was mine and it was
wrong. Stockfish applies the correction as cv*|cv|/11175 -- QUADRATIC, so a
weakly-confirmed correction barely moves the evaluation and only a repeatedly
confirmed one moves it a lot. fastsearch132 used entry//8, linear, which at
entry=128 applies 16cp where the reference applies 1cp. Same code path, same
technique, only the curve differs: 132 measured ~0 and wandered between 43.3%
and 51.0% over five checkpoints; 135 measured +29.0 +/- 31 over 300 games.
Reading the reference implementation would have saved a 100-game run.

CAPTURE LMR, and why a technique from a stronger engine can still be wrong
here: Ethereal reduces captures by 3 (2 if giving check), and our LMR already
uses Ethereal's exact quiet formula, so this looked like a pure gap. It
measured -9.4 and formally rejected. The likely reason is that fastsearch131
had just added SEE PRUNING TO THE MAIN SEARCH, which already removes captures
that lose material by force -- so what remained to reduce were captures worth
searching. Same failure mode as taking Stockfish's KingAttackWeights without
the machinery those weights belong to: a technique is calibrated against its
own engine's surroundings, and ours had just changed.

TRANSPOSITION TABLE: published figures put the first hash doubling at +50-70
Elo. Two doublings here measured +11.6 +/- 32. The gap is worth recording:
at ~1M nodes per move a 4M-entry table already holds most of a single search,
our 72.8% miss rate is largely genuinely unique positions rather than
evictions, and the TLB pressure the literature warns about offsets part of
the gain. Published Elo figures are calibrated to engines searching far more
nodes than this one.

## 2026-09-08 evening: what the annotated games actually show

Rounds 61-73, several with Stockfish annotations supplied by the user. This is
the first externally-validated evidence this project has had; every prior
diagnosis used our own engine as referee, which structurally cannot find errors
our evaluation shares -- i.e. exactly the errors that cost games.

### Errors concentrate in losses, and they are small

Filtered scan (playable positions only: |eval| < 800, >=3 legal moves, not in
check -- see tools/blunder_scan.py for why each filter exists):

    wins   r61 r63 r66 r67   1 error >=100cp in 232 moves   (0.4%)
    draw   r65               2 errors in 52 moves           (3.8%)
    losses r62 r64           5 errors in 77 moves            (6.5%)

Sixteen times the error rate in games we lose. The engine is close to
error-free in games it wins; losses come from a handful of 150-530cp mistakes
in still-playable positions, not from a general tendency to blunder.

### Three specific undervaluations, all externally confirmed

    enemy queen beside our king      priced at 11cp   (r60, r62, r69)
    our pawn one square from queening priced at 95cp   (r73)
    round 72                          NO error found at all

The first is the KING_ZONE_ATTACK_WEIGHT_Q = 10 inversion (fixed in
fastsearch140, which also fixed the round-58 g4 blunder that three earlier
attempts could not). The second is PASSED_BONUS peaking at 95 on the seventh
rank. The third is the important one.

### Round 72: the loss with no mistake in it

71 moves, lost by checkmate. Our scan: ZERO errors >=100cp. Stockfish, per the
user: no real mistakes -- we played GOOD moves where the opponent played BEST
moves. The evaluation bled steadily from -55 at move 7 to -893 at move 57 in
slices of 20-47cp.

Corroborated by the one two-sided measurement we have (round 64, tools/
two_sided.py): opponent mean cost 0.2cp per move, ours 23.2cp.

THIS IS THE CENTRAL FINDING. Against strong opponents we do not lose to
identifiable defects. We lose ~20cp per move, sustained, and after sixty moves
that is a lost game with nothing for a game review to flag. No constant fixes
that. Only a better evaluation (NNUE) or materially more depth does.

It also explains the session's results: search work measured +52 and +27,
while five of six evaluation-constant changes measured flat or negative.
Search improvements lift every move slightly; a constant only helps in the
rare positions where that term dominates.

### Clock distribution, round 72

    moves <= 25 :  3.57 s/move   (18 real thinking moves)
    moves >= 40 :  1.62 s/move   (23 real thinking moves)

We spend most where the position was handed to us and least where the game is
lost. The allocator was not the constraint -- at move 40 it budgeted ~2.9s and
the search used 56% of it, while in the opening the search ran OVER soft into
hard time. Endgame iterations are cheap and deep, so the growth prediction
fires early. Tested as the predictive-stop floor change in the search bundle.
