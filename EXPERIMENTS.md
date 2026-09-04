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
