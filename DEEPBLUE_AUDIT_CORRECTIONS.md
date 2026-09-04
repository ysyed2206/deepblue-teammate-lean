# Deep Blue — audit corrections

Corrections to previous planning-document claims, recorded because a prior
research pass already contained at least one material mistake and the brief
requires every "missing feature" or "why did this happen" claim to be
verified against actual source before being trusted or acted on.

---

## AC-001 — The growth-based iteration predictor is timing-threshold-sensitive,
## not just "noisy from warmup/OS effects"

**Where** `deepblue/fastsearch4.py:576-603` (`FastEngine4.search`), and
identically in every fastsearchN variant that inherits this loop unchanged
(14, 15, 17 all carry the same code).

**Prior claim (CLAUDEREAD.md "TIMING WARNING")** that single fixed-depth
elapsed-time values "varied heavily with warmup / OS effects" and the fix is
just to run more repeats.

**What source inspection actually shows** The stopping rule is:

```python
if previous_ms > 1.0:
    growth = min(6.0, max(2.0, iteration_ms / previous_ms))
previous_ms = iteration_ms
if elapsed + iteration_ms * growth > soft_ms:
    break
```

`growth` is a single-sample ratio of the last two iteration times, clamped to
`[2.0, 6.0]`. At positions where the branching factor jumps sharply between
two consecutive depths (typical in tactical positions right before a
combinatorial opening-up), this ratio saturates at the 6.0 cap and the
predictor becomes a hard pass/fail test with **no margin**: whether depth
N+1 is attempted depends on which side of the threshold ordinary OS/CPU
timing jitter (context switches, turbo-clock ramping, GC, thread wakeup
latency for the `threading.Timer` hard-stop) lands the *measured*
`iteration_ms` for depth N.

**Verified reproduction** (`deepblue/fastsearch4.py`, unmodified champion,
FEN `r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1`,
`soft_ms=1000, hard_ms=1400`, three calls in one warmed process, fresh
engine instance each call):

```
run 0: depth 5, 62,243 nodes, 187 ms, score -457
run 1: depth 9, 326,513 nodes, 844 ms, score -455
run 2: depth 5, 62,243 nodes, 172 ms, score -457
```

Node counts at depth 5 are **bit-identical** between run 0 and run 2
(62,243 both times), which proves the search tree itself is deterministic —
the flip is entirely in the wall-clock-based decision to attempt depth 6,
not in the algorithm.

**Consequence for every timed comparison already recorded** A single-repeat
(or low-repeat) timed run can show one candidate "stopping 4.5x earlier"
than another on the exact same position purely from this coin-flip, with
zero relation to the code change under test. The per-position *last-sample*
rows printed by `tools/compare_fast_variants.py` are therefore **not
individually trustworthy** at low repeat counts; only the script's
per-position-then-cross-position **median-depth summary line** (which
already medians over `--repeats`) should be read. This session used
`--repeats 9` for exactly this reason; even that is a minimum, not a cure.

**Action taken** None to the champion (out of scope / no correctness
issue — this only affects self-measurement, not game legality or score
correctness). Recorded so nobody re-derives "candidate X collapses under
time pressure" from a low-repeat run again. If this predictor is ever
revisited as a feature (time management is item 20 in the donor roadmap),
the fix is to widen the growth estimate's memory (e.g. average of the last
2-3 ratios, or a softer margin before capping) rather than a single-sample
6x-clamped ratio.

**Addendum — this also contaminates the paired playing-gate signal, not
just node-count benchmarks.** The identical 16-game gate
(`tools/paired_fast_variants.py fastsearch4 fastsearch15/17 --move-ms 80
--openings 8`) was run twice back-to-back with no code change in between:

```
fastsearch15 vs fastsearch4: run1 +6=6-4 (56.2%)   run2 +5=6-5 (50.0%)
fastsearch17 vs fastsearch4: run1 +5=8-3 (56.2%)   run2 +5=4-7 (43.8%)
```

fastsearch17's result **flips sign** between two identical reruns — from
"mildly positive" to "mildly negative" — purely from this timing jitter, at
an 80 ms soft / 112 ms hard budget where a single iteration is a large
fraction of the whole move budget and the depth-N-vs-N+1 coin flip described
above can decide the move actually played, which then diverges the rest of
the game. This means **16-game gates at fast time controls carry a real,
now-quantified noise floor beyond ordinary opening-sampling variance**, and
a single 16-game run should never be read as a verdict — see
`CLAUDE_REMOTE_RESULTS.md` for the pooled-sample readings used instead.

---

## AC-002 — fastsearch14 / P1a's guarded branch is provably unreachable in
## normal play, not merely "structurally inert on this suite"

**Where** `deepblue/fastsearch14.py:455-456` (TT write path inside
`negamax`).

**Prior claim (CLAUDEREAD.md "FASTSEARCH14 — P1a")** that the patch
"appears structurally inert" on the fixed-depth suite and next steps were to
"determine whether the condition is simply rare, our TT replacement
semantics differ, or the patch is genuinely irrelevant."

**What source inspection + instrumentation actually shows** The patch's
guard is:

```python
if best_move == np.uint32(0) and tt_key[index] == value:
    best_move = tt_move_arr[index]
```

`best_move` is initialised to `0` (`fastsearch14.py:381`) but the move loop
sets `best_move = move` the moment **any** move improves on `-INFINITY`
(`fastsearch14.py:422-423`), which is the first move examined at any node
with at least one legal move. So `best_move` is non-zero at the TT-write
point for every node that reaches it having searched at least one move —
including ordinary fail-low (all-node) results, which is exactly the case
the donor roadmap's "TT move preservation" idea is meant to help. The
`best_move == 0` guard as coded can only be true for a node that writes to
the TT having examined **zero** moves, which does not happen on this
engine's move-loop structure outside of a mid-loop timeout abort (and an
abort skips the TT write entirely via `tt_usable and not stop[0]`).

**Verified instrumentation** Added two temporary counters,
`P1A_CANDIDATE` (guard's condition true) and `P1A_FIRE` (guard true *and*
would have actually changed `best_move`), to `deepblue/fastsearch14.py`.
Ran all 5 engineering-suite positions at ~1s/move (1.5M+ total nodes):

```
P1A_CANDIDATE = 0   P1A_FIRE = 0   (all 5 positions)
```

Zero fires across every position, confirming this is not sampling luck —
the branch is dead code under normal execution, which is *why* fastsearch4
and fastsearch14 are bit-identical on every equal-depth and timed
comparison run so far.

**Correction to record** The donor idea ("preserve a useful TT move instead
of overwriting it with a null move") is real and used in Reckless-family
engines, but this port's precondition is wrong: it should gate on the
*bound type* being produced (`best <= original_alpha`, i.e. an upper-bound
write) rather than on `best_move == 0`, since an all-node here still
carries a real (if unhelpful) `best_move` from the first move tried. This
session did **not** implement that correction — see
`CLAUDE_REMOTE_RESULTS.md` fastsearch14 entry for the recommendation
(REJECT as coded; a corrected version would be a *new*, independent
candidate, not a revival of fastsearch14).

---

## AC-003 — Deep Blue's history table is piece-to, NOT the "butterfly"
## (side x from x to) shape `DONOR_NEXT_IDEAS.md` claimed it already had

**Where** `deepblue/fastsearch4.py:540` (and identically in every
fastsearchN variant that carries the history table forward unchanged,
including fastsearch18/20/21/22): `self.history = np.zeros((12, 64),
dtype=np.int32)`, read/written as `history[piece, to_square]`
(`fastsearch4.py:159,432`) — 12 piece-color combinations x 64 destination
squares, no side-to-move dimension of its own (folded into the 12) and,
critically, **no from-square dimension at all**.

**Prior claim** (`DONOR_NEXT_IDEAS.md`, "History decomposition" > "Table
shape"): *"Pawnstar is the closest match to Deep Blue's existing butterfly
(side x from x to)... 16384 was not Patch A's error — it's the majority
constant."* This describes Deep Blue's table as already being a from-to
butterfly. **It is not.** A true butterfly table needs a from-square axis;
Deep Blue's has none. The prior write-up conflated "Deep Blue has *a*
history table indexed in a donor-precedented way" with "Deep Blue has *the
specific* from-to butterfly shape" — closer in spirit to Viridithas's
piece-to design (also noted in the same donor research, but not flagged as
the actual match).

**Consequence** Every one of the 8 individually-testable history pieces
listed in `DONOR_NEXT_IDEAS.md` (gravity, bonus shape, malus, etc.) is
still valid as an idea, but was implicitly scoped as "layer onto the
existing table shape" — which quietly assumed a from-to axis that isn't
there. `piece[from]` information (which square a piece is escaping FROM,
independent of which piece it is) is exactly what a butterfly table adds
over piece-to, and the donor sequence (`DEEP_BLUE_BLUEPRINT.md`'s "primary
donor roadmap") lists "butterfly history" as its own step before "TT
semantic improvements" for a reason — it's a substrate change, not a
tuning change.

**Action** History decomposition candidate #1 ("butterfly indexing
substrate") tests the from-to-side reindex in isolation, same bonus
mechanism (`+= depth*depth`, unbounded, no malus/gravity) as the current
piece-to table, so the indexing change is measured on its own before any
bonus/malus/gravity change is layered on top of either shape. See
`CLAUDE_REMOTE_RESULTS.md` for the resulting candidate's record.
