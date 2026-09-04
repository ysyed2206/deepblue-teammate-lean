# Deep Blue — remote autonomous session results

Champion: `deepblue.fastsearch4` (unchanged this session). Format per
experiment: ID, parent, donor/reference, code delta, correctness,
equal-depth, timed, playing, bugs found, deviations, recommendation.

See `DEEPBLUE_AUDIT_CORRECTIONS.md` for corrections to prior planning-doc
claims (AC-001: iteration-predictor timing sensitivity; AC-002: P1a dead
branch).

---

## EXP-R01 — fastsearch14 / P1a: preserve TT move on null-best write

**Parent** fastsearch4. **Donor reference** Reckless-family "TT move
preservation" roadmap item (see `DEEP_BLUE_BLUEPRINT.md` donor sequence);
exact donor source not yet pulled this session (see Donor Archaeology
Backlog below).

**Code delta** `deepblue/fastsearch14.py:451-461` — on a same-key TT
write, if `best_move == 0`, copy the existing entry's stored move forward
instead of overwriting it with 0. Plus (this session) two diagnostic
counters `P1A_CANDIDATE`/`P1A_FIRE` added at `fastsearch14.py:56,455-458`
to instrument the branch; no behavioural change.

**Correctness** `tools/regression_fast_variant.py fastsearch14`: 32/32
pass (re-verified after adding instrumentation).

**Equal-depth (depth 7, 5-position suite)** Bit-identical nodes/move/score
to fastsearch4 on all 5 positions (start 285,058; kiwipete 574,210; closed
463,365; tactical 93,517; endgame 49,925).

**Timed (`--movetime-ms 1000 --repeats 9`)** Median depth 7.0, median node
ratio 1.000 vs fastsearch4 — structurally inert, confirmed at higher repeat
count than the original 3-repeat pass.

**Instrumented mechanism check** Added `P1A_CANDIDATE`/`P1A_FIRE` counters
and ran all 5 suite positions at ~1s/move (1.5M+ total nodes):
`P1A_CANDIDATE = 0, P1A_FIRE = 0` on every position. **The guarded branch
never executes.**

**Root cause (see AC-002)** `best_move` is set to the first move examined
at any node (`fastsearch14.py:422-423`, `score > -INFINITY` is true for the
first move tried), so it is never `0` at the TT-write point except after a
zero-move node, which does not reach the write path. The precondition
should gate on the produced bound being `UPPER` (`best <= original_alpha`),
not on `best_move == 0`. This is a port bug relative to donor intent, not a
"maybe rare" situation.

**Playing** Not run at scale — a dead-mechanism candidate is not worth a
paired budget beyond the initial 16-game sanity check already taken
(+6=5-5, 53.1%, indistinguishable from noise given AC-001, and expected
since the code cannot differ from fastsearch4's behaviour by construction).

**Bugs found** The branch's `best_move == 0` guard is unreachable in
normal play (AC-002).

**Deviations from donor** Precondition should be bound-type-based, not
move-nullness-based, per above.

**Recommendation: REJECT fastsearch14 as coded.** Not "low priority" —
conclusively dead code, evidenced by direct instrumentation, not inference.
A corrected version (gate on `best <= original_alpha` instead of
`best_move == 0`) would be a genuinely new, untested candidate — proposed
as `fastsearch18` in the backlog below, not a revival of 14.

---

## EXP-R02 — fastsearch15 / P1b: TT cutoffs restricted to non-PV nodes

**Parent** fastsearch4. **Donor reference** standard PV/non-PV TT-cutoff
split (donor roadmap item "TT semantic improvements"); exact donor source
not yet pulled this session.

**Code delta** Not re-diffed line-by-line this session (see backlog item to
diff precisely against fastsearch4 and confirm PV classification per
CLAUDEREAD's explicit instruction to "audit the exact PV classification
before trusting it" — **not yet done**, flagged below).

**Correctness** 32/32 regression (per CLAUDEREAD prior state; not rerun
this session since no new edits were made to fastsearch15).

**Equal-depth (depth 7)** start 285,058→285,264; kiwipete 574,210→574,360;
closed 463,365→463,411; tactical 93,517→94,188; endgame 49,925→49,961.
Median node ratio 1.001. Best move/score identical on all 5. Confirmed
stable at `--repeats 9` this session (median depth 7.0, ratio 1.001).

**Timed** Same suite/repeats as above — no material change from the
original pass; structurally a very small, non-explosive change.

**Playing — three independent paired runs this session** (all
`--move-ms 80`, colours reversed per opening, no code changes between
runs):

| Run | Openings | Games | Result | Score |
|---|---:|---:|---|---:|
| 1 | 8 | 16 | +6 =6 -4 | 56.2% |
| 2 (repeat of run 1) | 8 | 16 | +5 =6 -5 | 50.0% |
| 3 (new/larger set) | 10 | 20 | +8 =8 -4 | 60.0% |
| **Pooled** | — | **52** | **+19 =20 -13** | **55.8%** |

Run-to-run variance (50.0%–60.0%) is consistent with AC-001's timing-jitter
finding, but unlike fastsearch17 (below), fastsearch15 **never dropped
below 50%** across three independent replays, and the pooled 52-game read
is a consistent mild positive. Zero illegal moves/crashes/flags across all
84 games played this session (14+15+17 combined).

**Bugs found** None new.

**Deviations from donor** Not yet audited — see backlog.

**Recommendation: HOLD → promote to a larger, slower-time-control paired
test next.** This is the strongest of the three candidates this session:
node-identical-ish structure, zero reliability problems, and a pooled
55.8% over 52 games that stayed positive on every independent replay. Not
yet KEEP — 52 games at a fast, jitter-prone time control is still a smoke
gate, and the PV-classification code has not been re-audited against the
donor idea as CLAUDEREAD explicitly required. Do that audit before any
further promotion, then run a paired test at a slower time control (less
exposed to AC-001) with a larger opening set.

---

## EXP-R03 — fastsearch17 / P4: floor depth-0-in-check to depth 1 + MAX_PLY guard

**Parent** fastsearch4 (independent of 14/15). **Donor reference** standard
"don't drop straight into qsearch while in check" guard, ubiquitous across
donor engines; exact source not pulled this session.

**Code delta** Not re-diffed this session (backlog).

**Correctness** 32/32 regression (prior state, not rerun — no edits made).

**Equal-depth (depth 7)** start -0.04%, kiwipete -0.86%, closed +10.0%,
tactical -2.64%, endgame -14.29% node deltas vs fastsearch4. Median node
ratio 0.991 (confirmed at `--repeats 9` this session). Best move/score
identical on all 5 positions. Genuinely mixed per-position structural
effect, as previously noted — no blow-up, no collapse.

**Playing — three independent paired runs this session** (same protocol as
EXP-R02):

| Run | Openings | Games | Result | Score |
|---|---:|---:|---|---:|
| 1 | 8 | 16 | +5 =8 -3 | 56.2% |
| 2 (repeat of run 1) | 8 | 16 | +5 =4 -7 | 43.8% |
| 3 (new/larger set) | 10 | 20 | +6 =7 -7 | 47.5% |
| **Pooled** | — | **52** | **+16 =19 -17** | **49.0%** |

Unlike fastsearch15, this candidate **flips sign** between identical
reruns (56.2% → 43.8%) and the pooled result sits almost exactly at parity.
Given the mixed equal-depth structural signal (closed middlegame +10%
nodes, endgame -14%) already flagged as "mixed but interesting," the
playing evidence now reads as consistent with **no real effect either way**
at this time control, not as a promising candidate being obscured by noise.

**Bugs found** None new. Zero illegal/crash/flag across all games.

**Deviations from donor** Not yet audited — see backlog.

**Recommendation: HOLD, downgraded from "definitely continue."** Pooled
49.0% over 52 games, with a sign flip between identical replays, is the
signature of a near-neutral change rather than a suppressed positive one.
Worth exactly one more test — a slower time control (less exposed to
AC-001's jitter) and/or a differential node-by-node structural audit at the
"closed" position (the +10% outlier) to understand *why* that position
diverges — before deciding REJECT vs. small KEEP. Not a priority over
fastsearch15 or the priority-list items below.

---

## EXP-R04 — fastsearch18 / priority #1: transposition table inside qsearch

**Parent** fastsearch4 (independent of 14/15/17 — not stacked on any of them,
per standing instruction not to stack onto unpromoted candidates).

**Donor reference** Donor roadmap item "qsearch TT" (`DEEP_BLUE_BLUEPRINT.md`
near-term target architecture, `qsearch -> ...`). Generic technique, not tied
to one donor's exact control flow this round — implemented from first
principles against this engine's own TT conventions rather than ported.
Full donor-exact cross-check not yet done (see backlog).

**Code delta** New file `deepblue/fastsearch18.py`. `quiescence()` now takes
the zobrist `value` and the TT arrays, threads the hash incrementally through
its own recursive calls exactly the way `negamax` does (`Z.apply_move` with
the same `canonical_ep_file` before/after bookkeeping — the EP-canonical-hash
invariant is reused, not re-derived). One shared probe near the top of
`quiescence` (EXACT/LOWER/UPPER cutoff, unrestricted by PV like fastsearch4's
own baseline — not combined with fastsearch15's unpromoted PV restriction).
One store at the end of each branch's normal move-loop path, depth stored as
`0`, mirroring fastsearch4's own existing replacement rule
(`tt_depth[index] <= depth or tt_key[index] != value`) with `depth=0`. All of
qsearch's existing shortcut returns (stand-pat cutoff, checkmate/stalemate,
fifty-move draw) are unchanged and do not touch the TT, matching how RFP's
shortcut in `negamax` also never writes an entry. Qsearch move ordering is
deliberately unchanged (no TT-move preference added) to keep this one
isolated idea. Full diff: `git diff --no-index deepblue/fastsearch4.py
deepblue/fastsearch18.py`.

**Correctness** `tools/regression_fast_variant.py fastsearch18`: 32/32 pass
(re-verified twice, including after adding the eviction-instrumentation
counters below).

**Equal-depth (depth 7, exact — `max_depth=7` with a 60s budget so the
AC-001 iteration-predictor jitter cannot fire at all, not a time-boxed
measurement)**:

| Position | fs4 nodes | fs18 nodes | ratio | move/score |
|---|---:|---:|---:|---|
| start | 285,058 | 284,328 | 0.997 | identical (b1c3, +8) |
| kiwipete | 574,210 | 418,233 | **0.728** | identical (d5e6, -33) |
| closed | 463,365 | 446,789 | 0.964 | identical (d4c5, +27) |
| tactical | 93,517 | 58,939 | **0.630** | identical (c5c4, -455) |
| endgame | 49,925 | 48,546 | 0.972 | identical (b4f4, +58) |

Every position: bit-identical best move and score to fastsearch4 at the same
depth. Real, substantial node reductions (up to -37%), concentrated exactly
where EXP-018 (`EXPERIMENTS.md`) already showed quiescence dominates node
count (kiwipete, tactical). This is the strongest structural result of any
candidate this session, and it is immune to the AC-001 timing confound by
construction (fixed depth, generous budget, no early-stop decision involved).

**Mechanism verification (learned from AC-002 — don't trust a plausible
mechanism, instrument it)** Added `QTTPROBE`/`QTTHIT` counters. Qsearch TT
hit rate across the suite: 1.3%–5.2% of probes. Non-zero and structurally
sane (higher in tactical/capture-rich positions, as expected) — this
candidate's mechanism is confirmed live, unlike fastsearch14's.

**Replacement-policy invariant — instrumented directly, not inferred**
(standing instruction: prove overwrite events, don't infer from aggregate
slot counts). Added `QTT_EVICT_REAL` (a qsearch write evicts a *different*
key that was holding a real-search, depth>=1 entry) and
`QTT_EVICT_QSEARCH` (a qsearch write overwrites an older qsearch-authored
entry for the same key). Measured at ~2s/move across the suite:

| Position | nodes | QTT_EVICT_REAL | QTT_EVICT_QSEARCH |
|---|---:|---:|---:|
| start | 284,328 | 126 | 9 |
| kiwipete | 418,233 | 243 | 404 |
| closed | 446,789 | 248 | 111 |
| tactical | 242,799 | 52 | 349 |
| endgame | 266,956 | 148 | 68 |

**Finding, stated precisely**: yes, a qsearch write occasionally evicts a
real-search entry — but only via the same different-key-collision path that
fastsearch4's own `negamax`-to-`negamax` writes already use unconditionally
(`tt_key[index] != value` replaces regardless of relative depth in the
*existing, accepted* champion). This is not a new failure mode introduced by
this candidate; it is the same pre-existing replacement policy, now
exercised more often because qsearch now writes far more frequently than
main search (41,827 qsearch-authored vs 10,206 real-search-authored slots
observed in one full search). The measured rate (52-248 evictions per
100k-450k nodes, ~0.02%-0.1%) is small and did not produce any move/score
divergence at fixed depth on this suite. Recorded as a known, bounded,
quantified cost rather than asserted safe by inference.

**Timed (time-boxed, `--repeats 9`, subject to AC-001 jitter — read only the
median-of-medians line)** Median node ratio 0.984 vs fastsearch4 in the
time-boxed harness; directionally consistent with the clean fixed-depth
result above, though individual per-position rows in that run are not
independently trustworthy (see AC-001).

**Playing (smoke gates only, per standing instruction not to over-invest
here given known AC-001 noise at 80 ms/move)**:

| Run | Games | Result | Score |
|---|---:|---|---:|
| 1 | 16 | +3 =8 -5 | 43.8% |
| 2 (repeat) | 16 | +9 =5 -2 | 71.9% |
| **Pooled** | **32** | **+12 =13 -7** | **57.8%** |

A third, larger (10-opening/20-game) run was attempted twice and both times
the process was killed (exit 143) after being moved to background past the
280s foreground timeout; a smaller 3-opening/6-game smoke run in between
completed normally in well under the timeout with a clean 50.0% result and
zero problems, so this is not evidence of a hang or bug in fastsearch18 —
it is unexplained infrastructure behavior on the two larger background runs
specifically, not diagnosed further per the standing instruction to not
over-invest in stabilizing a fast-TC playing sample once structural evidence
is already sane. Zero illegal moves/crashes/flags across every game played
this session for this candidate.

**Bugs found** None. `QTT_EVICT_REAL` is a measured cost, not a bug (see
above).

**Deviations from donor** Not cross-checked against one specific donor's
exact qsearch-TT control flow this round; implemented from the codebase's
own existing TT conventions instead (see backlog item below for the
donor-exact cross-check, non-blocking).

**Adversarial correctness suite (Agent E, independently re-run and
verified by the coordinator, not just trusted)**: `tests/qsearch_tt_adversarial.py`
(new file, 593 lines) — 32/32 checks pass, covering TT read/write
consistency under dirty-table reuse, mate-distance round-trip across
differing store/read plies, 3 stalemate configurations through both the
stand-pat and normal qsearch paths, promotion move-encoding round-trip
through the TT's `best_move` field (all 4 promotion pieces), canonical/
pinned/check-resolving EP-hash treatment inside qsearch specifically (3
distinct FEN configurations, cross-checked against `fastsearch4` and
against `python-chess`), and the halfmove-clock boundary at 70/85/99/100
against `TT_HALFMOVE_SAFE_LIMIT=80` (including a warm-vs-cold check
proving the TT-disabled window doesn't leak stale entries). Zero failures.

**Independent red-team audit (Agent B, opus) — real corrections, no
HIGH/CRITICAL findings.** Full ablation study (store-only vs probe-only
variants) attributes ALL behavioural divergence from fastsearch4 to the
**probe** (which has no depth condition and can consume a deep negamax
entry from a shallow qnode), not the store/eviction path the coordinator
had focused its own risk analysis on — the store-only variant is identical
to fastsearch4 on every diverging case; a depth-0-restricted probe is too.
Two corrections filed against this session's own claims:

- **Finding A1 (MEDIUM, docstring fixed in `fastsearch18.py`)**: the
  original docstring claimed a qsearch write "can only ever overwrite
  another qsearch entry... or an empty slot" — false, contradicted by the
  coordinator's own measured evictions (the `tt_key[index] != value`
  disjunct replaces a different-key entry unconditionally, same as
  fastsearch4's own negamax-to-negamax writes). Corrected in place.
- **Finding A2 (MEDIUM)**: "bit-identical at fixed depth" does not
  generalise past the 5-position suite. A 57-position x depth-1-7 sweep
  (399 pairs) found 8 divergences, **including one actual best-move
  change**: `8/8/4k3/8/8/3NKN2/8/8 w - - 0 1` at depth 5, fastsearch4 plays
  e3d4 (652), fastsearch18 plays e3e4 (657). The fixed-depth-identity
  argument is not a promotion shortcut here — paired games are the real
  gate, which the project's test philosophy already required regardless.
- **Finding B (LOW, latent, zero observed occurrences)**: theoretically, an
  unguarded-depth probe could narrow the search window using a deep
  entry's bound and then return a fail-soft result the caller reads as
  exact but that is unsound relative to that bound (fastsearch4's negamax
  has the same pattern but bounded by `tt_depth >= depth`; qsearch's probe
  is unbounded). Instrumented over 1.64M qnodes / 9 positions: 7,963
  narrowing events, 7,962 resolved by an immediate, sound cutoff, 1 reached
  the move loop without tainting the eventual TT write. Recorded for the
  record, not a blocker.
- Independently re-confirmed: EP/mate-normalisation/stop-flag/Numba safety
  all CONFIRMED SAFE via targeted ablation and instrumentation (0
  mismatches over 597k qnodes for hash-thread verification, 0/10 drift
  with the stop-guard present vs 1/10 with it stripped, identical mate
  scores across fresh/polluted/re-searched TT states). TT lifetime
  (persists across moves within one engine instance) confirmed identical
  to fastsearch4's own existing behaviour, not fastsearch18-specific.
  Falsification attempt with `TT_BITS` lowered 20→10 (~1000x the collision
  rate): still 32/32 regression, still legal 50-ply games.

**Recommendation: KEEP as the working baseline for further qsearch/ordering
work; not yet promoted to champion, and the promotion gate is paired games
(as always), not the fixed-depth-identity argument this session leaned on
too heavily at first.** fastsearch21 and fastsearch22 (both parented on
fastsearch18) inherit the same probe mechanism and therefore the same A2
caveat — their own "bit-identical at fixed depth on the 5-position suite"
readings should be read with the same qualification. This candidate has the strongest,
cleanest structural evidence of the session: zero move/score divergence at
matched depth across the whole suite, real and substantial node reductions
concentrated exactly where the architecture doc says qsearch dominates, a
verified-live mechanism, and a directly-instrumented (not inferred) bound on
its one measurable side effect. The playing-gate signal is a smoke test only
(consistent with, not contradicting, the structural result) and should not
gate this candidate given AC-001. Next: a donor-exact cross-check (backlog),
then treat as the new parent for priority item #3 (SEE good/bad noisy
partition + qsearch skip), which touches the same code region and should be
built on top of this rather than independently against fastsearch4, per the
"preserve both an independent candidate and a cumulative dependent one"
instruction if/when that stacking happens.

## EXP-R05 — fastsearch19 / priority #7: RFP quiet-TT-move guard A/B

**Parent** fastsearch4 (independent candidate, not stacked on 18).

**Code delta** `deepblue/fastsearch19.py` — removed exactly one condition
from fastsearch4's RFP gate: `not tt_move_is_quiet` (and its supporting
computation). Everything else byte-identical to fastsearch4.

**Correctness** 32/32 regression pass.

**Equal-depth (exact depth 7, `max_depth=7`, no time confound)** Node
counts, best move and score **bit-identical** to fastsearch4 on all 5 suite
positions (ratio 1.0000 everywhere). `RFP_TRY` (attempts) rose on 4/5
positions (e.g. closed: 84,942→86,183) but `RFP_CUT` (successful cutoffs)
stayed **exactly identical** everywhere. I.e. the guard's removal does let
RFP fire more often at nodes with a quiet TT move, but on this suite every
one of those extra attempts still fails the `static_eval - margin >= beta`
margin test and falls through to normal search — so it's inert on this
suite for a structurally different reason than fastsearch14 (that branch
was unreachable by construction; this one is reachable and exercised, it
just never wins here).

**Playing** One 16-game smoke gate only (standing instruction: don't
over-invest in a marginal fast-TC result once structural evidence is
in-hand): +6=3-7, 46.9%. Single data point, not pooled — do not read this
as a verdict given AC-001.

**Recommendation: HOLD.** Structurally inert-on-this-suite is a real,
verified finding (not inferred), but a 5-position suite is too small to
conclude the guard is universally unnecessary — it may matter on positions
outside this suite where a quiet TT move genuinely does flag a hidden
defence. Next step if resumed: instrument how often `tt_move_is_quiet`
actually gates a would-be-successful cutoff (not just an attempt) over a
larger/longer position sample, which would settle this directly rather
than by further paired-game inference.

---

## Donor archaeology / design outputs (from parallel subagents)

**SEE good/bad noisy qsearch partition — full design received, not yet
implemented.** Agent C (sonnet) read `deepblue/fastsearch18.py`,
`deepblue/see.py` (confirmed `see_value`/`see_ge_zero` are both
`@njit(cache=True, nogil=True)`, directly callable from qsearch, signature
`(bb, occ, mail, st, move)`, pre-move state — no make/unmake needed) and
`deepblue/fastcore.py`'s `generate_pseudo_tactical` (confirmed it yields
only captures/EP/capture-promotions/quiet-promotions, never plain quiets,
so no band-collision risk in the non-check qsearch branch). Two
independently-testable candidates specified:

- **Design A (ordering only)**: new `ORDER_CAPTURE_GOOD`/`ORDER_CAPTURE_BAD`
  bands in `order_qmoves`, SEE called only when `count > 1` (skip when
  reordering can't matter) and never on promotions/capture-promotions (SEE
  unvalidated there). Numerically verified non-overlapping with existing
  `ORDER_PROMOTION`/`ORDER_CAPTURE`/killer bands.
- **Design B (bad-noisy skip)**: `SEE < 0` skip inserted ONLY in
  quiescence's non-check-branch move loop (never the in-check/evasion
  loop — flagged as the correctness-critical scoping boundary, since
  skipping a bad-SEE check evasion could miss the only legal reply), after
  the stand-pat cutoff, before `make_move`. Traced through the existing
  `legal_tactical`/`any_legal_move` stalemate fallback and confirmed it
  degrades safely to the already-tested "empty tactical list" path, not a
  new one.
- 9 targeted correctness tests specified before any SPRT-style run,
  including the two scoping-boundary tests (in-check bad-capture-is-only-
  move; not-in-check bad-capture-is-only-move, expect `stand_pat` not
  `DRAW_SCORE`).
- Explicitly recommends testing A and B **independently**, never stacked,
  and building each on `fastsearch18.py` directly (not on each other).

Not yet implemented this session — queued in `BACKLOG.md` NOW/NEXT. Full
design detail is in this session's transcript; re-derive from
`deepblue/see.py` + `deepblue/fastsearch18.py` directly if this file is
being read in a future session without that transcript.

## EXP-R06 — fastsearch20 / NMP_DELTA.md delta #3: null_depth floor fix

**Parent** fastsearch5 (NMP v1, rejected at 40.6%) — deliberately, since
this candidate tests one specific hypothesis about *why* v1 failed, not a
fresh NMP design. See `NMP_DELTA.md` for the full source-grounded delta
analysis against fastsearch5's actual code (not assumption).

**Code delta** One condition added to the NMP trigger gate:
`depth - 1 - NMP_REDUCTION >= 1` (skip NMP rather than clamp `null_depth`
to 0 and silently recurse straight into quiescence). Nothing else changed
from fastsearch5. Added `NMP_SKIPPED_SHALLOW` instrumentation counter.

**Correctness** 32/32 regression pass.

**Equal-depth (exact depth 7)** Best move and score **bit-identical** to
both fastsearch4 and fastsearch5 on all 5 suite positions — the fix does
not change search *correctness* at fixed depth, only which nodes attempt
NMP. Mechanism verified directly (not inferred): `NMP_TRY` collapsed
massively fastsearch5→fastsearch20 (kiwipete 1888→52, start 517→57, closed
998→94, endgame 137→25), and `NMP_SKIPPED_SHALLOW` accounts for nearly all
of that drop (kiwipete 1590, start 355, closed 830, endgame 135) —
confirming the overwhelming majority of fastsearch5's NMP activity was
happening through the now-closed qsearch-collapse path, exactly as
`NMP_DELTA.md` hypothesized. fastsearch5's raw node counts vs fastsearch4
were wildly unstable (kiwipete +104% nodes, i.e. NMP v1 made the search
*worse* than no NMP at all there); fastsearch20 is less unstable
(kiwipete +39%, closed -9%, endgame -21%) but still not a clean win over
fastsearch4 on this small suite — expected, since real NMP payoff shows up
in playing strength more than fixed-depth node count.

**Playing** One 16-game smoke gate: +1=7-8, **28.1%**.

**CORRECTED (independent research pass, Agent E) — the two claims above
this line are wrong, and the playing result is uninformative, not
negative.** Full re-analysis:

1. **"fastsearch5's node counts were wildly unstable" is a
   mis-description.** They are fully deterministic at fixed depth
   (reproduced exactly) and simply position-dependent: fastsearch5 was
   actually a large node *winner* on 3/5 positions (start -60%, closed
   -41%, endgame -40%) and only a loser on kiwipete (+104%) and roughly
   flat on tactical. The qsearch-collapse bug was carrying essentially
   ALL of that -60%/-41%/-40% saving — removing it (fastsearch20) also
   removed the saving, not just the pathology: start -60%→-1.5%, closed
   -41%→-9%, endgame -40%→-21%, while kiwipete only improved from +104%
   to +39% (still net-negative there). fail-high rate on kiwipete
   collapsed from 80.7% (1523/1888) to 32.7% (17/52).
2. **"real NMP payoff shows up in playing strength more than fixed-depth
   node count" inverts the mechanism.** NMP has no channel to IMPROVE
   move quality at fixed depth — it can only prune (degrade or leave
   unchanged), and here it left every move/score identical on all 5
   positions. Its entire value proposition IS fixed-depth node reduction,
   converted into extra depth under a clock. +39% nodes for identical
   answers on kiwipete is pure cost with no offsetting upside to expect
   from a playing test.
3. **The 28.1% smoke gate is not a "notably low" outlier — it's
   unremarkable.** Pooling every 16-game/8-opening/80ms gate recorded this
   session (n=11, values 53.1, 56.2, 50.0, 56.2, 43.8, 43.8, 71.9, 46.9,
   40.6, 43.8, 28.1): mean 48.6%, empirical SD 11.1pp (vs ~7.5pp
   binomial-only — AC-001's jitter roughly doubles the variance, as
   fastsearch18's own 43.8→71.9 identical-rerun swing already showed).
   Expected minimum of 11 draws from N(48.6, 11.1) is ~31% — 28.1% is
   ~1.85 SD below the session mean, approximately what you'd expect the
   *smallest* of eleven neutral results to look like by chance alone.
4. **The decisive finding: the 80ms gate cannot exercise this feature at
   all.** The fix's gate arithmetic is `depth>=NMP_MIN_DEPTH(4) AND
   depth-1-NMP_REDUCTION(3)>=1` → the second clause requires `depth>=5`,
   making `NMP_MIN_DEPTH=4` dead code (the real threshold is 5, not 4).
   Measured directly on 90 real game positions (from champion self-play)
   searched at the gate's exact 80ms/112ms budget: **NMP fired 6 times
   across 90 whole searches, on 2 of 90 positions, producing 1 cutoff.**
   `NMP_SKIPPED_SHALLOW` = 1012 over the same run — essentially every NMP
   opportunity at this time control is the depth-4 case the fix now
   refuses. The 28.1% paired result was measuring a change that is inert
   on ~98% of moves played; whatever produced that number, it was not NMP
   semantics. (Secondary, unconfirmed: fastsearch20 completed shallower
   than fastsearch4 on 12/90 positions vs deeper on 4/90 (binomial
   p≈0.04) — plausibly the extra `NMP_SKIPPED_SHALLOW` diagnostic counter
   and gate check adding pure per-node overhead nudging AC-001's knife-edge
   growth predictor. A candidate carrying diagnostics the baseline lacks
   should not be the binary that plays a paired gate — noted as a
   methodology fix for future candidates.)

**Bugs found** None new (the whole point of this candidate was fixing a
previously-identified one). See EXP-R11 below for a further, donor-sourced
correction to delta #3's own premise.

**Recommendation: superseded — see EXP-R11.** Do not run a second/larger
80ms paired sample of fastsearch20 (point 4 above shows this would remain
uninformative regardless of sample size). Do not implement `NMP_DELTA.md`
deltas #1/#2 on top of this candidate specifically (they would be provably
inert given how rarely NMP fires here, repeating the fastsearch14/
fastsearch19 "test a mechanism that barely executes" mistake).

## EXP-R07 — fastsearch21 / priority #3 Design B: SEE bad-noisy qsearch skip

**Parent** fastsearch18 (qsearch TT), deliberately — this tests the skip on
top of Deep Blue's own already-landed qsearch TT rather than assuming
Pawnstar's bundled SPRT (skip + qsearch-TT-*removal*) transfers.

**Code delta** One insertion, in quiescence's non-check-branch move loop
only (never the checked/evasion loop, which is a structurally separate
loop and untouched): a pseudo capture with `see_value(...) < 0` is skipped
before `make_move`, excluding promotions per `see.py`'s own documented
constraint. Full design from Agent C (see donor-archaeology section above)
implemented as specified, including the correctness-boundary reasoning
about the `legal_tactical`/`any_legal_move` stalemate fallback.

**Correctness** 32/32 regression pass.

**Equal-depth (exact depth 7)** Best move and score **bit-identical** to
fastsearch18 on all 5 suite positions, with real additional node
reductions on top of fastsearch18's own reduction vs fastsearch4:
kiwipete -17.0%, tactical -30.7%, closed -9.8%, start -2.1%, endgame -1.1%.
`QSEE_SKIP` mechanism confirmed heavily active (407-16,974 skips per
search across the suite) with zero move/score divergence on this suite.

**Playing** One 16-game smoke gate: +5=3-8, 40.6%. Mildly negative, well
within the AC-001 noise band already established (fastsearch18 itself
swung 43.8%-71.9% across identical reruns) — not read as a verdict.

**Recommendation: HOLD, promising structural evidence, needs a second
playing sample before further investment.** Zero correctness divergence at
fixed depth despite aggressive pruning (up to ~17k skipped moves in one
search) is a strong signal the correctness-boundary reasoning (checked
branch untouched, stalemate fallback unaffected) holds in practice, not
just in theory. The one playing data point is inconclusive on its own.
Next: a second paired sample, and/or the Design A (SEE ordering, no
skipping) candidate as an independent comparison point, per Agent C's
design and the "test A and B independently" instruction — not yet built
this session.

## EXP-R08 — fastsearch22 / priority #3 Design A: SEE good/bad qsearch ordering

**Parent** fastsearch18. Independent of fastsearch21 (Design B) — not
stacked, per Agent C's design and the one-idea-per-candidate rule.

**Code delta** `order_qmoves` now takes `(bb, occ, mail, st, ...)` and
splits captures into `ORDER_CAPTURE_GOOD` (== unchanged `ORDER_CAPTURE`,
SEE>=0) and `ORDER_CAPTURE_BAD` (new, lower band, `1<<19`) for SEE<0,
skipping the SEE call entirely for promotions/capture-promotions (per
`see.py`'s own documented constraint) and when `count<=1` (nothing to
reorder). Two module-level `assert`s verify the new band never overlaps
`ORDER_CAPTURE_GOOD`/`ORDER_PROMOTION` given the MVV-LVA tie-break term's
actual bound (`PIECE_VALUE` max 20) — both pass at import time. Ordering
only, no move is ever skipped.

**Correctness** 32/32 regression pass.

**Equal-depth (exact depth 7)** Best move and score identical to
fastsearch18 on all 5 suite positions (subject to the same EXP-R04 Finding
A2 caveat — this reading doesn't necessarily generalise past this suite
either, not independently swept this round), with modest additional node
reductions on top of fastsearch18's own: kiwipete -3.25%, closed -3.11%,
tactical -6.22%, start/endgame ~flat. Smaller effect than Design B
(fastsearch21), as expected — ordering-only changes move order, not move
count, so its benefit is indirect (better cutoffs from trying good
captures first) rather than direct (fewer moves searched).

**Playing** One 16-game smoke gate: +4=6-6, 43.8%. Within the established
noise band, inconclusive on its own.

**Recommendation: HOLD.** Structurally sound (asserts pass, correctness
clean, small positive node effect), but the effect size is small enough on
this suite that a single smoke gate can't distinguish it from noise. Lower
priority than fastsearch21/Design B for further investment given the
smaller measured effect, unless donor evidence for ordering specifically
(vs skipping) turns up separately.

## EXP-R09 — fastsearch24 / history decomposition #1: butterfly indexing substrate

**Parent** fastsearch18 (qsearch TT), per current session direction to use
it as the experimental foundation; qsearch-TT attribution preserved in
the file's own docstring.

**Correction driving this candidate**: AC-003 found Deep Blue's history
table was never actually a from-to butterfly table — `history[piece,
to_square]`, shape (12,64), no from-square axis. A prior donor-research
pass had wrongly claimed it already matched the donor "butterfly"
convention. This candidate tests the indexing correction in total
isolation from any bonus/malus/gravity change.

**Code delta** History table reshaped `(12,64)` piece-to → `(2,64,64)`
side x from x to. `side` derived from the existing piece encoding
(`piece // 6`) rather than a new parameter. Both the read site
(`order_moves`) and write site (cutoff bonus application) updated. Bonus
mechanism itself is byte-identical to fastsearch18: `+= depth*depth`,
unbounded, no malus, no gravity — that's deliberately deferred to
candidates #2/#3.

**Correctness** 32/32 regression pass.

**Equal-depth (exact depth 7)** Best move/score identical to fastsearch18
on all 5 suite positions. Node deltas small and mixed (start -0.32%,
kiwipete +0.00%, closed +0.74%, tactical +0.11%, endgame +7.16%) — expected
and structurally sane: reindexing changes which quiet moves accumulate
credit together (previously any piece-type moving to square X shared one
counter; now each side's specific from→to pair has its own), so tie-breaks
among quiets shift slightly without changing the actual best line at this
depth.

**Playing** One 16-game smoke gate vs fastsearch18 (not fastsearch4, since
this candidate's comparison of interest is the indexing change itself):
+7=4-5, 56.2%.

**Recommendation: HOLD, proceed to candidates #2/#3 regardless.** Per this
session's explicit standing instruction not to over-invest in single
smoke-gate results, and since this candidate exists primarily to correct
an architectural misconception (AC-003) and provide a proper substrate for
future history-dependent LMR work (donor research confirms LMR-5 needs a
bounded/scaled history table regardless of table shape), it is being KEPT
as the foundation for candidates #2/#3 rather than gated purely on this one
playing sample.

---

## EXP-R10 — fastsearch21+22 combination analysis (Agent D): correctly NOT built

**Question investigated**: should fastsearch21 (SEE bad-noisy skip) and
fastsearch22 (SEE good/bad ordering) be combined into one candidate?

**Finding, with direct measurement (scratch-only instrumented variant, not
committed to the repo)**: a naive stack of both mechanisms exactly as
designed is close to node-count-neutral relative to fastsearch21 alone
(0.003%-0.19% node deltas across the suite, 0/5 best-move/score
divergence) — but costs 28%-114% MORE wall-clock time on 4/5 positions,
because fastsearch22's `order_qmoves` computes `see_value()` once per
capture to assign an ordering band, and fastsearch21's loop-level skip
computes `see_value()` again on the same move — every shared capture gets
SEE evaluated twice for a combination whose actual search tree barely
changes. A playing-strength test of this naive combination would measure
CPU overhead from duplicate SEE calls, not "does combining skip+order
help," making any result unattributable.

**Secondary finding**: fastsearch22's SEE-based ordering change lives in
the shared `order_qmoves` function, which is called from BOTH quiescence
branches — so unlike fastsearch21 (whose skip is scoped only to the
non-check branch, by design, with the checked/evasion branch untouched), a
naive combination would silently reorder check-evasion captures by SEE
too, a scope Design B's own docstring explicitly claims stays untouched.
This is exactly the kind of interaction the "don't assume additive"
instruction was meant to catch.

**Recommendation: keep fastsearch21 and fastsearch22 as separate HOLD
candidates (unchanged from EXP-R07/EXP-R08). Do not build a naive stacked
candidate.** If SEE-skip + SEE-order is revisited later, the productive
version is a single-SEE-call redesign (order_qmoves stores the good/bad
band in the scores array; the loop's skip check reads that stored band
instead of recomputing see_value) — a genuinely new, third design, not a
combination of the other two as-is, requiring its own correctness pass.

## EXP-R11 — fastsearch25 / history decomposition #2: symmetric malus

**Parent** fastsearch18, independent of fastsearch24 (built on fastsearch18's
ORIGINAL piece-to (12,64) table, not stacked on 24's reindex).

**Code delta** New preallocated per-ply scratch buffer `tried_quiets`
(no per-node heap allocation) records every quiet move made and searched
at a node. On a cutoff by a quiet move, the existing bonus (`+=
depth*depth`) still applies to the cutting move; malus (`-=depth*depth`,
exact negation, matching Pawnstar/Ethereal's symmetric 1.0x donor pattern)
now applies to every OTHER quiet move already tried at that node. A node
that never cuts gets no update at all (unchanged from fastsearch18). No
decoupling, no gravity/clamping, no width-damping — those are separate,
later candidates per the decomposition list in `DONOR_NEXT_IDEAS.md`.

**Correctness** 32/32 regression pass.

**Equal-depth (exact depth 7)** Best move/score identical to fastsearch18
on all 5 positions. Node deltas small and mixed (start +4.2%, kiwipete
+0.07%, closed +3.0%, tactical +0.00%, endgame -6.6%) — consistent with a
reordering-only structural effect at this depth, as expected (malus only
changes which quiets win ties in future move ordering, not move legality
or search completeness).

**Playing** One 16-game smoke gate vs fastsearch18: +6=6-4, 56.2%.

**Recommendation: HOLD, proceed to candidate #3.** Structurally sound,
no correctness issues, mild positive single-sample smoke result. Per
standing instruction, not over-invested in stabilizing this single sample
before moving to the next independent candidate.

---

## EXP-R12 — NMP delta #3 corrected: Coda uses clamp-not-skip, not fastsearch20's approach

**What Agent E's fresh donor re-verification found (re-fetched directly
from Stockfish/Ethereal/Viridithas/Reckless/Coda source, not reused from
`NMP_DELTA.md`)**: `NMP_DELTA.md`'s original framing — "a null probe
dropping into quiescence is a bug, per one Coda commit" — does not survive
contact with the donors as a general principle. **Four of five donors
examined do exactly what fastsearch5 did, more aggressively**: Stockfish
has no min-depth gate at all and `R=7+depth/3`, so `depth-R<1` for every
depth below ~12 (its null "probe" IS a quiescence search across most of
its own tree); Ethereal, Reckless and Viridithas hit the same qsearch-drop
below depth ~7-8. fastsearch5 hit it at exactly one depth (4); these
donors hit it constantly, by design.

**The one donor that avoids it (Coda) uses the OPPOSITE remedy from
fastsearch20**: Coda's actual line is `if depth - r < 1 { r = depth - 1 }`
— it clamps the REDUCTION so the null probe becomes depth-1, and NMP still
fires. fastsearch20 instead skips NMP entirely whenever the floor would
bind. `NMP_DELTA.md` had offered both options ("fix the floor... or raise
NMP_MIN_DEPTH") without a donor citation for which is real; fastsearch20
picked the one with zero donor precedent, which — per EXP-R06's
correction above — is also demonstrably why fastsearch20 barely fires at
Deep Blue's actual search depths (NMP's whole value proposition assumes
deep search; every donor's non-degenerate NMP begins around depth 7-12,
depths this engine barely reaches at 80ms and only reaches briefly even at
1-2s).

**Next NMP experiment, per Agent E's explicit recommendation, implemented
this round as `fastsearch23`** (see below) — a one-line change from
fastsearch20: clamp the reduction (Coda's actual mechanism) instead of
skipping. Gate primarily on fixed-depth node counts (the only channel NMP
can act through, per EXP-R06's point 2), not an 80ms paired gate (proven
uninformative for this feature in EXP-R06 point 4). Per Agent E's explicit
recommendation, this is intended as the LAST NMP candidate before parking
the axis if it doesn't recover fastsearch5's fixed-depth savings without
its kiwipete blow-up — NMP has now consumed three candidate-rounds
(fastsearch5, fastsearch20, this one) without a positive signal, and there
is a structural reason (search-depth mismatch with donor calibration) to
expect that to continue.

## EXP-R13 — fastsearch23 / Coda-style NMP clamp: mixed result, NMP axis shelved

**Parent** fastsearch20 (which parents fastsearch5). Implements the
donor-correct remedy identified in EXP-R12: clamp `null_depth` to a floor
of 1 (Coda's actual mechanism) instead of skipping NMP whenever the floor
would bind (fastsearch20's undertested choice). Gate: fixed-depth node
counts only, per Agent E's explicit protocol — NMP cannot improve move
quality at fixed depth, only prune, so a fixed-depth comparison is the
correct (and only informative) channel.

**Correctness** 32/32 regression pass.

**Equal-depth (exact depth 7), node ratio vs fastsearch4**:

| Position | fastsearch5 | fastsearch20 | fastsearch23 |
|---|---:|---:|---:|
| start | 0.396 | 0.985 | **0.594** |
| kiwipete | 2.040 | 1.393 | **1.776** |
| closed | 0.592 | 0.906 | **0.746** |
| tactical | 1.127 | 1.124 | 1.125 |
| endgame | 0.605 | 0.792 | **0.639** |

Best move/score identical to fastsearch4 on all 5 positions for
fastsearch23 (as for fastsearch5/20).

**Finding, stated precisely — this is genuinely mixed, not a clean win**:
On 3/5 positions (start, closed, endgame) the clamp fix recovers most of
fastsearch5's raw savings that fastsearch20's skip had thrown away,
exactly as EXP-R12 predicted. On tactical, all three are indistinguishable
(NMP barely engages there regardless of remedy). **On kiwipete, the clamp
fix is WORSE than the skip fix** (1.776 vs 1.393 node ratio, both worse
than fastsearch4) and nearly as bad as fastsearch5's original blow-up
(2.040). Instrumented directly: fastsearch23's kiwipete fail-high rate is
1526/1888 = **80.8%** — statistically identical to fastsearch5's original
80.7% (1523/1888) — meaning kiwipete's pathology was NEVER primarily about
the qsearch-collapse mechanism (EXP-R06/EXP-R12's theory); it is NMP
itself being unprofitable at that position even with a sound depth-1
probe, at fastsearch5's fixed `R=3`, with no verification search and no
eval-margin scaling on the reduction.

**Recommendation: shelve the NMP axis for this session, per Agent E's own
explicit decision criterion** ("if it does not [recover the savings
without the blow-up], the axis is dead... a genuinely useful negative
result"). Three candidate-rounds (fastsearch5, fastsearch20, fastsearch23)
have now been spent on this axis without a clean positive result, and
there is a structural reason to expect this to continue: Agent E's donor
comparison (EXP-R12) found every donor's non-degenerate NMP is calibrated
for 20+ ply searches (Stockfish R=7+depth/3, Coda min-depth 6), while Deep
Blue's own fixed-depth suite tops out around depth 7-9 even with generous
time budgets. Do NOT implement `NMP_DELTA.md` deltas #1/#2 (TT-consistency
guard, verification search) — they would need to prove themselves on top
of a mechanism that is not yet net-positive even in its simplest form.
Revisit NMP only if/when a much deeper search (from other landed
improvements — history, SEE, LMR) changes this depth-mismatch premise, or
with fresh donor-sourced adaptive-R/eval-margin evidence specifically
targeted at shallower search depths.

## EXP-R14 — fastsearch26 / history decomposition #3: bounded gravity + donor bonus curve

**Parent** fastsearch18, independent of fastsearch24/25 (original piece-to
table, not stacked on either). Deliberately combines two sub-pieces
(gravity + bonus curve) into one candidate per explicit session naming —
noted as a small deviation from strict one-idea-per-candidate, flagged so
a future reader knows to split them if this result is ever revisited.

**Code delta** `MAX_HISTORY=16384` (majority donor constant), bonus capped
at `min(depth*depth, 400)` (Pawnstar's curve, chosen as the smallest
change from the existing raw `depth*depth`), gravity applied on write:
`entry + bonus - entry*bonus//MAX_HISTORY`, with a defensive `±MAX_HISTORY`
clamp added on top (belt-and-suspenders against Numba integer-truncation
edge cases, matching Viridithas's own defensive clamp). No malus (that's
candidate #2, tested independently).

**Correctness** 32/32 regression pass.

**Equal-depth (exact depth 7)** **Bit-identical to fastsearch18 on all 5
positions — not just move/score, but node count too (ratio 1.0000
everywhere).** Mechanistically explained, not just observed: at depth≤7,
`depth*depth` maxes at 49, far below the 400 cap, so the bonus-curve half
of this candidate is a complete no-op on this suite by construction (the
cap only binds at depth≥20). The gravity half is technically live from
the first update onward, but its damping term
(`entry*bonus//MAX_HISTORY`) is tiny at these bonus/entry magnitudes
(e.g. entry=1000, bonus=49 → damping=2, ~4% of the bonus) and evidently
never flipped a move-ordering tie-break on this particular 5-position
sample.

**Playing** One 16-game smoke gate vs fastsearch18: +3=9-4, 46.9%.

**Meta-observation, worth recording**: this is the THIRD candidate this
session (after fastsearch14's dead branch and fastsearch19's live-but-
never-winning branch) to be correctly implemented, verifiably live, and
still structurally inert on the standard 5-position/depth-7 suite. Taken
together, this suggests the standard suite may be systematically
under-powered for detecting subtle history/ordering-mechanism effects
specifically (as opposed to pruning/TT changes, which it has caught
cleanly every time — fastsearch17, 18, 21 all showed real effects on it).
Future history/ordering candidates should probably be evaluated with a
broader position sweep or a longer time control from the start, rather
than defaulting to this suite the way pruning/search candidates have been.

**Recommendation: HOLD, inert-on-standard-suite (mechanistically explained,
not just observed).** Not rejected — the mechanism is real and live, just
not exercised strongly enough at these depths/bonus magnitudes to show an
effect on this particular sample. Do not build LMR-5 (history-scaled
reduction, which needs a bounded/scaled table) on top of this candidate
specifically without first confirming the gravity/cap actually produces
different, less-saturated history values than fastsearch18's raw
accumulation over a LONGER search (deeper than depth 7) or a real game's
worth of moves, where cumulative bonus totals would plausibly exceed 16384
under the old unbounded scheme and gravity's bounding would start to
matter for real.

## EXP-R15 — Agent C's fixed-depth paired-game tool: validated and exercised

**Tool** `tools/paired_fast_variants_fixed.py` (Agent C, round 2). Reviewed
in full by the coordinator (not just trusted): correctly reuses
`paired_fast_variants.py`'s OPENINGS/loader without duplication, disables
both AC-001 time-based early-stop checks via an astronomically large
`soft_ms` so the loop is bounded purely by `max_depth`, uses `hard_ms` only
as a documented safety valve (flagged, not silently absorbed, when it
binds), and includes a SHA1 move-sequence digest per game for determinism
verification. Design reasoning for NOT implementing fixed-node mode
(would require adding a node-budget stop condition inside the njit search
loop itself — out of scope, "do not modify competition search logic
merely for the test") is sound and independently agreed with.

**Exercised**: fastsearch21 vs fastsearch4, fixed depth 6, 4 openings (8
games): +2=5-1, **56.2%**, zero illegal moves, 128.5M total nodes, one
safety-cap trigger (a French-defense game as White hit the 120s/move
safety valve before completing depth 6, min_completed_depth=1 — correctly
flagged in the output rather than silently returned as a clean depth-6
result; not investigated further this round, worth a look if fastsearch21
is pursued further). This is a substantially cleaner signal than any 80ms
smoke gate this session — deterministic given identical code (repeatable
via the printed digests), immune to AC-001 by construction.

**Recommendation**: use this tool as the default development-screening
gate going forward, reserving `tools/paired_fast_variants.py` (equal-time)
for final promotion confirmation only, per the tool's own stated purpose.

## EXP-R16 — fastsearch27 / priority #4: LMR-1 (bare, exempt-style late move reductions)

**Parent** fastsearch18 (this session's clean working baseline), deliberately
not stacked on the still-HOLD SEE/history candidates.

**Code delta** In negamax's non-first-move PVS branch: a quiet,
non-TT move at `depth>=3` that is at least the 4th legal move examined,
with the current node not in check, gets a null-window probe at
`depth-1-1` (fixed R=1) instead of `depth-1`. If that reduced probe fails
high (`score>alpha`), re-verify at full depth, same null window, before
falling through to the existing (unchanged) full-window PVS re-search.
`LMR_TRY`/`LMR_RESEARCH` instrumentation added. Full donor citation trail
(Reckless 2023-era `87d06bf4`+`dd2acc64`) in the file's own docstring.

**Correctness** 32/32 regression pass.

**Equal-depth (exact depth 7)** Substantial node reductions across the
whole suite: start 0.196x, kiwipete 0.530x, closed 0.350x, tactical
0.759x, endgame 0.320x — 20%-76% of fastsearch18's node count for the
same nominal depth. Best move identical on all 5 positions. Score
identical on 4/5; **endgame shows a score-only divergence with the SAME
best move** (58 -> 144) — this is the expected LMR trade-off (a
reduced-then-not-re-verified subtree returning a less precise value), not
a move-choice error or a correctness bug; regression suite (which
includes rule/mate/stalemate correctness checks, not just best-move
matching) still passes 32/32. Re-search rate very low across the board
(0.0%-0.4%) — the large majority of reduced probes correctly stay
reduced rather than needing the expensive fail-high re-verification,
a healthy calibration signal (an R that was too aggressive would show a
much higher re-search rate).

**Playing** Using Agent C's new deterministic fixed-depth tool
(`tools/paired_fast_variants_fixed.py`, validated in EXP-R15) rather than
the noisy 80ms gate — pending at time of writing, see follow-up entry or
BACKLOG.md for the result once the run completes.

**Recommendation: promising pending the playing result — the strongest
node-reduction magnitude of any candidate this session, with clean
correctness and a well-calibrated (low) re-search rate.** If the fixed-depth
playing signal is positive or neutral, this is a strong candidate to KEEP
as the new working foundation for LMR-2 through LMR-9 (see
DONOR_NEXT_IDEAS.md's dependency chain) and eventually LMP.

## EXP-R17 — fastsearch15 PV-classification re-audit (finally done, 2 rounds overdue)

**What was checked** `diff deepblue/fastsearch4.py deepblue/fastsearch15.py`
read directly (not summarized from memory). The entire change: `is_pv =
beta - original_alpha > 1` is computed once, using `original_alpha`
(captured before any TT-bound narrowing touches `alpha`), and the existing
TT-bound cutoff (`if tt_depth[index] >= depth and ply > 0: ...`) gains a
`not is_pv and` prefix. `tt_move = tt_move_arr[index]` (used for move
ordering) remains outside that gate, unconditional.

**Verdict: the classification is correct.** `beta - alpha > 1` is the
standard, textbook way to detect a full/wide-window (PV) search in a
fail-soft negamax+PVS implementation: a PVS null/scout-window probe is
always called with `-alpha-1, -alpha`, giving `beta - alpha == 1` exactly,
so `> 1` correctly and exclusively identifies non-scout (i.e. PV) calls —
matches Stockfish's own `PvNode = (beta - alpha > 1)` convention and
donor consensus generally. Checked and confirmed: (a) `original_alpha` is
captured before any mutation, so the classification isn't accidentally
computed against an already-narrowed window; (b) the gate scopes exactly
what the design claims — TT move ordering stays available at PV nodes,
only the score/bound-based early-return is blocked there; (c) the
pre-existing `ply > 0` conjunct (present in fastsearch4 too, not
introduced by fastsearch15) is dead code in practice since `negamax` is
never called with `ply=0` (only `search_root`'s own separate loop uses
ply 0) — a pre-existing minor redundancy, not a fastsearch15-specific
issue, out of scope for this audit.

**No bug found.** This removes the last blocker CLAUDEREAD.md flagged for
fastsearch15 ("audit the exact PV classification before trusting it").
Combined with its existing evidence (55.8% pooled over 52 games across 3
independent replays, never negative, node-neutral structurally — see
EXP-R02), fastsearch15 is now the most fully-vetted HOLD candidate this
session. Recommend a fixed-depth confirmation run (Agent C's tool, once
CPU load allows) as the next step before considering promotion.

## EXP-R18 — fastsearch17 donor cross-check (also finally done)

**What was checked** `diff deepblue/fastsearch4.py deepblue/fastsearch17.py`
read directly. Two changes, bundled (a mild one-idea-per-candidate
deviation, but they're not really independent — see below): (1) a
`ply >= MAX_PLY - 2` safety guard at the very top of `negamax`, returning
a static eval before any ply-indexed array is touched; (2) `checked` is
computed earlier (before the depth<=0 check instead of after), and
`if depth <= 0 and not checked: return quiescence(...)` / `if depth <= 0:
depth = 1` replaces the old unconditional `if depth <= 0: return
quiescence(...)`.

**Verdict: correct, and the pairing is sensible, not arbitrary.** This is
horizon check extension — a standard, near-universal alpha-beta technique,
and explicitly named as its own step ("depth-zero check handling") in this
project's own `DEEP_BLUE_BLUEPRINT.md` donor roadmap. The rationale is
search-quality, not raw correctness: fastsearch4's quiescence already
handles being-in-check-at-depth-0 *safely* (its own `checked` branch
generates every legal evasion, not just captures — no missed-mate risk),
but that branch has no TT probing (pre-fastsearch18), no killers/history
ordering, and none of full negamax's pruning machinery — so a checked
horizon node gets meaningfully weaker search discipline than one more real
ply would give it. Flooring depth to 1 instead of dropping to qsearch
fixes exactly that, at the cost of a worst-case unbounded ply-growth risk
in a long forced-check sequence (each such node re-floors to depth 1 and
recurses) — which is precisely what the paired MAX_PLY guard bounds. The
two changes are a matched pair (mechanism + its own safety net), not two
unrelated ideas bundled by convenience.

Traced the code path itself: after flooring, `depth=1` is a real depth
value, not a sentinel — RFP is correctly skipped (already gated on
`not checked`, and `checked=True` is exactly why the floor fired), the
move loop generates and orders normally, children recurse at `depth-1=0`
and correctly resume the ordinary depth<=0 path, and any eventual TT write
stores `tt_depth[index]=1`, self-consistent with a genuine depth-1 result
for future probes. No bug found.

**This resolves the other outstanding BACKLOG item.** Combined with
fastsearch17's existing evidence (pooled 49.0% over 52 games, sign-flipped
between replays — see EXP-R03, still genuinely near-neutral, not
resolved by this code audit since the audit found no bug to explain a
skew either direction), the code-correctness question is now closed; the
playing-strength question remains open and unresolved, consistent with
its existing HOLD status.

## Session methodology note

All three candidates' paired gates were run **sequentially**, not in
parallel, and CPU load was checked before starting (8% load, no visible
python/training processes on this machine at session start) per the
"avoid CPU-heavy parallel benchmarks while NNUE trains" instruction.
`nnue_lab/**` was not read from or written to at any point this session.

## Donor Archaeology Backlog (not yet done this session)

Flagged rather than silently skipped, per "verify every claim" — these are
explicitly required by CLAUDEREAD.md and not yet completed:

1. Diff fastsearch15 against fastsearch4 precisely and re-audit its PV
   classification against actual donor semantics (explicitly requested,
   not done).
2. Diff fastsearch17 against fastsearch4 precisely and pull the actual
   donor check-eviction-guard implementation (Pawnstar/Coda/Reckless) to
   compare exact control flow, not just the general idea.
3. Pull actual fastsearch5 source + historical Reckless NMP commit and
   produce `NMP_DELTA.md` before any NMP v2 work (explicit instruction,
   blocking any future NMP work).
4. A corrected P1a candidate (`fastsearch18`?) gating on `best <=
   original_alpha` instead of `best_move == 0`, if judged worth testing
   after the priority-list items below.

## Recommendation for next work (priority list order)

1. TT inside qsearch (priority #1) — not started this session; next up.
2. Donor-exact history decomposition — blocked on donor archaeology.
3. SEE good/bad noisy partition + qsearch skip.
4. LMR — only after ordering is healthy.
