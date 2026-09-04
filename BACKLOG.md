# Deep Blue — rolling backlog

Updated by the coordinator as work progresses. See `CLAUDE_REMOTE_RESULTS.md`
for full experiment records and `DEEPBLUE_AUDIT_CORRECTIONS.md` for
corrections to prior claims. `DONOR_NEXT_IDEAS.md` holds raw donor-mining
output before it's triaged into this file.

## CPU load note (session-critical, read before running any benchmark)

NNUE training is confirmed ACTIVELY RUNNING as of this point in the
session (`nnue_lab.production.train`, PID chain from 20:32:23, plus 9+
parallel `nnue_lab.production.download_window` shard downloads and a
`mypy` check on nnue_lab files, one process with 5,516 accumulated
CPU-seconds) — CPU load measured at 97%. This explains several recent
paired-game/fixed-depth benchmark attempts timing out or running far
slower than earlier in the session (e.g. fastsearch27's fixed-depth-6
playing gate did not complete in two attempts where fastsearch21's
equivalent test had finished cleanly in ~325s earlier) — CPU contention
with NNUE, not a tooling bug. `nnue_lab/**` was NOT touched, read, or
interfered with in diagnosing this; only read-only process listing was
used. Two orphaned scratch scripts (`p1_taint.py`, `p6_rep.py`, both in
the session scratchpad, both from a red-team subagent that died mid-run
to external API errors) were also spotted running but are a minor
contributor and were left alone rather than risk touching the wrong
process during active NNUE training.

**Action**: CPU-heavy benchmarks (paired games, either the 80ms arena or
the new fixed-depth tool) are paused until load drops. Fixed-depth
STRUCTURAL results (node counts, correctness, move/score comparison) are
NOT affected by CPU contention — they're deterministic and CPU-speed-
independent, just slower to obtain wall-clock-wise — so fastsearch27's
existing fixed-depth-7 evidence (EXP-R16) stands as reliable evidence.
Only the *playing*-gate confirmation for fastsearch27 is deferred.

## NOW (round 2 complete except red-team; round 3 not started)

Round 2 status, all DONE: donor archaeology for history->LMR->LMP (Agent
A, rich commit-level Reckless evidence, DONOR_NEXT_IDEAS.md); SEE21/22
combination analysis (Agent D, correctly declined to build a naive stack
— EXP-R10); deterministic paired-test-mode tool built
(`tools/paired_fast_variants_fixed.py` by Agent C — exists on disk,
**not yet reviewed or exercised by the coordinator**); NMP research +
corrected EXP-R06 (Agent E, produced EXP-R12/R13); fastsearch23 (NMP
clamp-not-skip, PARK the axis — EXP-R13); fastsearch24 (butterfly
indexing, HOLD — EXP-R09); fastsearch25 (symmetric malus, HOLD — EXP-R11);
fastsearch26 (bounded gravity + donor bonus curve, HOLD, inert-on-suite —
EXP-R14).

**Outstanding**: red-team audit of fastsearch18+21 failed twice this round
due to external API errors (ENOTFOUND, then rate-limit — not a task
problem). Relaunch next. Agent C's new fixed-depth paired-game tool has
not yet been reviewed/used by the coordinator — do that before round 3's
LMR work, since LMR-1 is the next natural candidate per the donor roadmap
(priority #4, "LMR only after ordering is healthy") and the fixed-depth
tool is exactly the cleaner evidence channel this project needs for it.

Prior round (round 1) recap: fastsearch18 (qsearch TT, KEEP-as-baseline),
fastsearch19 (RFP guard A/B, inert-on-suite), fastsearch20 (NMP delta #3
v1, superseded by fastsearch23), fastsearch21 (SEE skip, HOLD-promising),
fastsearch22 (SEE ordering, HOLD).

## NEXT

- **LMR-1** (priority #4, "LMR only after ordering is healthy" —
  arguably still blocked on history decomposition proving itself, see
  below, but the donor mechanics are fully specified): bare LMR, exempt
  style — `depth>=3, move_count>=4, quiet-only, not-checked, not-tt-move`,
  fixed `R=1`, null-window-then-full-window re-search (get this ordering
  right — donor's #1 catastrophic-failure pattern is re-searching at full
  window instead of null window first). Full dependency chain (LMR-1
  through LMR-9) in `DONOR_NEXT_IDEAS.md`. Use Agent C's new
  `tools/paired_fast_variants_fixed.py` for cleaner evidence once reviewed.
- **LMP-1 and LMP-7** (priority #5, independent of LMR per donor's own
  final verdict — do not couple them): LMP-1 bare quiet LMP
  (`move_count >= 3+depth*depth`), LMP-7 qsearch movecount cutoff
  (`move_count>=3: break`, fully independent, testable immediately without
  waiting on anything else). Both specified in `DONOR_NEXT_IDEAS.md`.
- **SEE good/bad noisy qsearch partition** (priority #3): Design A/B both
  done (fastsearch21/22, both HOLD). Do not build a naive combination
  (EXP-R10 found it's an unattributable double-SEE-cost result) — if
  revisited, needs the single-SEE-call redesign Agent D specified instead.
- **History decomposition #4-8** (priority #2): pieces 1-3 done
  (fastsearch24/25/26, all HOLD, all inert-on-standard-suite per EXP-R14's
  meta-observation — consider a broader position sweep or longer TC for
  future history candidates before assuming inertness generalizes).
  Remaining: malus decoupling, malus width-damping, killer/history band
  clamp, persistence-vs-reset. Lower priority than LMR/LMP now that the
  substrate exists.
- Review/exercise Agent C's `tools/paired_fast_variants_fixed.py` before
  relying on it for LMR/LMP evidence. DONE — see EXP-R15, validated and
  exercised.
- ~~fastsearch15 PV-classification re-audit~~ DONE, see EXP-R17: no bug
  found, classification correct. Ready for a fixed-depth confirmation run
  once CPU load allows.
- ~~fastsearch17 donor cross-check~~ DONE, see EXP-R18: no bug found,
  matches standard horizon-check-extension practice. Playing-strength
  question remains open (near-neutral, unresolved by this code audit).
- **fastsearch27 (LMR-1)**: fixed-depth-7 structural evidence is strong
  (EXP-R16) but the fixed-depth playing-gate confirmation did not complete
  (CPU contention with active NNUE training — see the CPU load note
  above). Retry once load drops.

## LATER (Tier 2)

- Aspiration windows (donor-mine Reckless/Coda evolution first).
- Correction history (start with the simplest single table, not all at
  once — explicit standing instruction).
- Capture history (only after main history stable; avoid Patch07's
  history-sum collision).
- Continuation history (only after main history works; smallest useful
  context first).
- Staged move picker (earliest historically-successful donor version, not
  current Stockfish complexity).
- Forward futility pruning (beyond RFP).
- Further qsearch refinement (SEE threshold pruning, delta pruning, TT-move
  ordering in qsearch) — after the SEE partition lands.

## LATER (Tier 3 — rank by evidence x compatibility / risk / cost)

ProbCut, singular extensions, TT replacement/buckets, countermove ordering,
history-aware LMR, improving heuristic, eval-based LMR adjustment,
node-type-aware reductions, deeper NMP verification, time-management
improvements, PV stability, node-effort allocation, move-number time
scaling, cheap tempo/eval tuning if NNUE isn't ready yet. Not started.

## BLOCKED

- NMP work beyond fastsearch23 — SHELVED (not just blocked), see the NMP
  axis entry under HOLD below and EXP-R13. Revisit only after search depth
  materially deepens (e.g. from LMR/history landing).
- LMR-5 / LMP-6 (history-dependent terms) — blocked on confirming
  fastsearch24/25/26's bounded history actually diverges from unbounded
  accumulation over a longer search than the depth-7 suite (EXP-R14 flags
  this isn't yet confirmed).
- LMP-5 (improving-divisor) — blocked on Deep Blue having no `improving`
  flag or per-ply static-eval stack; needs its own prerequisite candidate.

## ACCEPTED / PROMISING (not champion — champion stays fastsearch4)

- **fastsearch18** (qsearch TT) — strong structural evidence (up to -37%
  nodes, verified-live mechanism, 32/32 adversarial correctness suite
  independently re-verified, no HIGH/CRITICAL findings from an independent
  red-team audit). CORRECTED by that audit: "bit-identical at fixed depth"
  does NOT generalise past the 5-position suite (8/399 diverge on a
  broader sweep, including one real best-move change) — paired games are
  the actual promotion gate, not fixed-depth identity. Docstring's false
  eviction-scope claim fixed in place. KEEP as working baseline for
  further qsearch work regardless — no correctness defect found, just an
  overclaimed guarantee, now corrected. See EXP-R04.
- **fastsearch15** (P1b, PV TT-cutoff split) — pooled 55.8% over 52 games,
  never negative across 3 independent replays, node-neutral structurally.
  HOLD pending the PV-classification re-audit above. See EXP-R02.
- **fastsearch21** (SEE bad-noisy qsearch skip, Design B, parent=18) —
  identical moves/scores to fastsearch18 at fixed depth 7 despite skipping
  up to ~17k moves per search; real node reductions on top of 18's own
  (-31% at tactical). One smoke gate 40.6% (mildly negative, within known
  noise band). HOLD, needs a second sample. See EXP-R07.
- **fastsearch22** (SEE good/bad qsearch ordering, Design A, parent=18) —
  same-position-set identical to 18 at fixed depth 7, smaller node effect
  than Design B (ordering only, no skip). One smoke gate 43.8%. HOLD,
  lower priority than 21 given smaller measured effect. See EXP-R08.

Note: fastsearch18/21/22 all share the "bit-identical at fixed depth
doesn't generalize past the 5-position suite" caveat from the red-team
audit (EXP-R04 Finding A2) — paired games are the real gate for all three,
not the fixed-depth argument.

- **fastsearch24** (history decomposition #1, butterfly indexing, parent=18)
  — corrects AC-003 (table was never actually from-to butterfly). Move/
  score identical to 18 at fixed depth 7, small mixed node deltas
  (expected — reindexing changes quiet tie-breaks). One smoke gate 56.2%.
  See EXP-R09.
- **fastsearch25** (history decomposition #2, symmetric malus, parent=18,
  original piece-to table) — new per-ply tried_quiets scratch buffer,
  malus = -bonus on quiets tried-but-not-cutting. Identical at fixed
  depth 7, small mixed node deltas. One smoke gate 56.2%. See EXP-R11.

## HOLD

- **fastsearch17** (P4, depth-0-in-check floor) — pooled 49.0% over 52
  games with a sign flip between identical replays; near-neutral, not
  clearly positive. One more slower-TC test or a structural audit of the
  "closed" position's +10% node outlier before further investment.
  See EXP-R03.
- **fastsearch19** (P7, RFP quiet-TT-guard removed) — node/move/score
  identical to fastsearch4 at fixed depth 7 despite more RFP attempts
  (extra attempts never won the margin test on this suite); one 16-game
  smoke gate 46.9%. Structurally inert-on-this-suite is verified, not
  inferred, but the suite is small — instrument on a larger sample before
  concluding the guard is universally unneeded. See EXP-R05.
- **fastsearch26** (history decomposition #3, bounded gravity + Pawnstar
  bonus curve, parent=18, original piece-to table) — bit-identical to 18
  on all 5 positions AT EVEN THE NODE-COUNT LEVEL (mechanistically
  explained: bonus cap never binds at depth<=7, gravity damping too small
  to flip ties on this suite). Third candidate this session to be
  live-but-inert on the standard suite (after 14, 19) — flagged as a
  possible suite-sensitivity limitation for history/ordering work
  specifically. One smoke gate 46.9%. See EXP-R14.
- **fastsearch20** (NMP v2, delta #3 — null_depth floor fix, skip-based) —
  SUPERSEDED, see EXP-R12/EXP-R13: donor re-verification found skip has no
  donor precedent (Coda clamps instead) and fires only 6 times/90 searches
  at the 80ms gate, making the original 28.1% reading uninformative, not
  negative. See fastsearch23 below for the corrected follow-up and the
  session's final verdict on the NMP axis.
- **NMP axis — SHELVED for this session** (fastsearch5, fastsearch20,
  fastsearch23, three candidate-rounds, see EXP-R13). fastsearch23
  (Coda-style clamp) recovers most savings on 3/5 positions but is WORSE
  than fastsearch20 on kiwipete (80.8% failed-null-probe rate, statistically
  identical to fastsearch5's original 80.7% — kiwipete's pathology was
  never primarily the qsearch-collapse bug). Donor comparison shows every
  functioning NMP is calibrated for 20+ ply search; Deep Blue tops out
  around depth 7-9. Do not implement NMP_DELTA.md deltas #1/#2 until this
  depth-mismatch premise changes (e.g. after LMR/history land and deepen
  the search).

## REJECTED

- **fastsearch14** (P1a, TT move preservation) — guarded branch is
  provably unreachable in normal play (instrumented: 0 fires / 1.5M+
  nodes). Root cause: `best_move` is set on the first move examined at any
  node, so it is never literally `0` at a fail-low TT write. See AC-002 /
  EXP-R01. A corrected version (gate on `best <= original_alpha` instead)
  would be a new, untested candidate, not a revival.
- (carried over from before this session) NMP v1/fastsearch5 40.6%, LMR v1
  43.8%, raw numeric SEE ordering 37.5%, combined history Patch07/
  fastsearch12 46.9%, bounded History Patch A (structural regression).
