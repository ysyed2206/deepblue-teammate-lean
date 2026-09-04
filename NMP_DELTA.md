# NMP delta — fastsearch5 (actual source) vs donor findings

Required before any NMP work resumes (standing instruction). Every line
below is read directly from `deepblue/fastsearch5.py` (quoted with line
numbers), cross-checked against Agent A's donor archaeology in
`DONOR_NEXT_IDEAS.md`. This corrects, rather than assumes — one of Agent
A's flagged risks turns out to already be handled correctly in fastsearch5.

## What fastsearch5 actually has (verified by reading the file)

- Non-PV gate: `not is_pv` where `is_pv = beta - original_alpha > 1`
  (`fastsearch5.py:354,366`). Present, but coarser than donor's cut-node-
  specific gate (see below).
- Not-in-check gate, `depth >= NMP_MIN_DEPTH` (=4), `abs(beta) <
  MATE_THRESHOLD`, `st[3] < 99` (`fastsearch5.py:385-392`).
- Non-pawn-material/zugzwang guard: `non_pawn != 0` computed per side from
  N/B/R/Q bitboards (`fastsearch5.py:394-398`).
- `static_eval >= beta` entry condition, no margin (`fastsearch5.py:403`).
- No-consecutive-null: the null subtree's own recursive call passes
  `can_null=False` (`fastsearch5.py:427`), but that call's own children get
  `can_null=True` again in the normal move loop (`fastsearch5.py:472,479,
  486`) — this correctly prevents an immediate null-into-null while still
  allowing a later null move deeper in the same subtree after a real move
  intervenes, which is standard, not a bug.
- EP-aware null hash: `old_ep_file` captured before flipping `st[2]`, XORed
  into `null_value` only if valid (`fastsearch5.py:409,416-417`), side key
  always flipped (`fastsearch5.py:415`). Matches the codebase's own
  canonical-EP-hash convention used everywhere else.
- Fixed reduction `NMP_REDUCTION = 3` (`fastsearch5.py:84,418`), no depth or
  eval scaling at all.
- **Mate/decisive clamp on the null return IS present**:
  `return beta if null_score > MATE_THRESHOLD else null_score`
  (`fastsearch5.py:437`). This is functionally the same pattern Agent A
  found in Ethereal (`(value > TBWIN_IN_MAX) ? beta : value`). **Correction
  to the donor-archaeology flag**: Agent A named this "the single most
  likely silent-bug source for a 40.6% result" as a *missing* feature —
  reading the actual source shows it is not missing. Do not implement it;
  it's already there.
- RFP runs before NMP in the code (`fastsearch5.py:349-437`), matching
  donor consensus ordering.

## Real, verified deltas

### 1. MISSING — TT-consistency guard

Nothing in the NMP block (`fastsearch5.py:385-437`) reads `tt_bound`,
`tt_score`, or the TT-probe outcome from earlier in the same function
(`fastsearch5.py:330-343`) before attempting a null move. Donor engines
(Ethereal: `!ttHit || !(ttBound & BOUND_UPPER) || ttValue >= beta`) refuse
to null when the table already says this node fails low — nulling anyway
wastes a search and, per Ethereal's own logic, risks acting against
already-known information. Genuinely absent here.

### 2. MISSING — verification search

fastsearch5 has no re-search safety net at all: `if null_score >= beta:
... return ...` (`fastsearch5.py:435-437`) cuts unconditionally on a single
null-move probe. Every donor examined (Stockfish, Viridithas, Reckless,
Coda) adds a verification re-search, gated by depth (shallow nodes trust
the null result immediately; deeper nodes re-verify with real search before
trusting it), specifically to catch zugzwang-shaped positions the
non-pawn-material guard doesn't catch (that guard only rules out literal
pawn-and-king endings — plenty of positions with pieces on the board are
still zugzwang-like). Genuinely absent, and the donor commit history Agent
A found frames this as a real, separate source of Elo versus the basic
guards fastsearch5 already has.

### 3. BUG-SHAPED — null_depth floor allows dropping straight into qsearch

```python
null_depth = depth - 1 - NMP_REDUCTION      # fastsearch5.py:418
if null_depth < 0:
    null_depth = 0                          # fastsearch5.py:419-420
```

With `NMP_MIN_DEPTH = 4` and `NMP_REDUCTION = 3`, the smallest depth that
can trigger NMP is `depth=4`, giving `null_depth = 4-1-3 = 0`. The
recursive call is `negamax(..., null_depth, ...)`
(`fastsearch5.py:421-428`), and `negamax` immediately routes `depth <= 0`
into `quiescence()` (`fastsearch5.py:321-323`). So the *most common*
triggering depth for NMP in this implementation sends the null-move
verification straight into a full quiescence search, not a shallow
alpha-beta probe. One donor commit Agent A found (Coda: "capped NMP
reduction to ensure search never enters quiescence") treats this specific
condition as a bug worth a dedicated fix. This is the most concrete,
verifiable candidate explanation for why fastsearch5 (NMP v1) scored 40.6%
in its paired gate: a "cheap fail-high proof" that silently becomes a full
tactical search at the most common trigger depth is not the mechanism NMP
is supposed to be, and its behavior (returning `beta` on a decisive
qsearch-derived score, or a real qsearch score otherwise) is harder to
reason about than a shallow null probe.

### 4. Weaker than donor (plausibly conservative, not obviously a bug)

- Fixed `R=3` vs donor `4-7 + depth/3-5 + eval-margin term` — fastsearch5
  reduces less than every donor found, which should make it *safer* (less
  aggressive pruning) at the cost of pruning less. Not flagged as a likely
  failure cause.
- Plain `static_eval >= beta` vs donor's margin-relaxed entry (e.g. SF:
  `static_eval >= beta - 13*depth - ... + 365`, i.e. easier to satisfy than
  a bare `>=`) — fastsearch5's gate is *stricter*, firing NMP less often.
  Also not flagged as a likely failure cause, though it does mean
  fastsearch5's NMP has fewer opportunities to help at all.
- Coarser non-PV gate vs donor's specific cut-node gate — fastsearch5
  allows NMP at some non-PV, non-cut ("all") nodes that stricter donors
  would exclude. Plausible source of some inefficiency, not obviously a
  correctness bug.

### 5. Not a concern for this codebase specifically

Coda's "Null Sentinel Fix" (guards continuation-history state from
null-subtree pollution) doesn't apply — Deep Blue has no continuation
history yet. The equivalent concern for the killer table (shared,
ply-indexed, mutated throughout the whole search regardless of whether a
null move was involved) is architecturally already the same for every
sibling node at a given ply, null-move-triggered or not — nulling doesn't
introduce a new failure mode here that doesn't already exist in the
existing ply-indexed killer design.

## Recommended order for any NMP v2 candidate (per priority, safety first)

1. TT-consistency guard (delta #1) — cheap, safety-only.
2. Verification search with a depth-based ply-barrier (delta #2) — the
   donor-confirmed missing safety net.
3. Fix the null_depth floor so it cannot reach 0 (delta #3) — e.g. require
   `depth - 1 - NMP_REDUCTION >= 1` as part of the trigger condition itself
   (skip NMP rather than clamp into qsearch), or raise `NMP_MIN_DEPTH`
   enough that the floor can't bind. **Test this one in isolation first** —
   it's the most concrete, mechanistic candidate explanation for the
   original 40.6% result, and isolating it will show directly whether it
   was the actual cause.
4. Only after 1-3: revisit the reduction formula and entry-margin deltas in
   #4, each as its own separately-tested candidate, per the "one idea per
   candidate" rule.

Do not implement #1-3 as one combined candidate — test #3 alone first,
since it has the clearest mechanistic story, before adding #1 and #2 on
top (or in place of it, if #3 alone turns out to explain most of the gap).
