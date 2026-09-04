# Donor archaeology — raw findings

Raw output from donor-mining subagents, triaged into `BACKLOG.md`. Every
claim below cites its source; no claim is acted on in
`CLAUDE_REMOTE_RESULTS.md` without a Deep Blue source cross-check first.

**Coordinator spot-check**: 2 of Agent A's citations (Pawnstar
`search_state.h`'s `ScoreAndSortMoves` SEE scoring + qsearch early-exit
comment, and its README's "+38.65 ± 13.78 Elo" SPRT line) were independently
re-fetched and matched verbatim. The rest were not independently
re-verified this session — treat as high-confidence, not proven, and
re-check the specific line before coding against it if it's load-bearing.
Two things Agent A explicitly could NOT verify: Coda's actual NMP code body
(`src/search.rs` is 435KB, only the tunable-parameter table at the top was
fetchable) and Stockfish's per-table scaling fractions in `update_all_stats`.

---

## History decomposition (priority #2)

**Bonus formula**: two live families. depth² capped (Ethereal:
`16*d*d + 128*max(d-1,0)`, collapses to 32 above d=13; Pawnstar:
`min(d*d, 400)` out of `kHistoryMax=16384`) vs linear-capped, the modern
consensus (Stockfish `min(133*d - 81, 1487) + 364*(bestMove==ttMove) + ...`;
Reckless quiet `min(184*d,1742) - 72 - 42*cut_node`; Coda
`clamp(0, 1653, 245*d - 18)`; Viridithas `min(mul*d + offset, max)`).

**Malus**: present in every donor. Pawnstar/Ethereal use a symmetric ±bonus
(1.0x). Coda/Reckless/Stockfish give malus its OWN independently-tunable
slope/offset/ceiling (Coda's malus ceiling 1037 is actually LOWER than its
bonus ceiling 1653 — bonus overtakes malus above d≈4.3). **No donor uses a
flat 5x asymmetric malus** — the rejected internal Patch A's 5x ratio has
no donor precedent found. Reckless also divides its malus by the count of
quiets already punished this node, preventing one wide node from
mass-poisoning the table — Patch A had no such term.

**Decay = gravity, and that's the only decay any donor uses**: identical
form everywhere — `entry += bonus - entry * |bonus| / MAX`, clamped to
±MAX. **No donor runs a separate decay/aging pass on top of gravity.**
Patch A's "extra decay" (on top of whatever its bonus/malus already did)
has no donor precedent and is flagged as a likely direct cause of its
failure, alongside the 5x malus.

**Table shape**: Pawnstar is the closest match to Deep Blue's existing
butterfly (side × from × to, `kHistoryMax=16384`). Reckless is lower
(8192); Viridithas is piece-to, not from-to. **16384 was not Patch A's
error** — it's the majority constant.

**Killers**: three real answers — Pawnstar keeps killers and hard-clamps
quiet-history below the killer band (`kMaxQuiet = kKillerBase - 2`, so
history literally cannot outrank a killer by construction); Ethereal keeps
killers as picker stages; Reckless and Coda have removed killers entirely
(history-only quiet ordering, Coda's commit explicitly SPRT-validated).
Deep Blue has 2 killers today — Pawnstar's clamp-band model changes
nothing about that; killer removal is a separate, later, bigger swing.

**Individually-testable pieces, smallest first** (do NOT stack — one per
candidate, per this project's core testing philosophy):
1. Gravity only (no formula/malus change).
2. Bonus shape only (with gravity already in).
3. Symmetric malus only (exactly `-bonus`).
4. Malus decoupling (own slope/offset/ceiling) — only after 1-3.
5. Malus width-damping (divide by quiets-already-punished count).
6. Killer/history band clamp.
7. Persistence-across-searches vs reset — separate axis, not urgent.
8. Explicitly do NOT test: extra decay on top of gravity (no donor
   precedent; likely Patch-A failure cause together with the 5x malus).

## SEE good/bad noisy partition + qsearch skip (priority #3)

Local contract confirmed by reading `deepblue/see.py`: `see_value(bb, occ,
mail, st, move) -> int` and `see_ge_zero(...) -> bool`, pre-move state (no
make/unmake needed), returns 0 for non-captures, and the module's own
docstring says **do not use it for promotions in this version** — any skip
logic must exclude promotions, not just filter by `captured != NO_PIECE`.

**Ordering**: Pawnstar's is the minimal version — plain sign split,
`see>=0 ? base+see : see` (losing captures fall below all quiets). Ethereal
defers bad captures to a final picker stage. Coda/Reckless use
history-aware thresholds requiring capture-history state Deep Blue doesn't
have yet (defer).

**Qsearch skip — Pawnstar is the actual +38.65 Elo source, and it's a
sorted early-exit, not a per-move filter**: generate captures, sort by
SEE, `if (move.score() < 0) return best_score` (comment: "All moves after
this are -ve SEE and may be skipped"). **Caveat, load-bearing**: Pawnstar's
own SPRT bundles this WITH removing qsearch's TT entirely — it is not
isolated evidence for the SEE skip alone, and Pawnstar has no qsearch TT to
begin with (Deep Blue's fastsearch18 just added one — the two ideas
interact and should not be assumed additive). Ethereal instead uses an
alpha-relative floor `MAX(1, alpha - eval - 123)` (skips SEE==0 too, and
tightens as the position falls further below alpha).

**Individually-testable pieces**:
1. Qsearch skip, sign-only (`see_value < 0`), non-promotion captures only.
2. Same rule as sorted early-exit instead of per-move filter (cheaper).
3. Threshold ≥ 1 instead of ≥ 0 (Ethereal's floor).
4. Alpha-relative threshold `see >= max(1, alpha - eval - 123)` — only
   after 1/3 are settled.
5. Ordering-only (good captures ahead of quiets, bad after) in the MAIN
   search — independent of qsearch work entirely.
6. Explicitly deferred: history-aware thresholds (need capture history,
   don't have it); qsearch TT removal (we just added it in fastsearch18 —
   do not bundle these two ideas).

Full implementation-ready design for candidates 1 and 5-equivalent (Design
A/B) already produced by Agent C — see `CLAUDE_REMOTE_RESULTS.md`'s donor
archaeology section for the exact band constants, call sites, and 9
targeted tests. Ready to implement directly on `deepblue/fastsearch18.py`.

## NMP delta (blocking prerequisite for any NMP work)

fastsearch5 already has the 4 basic guards (see AC-002-adjacent correction
in CLAUDEREAD). Every donor examined has those 4 AND several more things
fastsearch5's summary doesn't mention having:

1. **Cut-node/non-PV gate** — SF/Viridithas require `cutNode`; Ethereal
   requires `!PvNode`. A hard gate, not tuning.
2. **TT-consistency guard** — don't null when the TT already says this
   node fails low (`!ttHit || bound != UPPER || ttValue >= beta`).
3. **Mate/decisive clamp on the null return** — `if null_score is decisive:
   return beta` (not the raw null score). **Flagged as the single most
   likely silent-bug source for a 40.6% paired result** — a wrong mate
   value returned from a null-move subtree is exactly the shape of bug
   that tanks a paired score without showing up in a small regression
   corpus.
4. **R must be capped so `depth - R >= 1`** (never drop straight into
   qsearch from a null move).
5. Reduction formula: base 4-7 + depth/3-5 + min((eval-beta)/174-256, 3-4).
   Coda's un-verified param table implies a similar shape.
6. **Verification search with a ply-barrier, not a bare boolean** — shallow
   nodes (depth < 12ish) return the null score immediately; only deep nodes
   pay for a real verification re-search, and the verification must be
   banned from re-triggering NMP inside its own subtree
   (`nmp_min_ply`/`ban_nmp_for` pattern, present in Viridithas/SF/Reckless/
   Coda alike).
7. Ordering: RFP before NMP everywhere (Deep Blue already has this).

**Individually-testable pieces, safety slices FIRST** (items 1-4 are
correctness/safety, not tuning — do these before touching the reduction
formula, since a 40.6% result looks more like a wrong-return-value bug than
a mistuned R):
1. Cut-node/non-PV gate only.
2. TT-consistency guard only.
3. Mate-clamp on the null return only.
4. R cap only.
5. Fixed R → depth-scaled R, no eval term.
6. Eval-scaled R term, on top of 5.
7. Margin entry threshold (replace bare `eval >= beta`).
8. Verification search with ply-barrier.

**Action required before ANY of this**: diff fastsearch5.py's actual code
against items 1-4 above to see which (if any) are already present — this
is a Deep Blue-side task, not further donor research, and is the literal
`NMP_DELTA.md` this project's standing instructions require before NMP
work resumes. Not yet done this session.

## History → LMR → LMP donor archaeology (round 2)

Reckless's 2025 ground-up rebuild re-added every search feature one at a
time under SPRT (commit dates verified via GitHub commit API, not
inferred): MVV-LVA → TT move ordering → PVS → **quiet butterfly history →
history formula improved → threat dimensions added → noisy history** →
**LMR** (`dfdd508a`, +127.73 Elo) → **LMP** (`0148e410`, +13.74 Elo). The
full move-ordering stack landed before LMR both times this happened in
Reckless's history (2023 era and 2025 rebuild) — sequencing is verified
twice, independently. **Caveat, stated plainly by the researching agent**:
no donor commit message or SPRT note explicitly says "ordering had to come
first because LMR depends on it" — the sequencing is real and repeated,
the causal claim is a well-supported prior, not a proven fact.

**Actionable finding**: Deep Blue's history (`history[piece,to_square] +=
depth*depth`, unbounded, no malus/gravity) has no stable scale — any
history-dependent LMR term added on top would have no consistent divisor
across depths. Every donor added gravity/bounds to history BEFORE adding
history-dependent LMR. This is a hard prerequisite, reinforcing (from a
different angle than AC-003) why history decomposition must land before
LMR-5 (history-dependent reduction) specifically, even though bare LMR-1
through LMR-4 don't need it.

**LMR mechanics** (all re-verified against actual Reckless/Stockfish/Coda
source, not memory): shape is universally `base + c*ln(depth)*ln(move_count)`
(table vs. closed-form is a determinism/platform choice, ~0 Elo either
way — Reckless moved table→formula specifically because floating-point
`ln()` is non-deterministic across platforms/Rust versions, directly
relevant to Deep Blue's own numba/cross-platform CI). Re-search is
**null-window, full depth** on a reduced fail-high, THEN full-window full
depth only if that lands inside (alpha,beta) — getting this backwards
(jumping straight to full window) is flagged as a classic, expensive
error. Initial guards were strict (`depth>=3`, `move_count>=4`, quiet-only,
not-in-check) and only loosened over ~2 years as ordering/other guards
matured — starting Deep Blue at `move_count>=4` (not the current-donor
`>=2`) is recommended given Deep Blue's ordering is far less mature than
current Reckless's.

**Failure-mode ranking for why a "43.8%" LMR v1 result is plausible** (Deep
Blue's own old LMR attempt's code no longer exists in context, so this is
donor-pattern-matching, not a diagnosis): missing/wrong re-search is the
single most catastrophic pattern (propagates shallow-tree scores as if
exact); re-searching at full window instead of null window is the second;
reducing tactical moves (captures/promotions/checks) in v1 is a common
early mistake — every donor's FIRST LMR exempted these entirely.

**LMP**: `3 + depth*depth` movecount threshold (self-limiting, no depth cap
needed), quiet-only, `!is_loss(best_score)`. An in-check guard was MISSING
for 18 months in Reckless's own 2025 rebuild before being added
(`ddfe4848`, +1.19 Elo over 138k games) — a real donor-documented "obvious
guard that was actually missing for a long time" case. LMR and LMP were
explicitly DE-COUPLED in Reckless's most recent history (`064af30a`,
"fully decouple reductions from move loop pruning," Elo-neutral) after
being coupled for months — donor's own final verdict is to build them as
independent gates, not fine-tune the coupling.

**Individually-testable pieces — LMR** (dependency order, one per
candidate): LMR-0 (prerequisite, = history decomposition candidates #2/#3
below) → LMR-1 bare LMR (depth>=3, move_count>=4, quiet-only, not-checked,
not-tt-move, fixed R=1, null-window-then-full-window re-search) → LMR-2
log/ilog2 reduction formula → LMR-3 reduce PV nodes less → LMR-4 killer
relief (Deep Blue already has 2 killers, near-free) → LMR-5 history-scaled
adjustment (blocked on LMR-0) → LMR-6 doDeeper/doShallower after a reduced
fail-high → LMR-7 check relief → LMR-8 allow captures to be reduced (last)
→ LMR-9 TT-move qsearch guard (only meaningful once reductions can push
depth to 0).

**Individually-testable pieces — LMP** (independent of the LMR list, don't
couple them): LMP-1 bare quiet LMP (`move_count >= 3+depth*depth`) →
LMP-2 in-check guard → LMP-3 gives-check carve-out → LMP-4 unadjusted
depth (no-op for Deep Blue currently, no extensions — record, don't test)
→ LMP-5 improving-divisor (blocked — Deep Blue has no `improving` flag or
per-ply static-eval stack, needs its own prerequisite candidate) → LMP-6
history term (blocked on LMR-0) → LMP-7 qsearch movecount cutoff
(`move_count>=3: break`, fully independent, testable immediately).

Full commit-by-commit citation trail (20+ Reckless commits with exact SHAs
and SPRT Elo deltas) is in this session's transcript; re-derive from
https://github.com/codedeliveryservice/Reckless commit history if this
file is read in a future session without that transcript. Two things NOT
verified: the causal claim above (sequencing only), and Deep Blue's own
historical LMR-v1 code (not in this session's context, never located in
git history).
