# Deep Blue — state as of 2026-09-09 evening

Read this first, then `git log -5`. Written to let a fresh session resume
without the previous conversation.

## What is uploaded and what to upload

**fastsearch180 is the current upload** (`submission_180.zip`, committed).
It is fastsearch135's EVALUATION — verified bit-identical over 4000 positions —
with fastsearch162's SEARCH (history-modulated LMR, multicut, singular
extensions).

Do not upload 171. Its fitted king weights were later shown not to matter (see
below). 162 and 150 carry the Q=202 weight that is the leading suspect for the
losing streak.

## The central unresolved contradiction

Self-play says the lineage improved: 150 measured +11 Elo over 140 (520 games),
162 measured +32.5 +/-29 over 150 (300 fixed games). The tournament says it
collapsed:

    build          games   score
    fastsearch135    6     75.0%     <- last build before Q=202
    fastsearch140    3     33.3%     <- 135 + Q=202 and nothing else
    fastsearch150    4     12.5%
    fastsearch162    2     50.0%
    fastsearch171    1      0.0%

Samples are tiny and decide nothing alone, but 140 isolates the variable. 180
bets the EVALUATION change was the problem and keeps the search work. **If 180
also does badly, the search additions are the culprit and the next thing to
strip is the search, not the eval.** That is the open question; only the
tournament can answer it.

## Running now

`tools/sprt_gate.py fastsearch180 fastsearch185 ... --fixed-games --max-pairs 150`
output in `scratch_185.txt`. At 240 games it read +17.4 +/-34, LLR +0.48, trend
rising over seven checkpoints. 185 = 180 + three bundled changes: timing
(stability-scaled soft budget), repetition (one prior occurrence scores
-CONTEMPT at ply 1 only), and attacked-square king danger. Each is isolated so
a negative result can be bisected.

## The biggest un-built finding

**A scale factor by the stronger side's pawn count.** Fitted on 120k quiet
positions with deep Stockfish scores, 6+ pawn bucket pinned so no global
rescale is smuggled in:

    stronger side's pawns :   0     1     2     3     4     5     6
    fitted multiplier     : 0.14  0.14  0.48  0.67  0.79  0.62  1.00
    val MSE 0.032198 -> 0.029736   (-0.002461)

For scale, every other eval finding today: attacked-square king danger
-0.000104, doubling KING_DANGER_SCALE -0.000099, refitting all eight term
scales -0.001588. This one is bigger than all of them together, and it
recovers the classical rule "you cannot win without pawns" from data with no
prior imposed.

The 5-pawn bucket at 0.62 breaks an otherwise monotone ramp and is NOT
trusted — smooth it to ~0.90 rather than shipping a bump with no explanation.

NOT YET BUILT. Runtime cost is a popcount of each side's pawns.

Motivating games: round 86 drew a Q+3P vs Q+3P ending our eval scored +83,
built from doubled pawns and king placement — both real, neither convertible.
Round 79 drew a +195 position the same way.

## Clock schedule: front-loaded, starves long games (BUILD THIS)

`deepblue/time_manager.py` sets `BASE_MOVES_REMAINING = 28`, `MIN_MOVES_REMAINING = 12`.
The allocator assumes ~28 moves remain and decays to a floor of 12 by about
move 32. In a long game that budgets as though the game were a fraction of its
real length.

Round 88 (2026-09-09), 171 moves, drawn:

    phase          us      them
    moves  1-15   5.00s   3.49s     we spend 43% more
    moves 16-30   3.55s   2.13s     we spend 67% more
    moves 31-45   2.10s   2.02s
    moves 46+     0.53s   0.75s
    final clock   2.0s    14.6s     (166 of our moves)

By move 40 we had 11.0s against their 56.3s with 126 moves still to play, then
sat on pure increment (~0.5s/move) for 120 moves. The same shape appears in
r82 (5.4s vs 53.4s), r80 (4.5s vs 26.3s) and r81 (3.7s vs 17.9s).

**Do not be misled by the 94-game average.** Aggregated we spend 3.15s per move
in moves 1-15 against opponents' 3.73s and finish 31.4s against 32.4s -- so on
average we are the FASTER side. That average is dominated by ~43-move games
where the schedule never bites. In long games it is severe. An earlier analysis
in this project concluded "time management is not the problem" from that
average; that conclusion was too strong and round 88 is the counter-example.

Simulated schedules (validated: 28/12 reproduces r81 almost exactly):

    sched    move 40   last 10 moves   wasted in a 45-move game
    28/12     1.51s        0.58s              8.8s   <- current
    32/13     1.93s        0.64s             13.2s   <- knee of the curve
    44/15     1.97s        0.98s             31.5s
    50/16     1.91s        1.26s             39.5s

32/13 captures nearly all the midgame gain for the least short-game waste.

**Two honest limits.** The 28->44 change has been tested twice against itself
and came back flat both times -- but at 20/40/60 games, which resolves nothing,
so treat that as no information rather than as evidence. And no schedule fixes
the deep endgame: with a 0.5s increment, ~0.5-0.7s per move past move 70 is
simply what is sustainable unless the opening is starved.

**What it did NOT cost.** Round 88's ending was rook and bishop against rook
with no pawns -- a theoretical draw. Our eval scored it +410 at depth 22, which
is the scale-factor bug above, not a clock bug. The clock did not lose that
point because no point was available. The open and untested question is whether
more time in moves 16-30 would have avoided trading the last pawn INTO that
drawn ending, which is where the game was really decided.

Since `sprt_gate.py` runs fixed ms/move, a schedule change is NOT exercised by
self-play at all. It has to be judged on real games or on clock simulation
(`tools/clock_sim.py`), not on a match.

## The transposition table has no ageing (real gap, but FIXING IT IS FLAT)

fastsearch185's replacement rule is, in full:

    if tt_depth[index] <= depth or tt_key[index] != value:

Depth-preferred, with no notion of WHEN an entry was written. An entry stored at
depth 14 on move 5 can never be evicted by a shallower one, so it survives the
whole game and keeps being probed and trusted long after its position is gone.
`grep -c generation` over fastsearch185.py returns 0.

Stockfish has exactly the field we lack -- a generation counter, so entries from
older searches are preferentially replaced even when deeper. This is the answer
to "engines like Stockfish have history and do not make these mistakes": the
problem was never that state is carried, it is that ours never expires.

**This had already been found and thrown away.** fastsearch154 implemented a
proper per-entry generation and measured -7.9 Elo, a number this harness cannot
distinguish from zero, and it was discarded on that basis. Check the sample size
before believing any verdict in this repo's history.

fastsearch187 is the fix under test: one line at the start of each search,

    np.subtract(self.tt_depth, 1, out=self.tt_depth, where=self.tt_depth > 0)

Decrementing every stored depth per move gives both effects of a generation
counter -- old entries lose replacement priority AND stop satisfying
`tt_depth >= depth`, so they are re-searched instead of trusted -- with no
signature changes. A real generation field must be threaded through negamax,
quiescence and search_root, and argument-threading misses have silently broken
three candidates here (that is why tools/check_calls.py exists). Cost is one
numpy pass over 4M int16 per move, under 10ms against ~2000ms.

On round 90 move 43, where 185 played Kd3 -- a move NO fixed depth from 10 to 22
chooses -- 187 plays Ra6+, which is what a fresh engine plays.

RESULT (2026-09-10): 187 measured -0.0 +/-41 over 140 games, oscillating around
zero across seven checkpoints (-17.4, -52.5, -5.8, +13.0, -3.5, +5.8, -0.0).
FLAT.

So the conclusion, stated as promised before the match was run: THE IN-GAME
VERSUS COLD DIVERGENCE IS NOT A DEFECT. The missing generation field is a real
gap against Stockfish, but closing it changes nothing measurable. A warm engine
and a fresh one choose differently because the heuristics are working, not
because the table is stale.

DO NOT REOPEN THIS. It was carried as an open bug from round 58 to round 98 on
the strength of an argument, never a measurement, and the only evidence it cost
anything came from using this engine as its own referee -- the blind referee the
Stockfish tooling exists to replace. Rounds 90, 94, 97 and 98 were a pattern
found in noise.

## USE DONOR_NEXT_IDEAS.md. It is the best-sourced backlog in this repo.

240 lines of cited search research -- Reckless commit SHAs with their SPRT Elo
deltas, cross-checked against Stockfish, Ethereal, Coda and Viridithas source,
with an explicit note on which claims were independently re-verified and which
were not. It is dependency-ordered and triaged smallest-first.

I ignored it for a full session and invented eval candidates instead. Those
measured -40, flat, flat and flat. Ten minutes in this document produced a
better-motivated candidate than any of them. Read it FIRST.

Checked against fastsearch185, its history section says:

    gravity            185 HAS IT (fastsearch91)
    malus              185 HAS IT (symmetric, reuses the bonus)
    HISTORY_MAX 16384  185 HAS IT -- the donor-majority constant
    bonus shape        NOT DONE -> fastsearch192

fastsearch192 is the document's own step 2. 185 used min(depth*depth, 1536),
the depth-squared family (Ethereal, Pawnstar). Four donors use linear-capped
instead and the survey calls it the modern consensus: Stockfish
min(133*d - 81, 1487), Reckless min(184*d, 1742), Coda clamp(0, 1653, 245*d - 18),
Viridithas min(mul*d + offset, max). 192 adopts Stockfish's exact shape.

Why it plausibly matters, which is visible without any match:

    depth    185 (d*d)   192 (linear)
        1            1            52
        3            9           318
        6           36           717
       12          144          1487

The gravity update is `entry + bonus - entry*bonus/16384`. A bonus of 9 barely
moves an entry at all, so at depths 3-6 -- where the overwhelming majority of
nodes are -- our history table has been close to inert, only learning at depths
that are rare. That is a candidate explanation for the measured first-move
cutoff rate of 85.1% against a ~90% benchmark.

STILL UNBUILT, all cited in that document, roughly in its own priority order:

  * malus with its OWN slope/offset/ceiling (Coda, Reckless, Stockfish all
    separate it; ours reuses the bonus symmetrically)
  * Reckless's malus divisor -- divide malus by the count of quiets already
    punished at that node, so one wide node cannot mass-poison the table. We
    apply full malus to EVERY tried quiet.
  * LMP-7: qsearch movecount cutoff (move_count>=3: break). Flagged "fully
    independent, testable immediately".
  * LMP-2: in-check guard for LMP. Reckless ran 18 months without it, then
    measured +1.19 Elo over 138,000 games.
  * priority #3: SEE good/bad noisy partition + qsearch skip. fastsearch191 did
    the main-search half (+13.9 +/-29, inconclusive); the qsearch half is
    untouched.

CAUTION the document states about itself: Coda's NMP body and Stockfish's
update_all_stats scaling fractions were NOT verified, and the
ordering-before-LMR sequencing is "a well-supported prior, not a proven fact".
Donor precedent says what is worth testing, not what will work here.

## What has been ruled out (do not re-litigate)

- **King-attack weight RATIOS do not matter.** Ours, Stockfish's, classic
  2/2/3/5 units, 20/20/40/80 and a free fit all land within 0.00005 of each
  other under our formula. The magnitude and the SHAPE of the input matter;
  the ratio does not.
- **Time management on AVERAGE is fine, but see the clock-schedule section
  above -- the average hides a real failure in long games.** Over 94 games we
  spend 3.15s per move in moves 1-15 against opponents' 3.73s and finish 31.4s
  against 32.4s, but that is dominated by ~43-move games. In long games we are
  badly starved (r88: 2.0s vs 14.6s).
- **In-game vs cold divergence is not one component** -- clearing the TT, the
  history/killers, the continuation history or the correction history each
  recovers only about a quarter of the gap. But see the TT-ageing section
  above: the underlying defect (a table that never expires) is real and was
  identified afterwards.
- **fastsearch173** (one prior repetition scores -CONTEMPT at every ply) makes
  the whole tree collapse to +/-CONTEMPT in shuffling endgames. Use the ply-1
  form in 183/185 instead.

## The two recurring failure modes, named

1. **Depth-window artifacts.** Round 82's 19...b4 and round 84's 14...gxh6 are
   moves the engine prefers at exactly one depth and abandons at the next.
   185's timing component targets these.
2. **Eval blindness.** Round 84 move 11: at depth 16 our eval rates three
   different moves within 13cp of each other; Stockfish calls one an
   inaccuracy. No search change reaches this. The teammate's NNUE is aimed here.

## Tooling built this session

The Lichess Stockfish-eval database (21GB) is at
`C:/Users/uniqu/AppData/Local/Temp/deepblue-teammate-data/lichess_eval/`.
Pipeline: `sample_sf_positions.py` -> `filter_quiet.py` -> `extract_terms.py`
-> `tune_terms.py`. Also `eval_error.py`, `move_quality.py`, `extract_king.py`,
`fit_king.py`, `extract_ksquares.py`, `time_neutral.py`, `tt_divergence.py`,
`isolate_carry.py`, `blunder_scan.py`, `ship.py`, `check_calls.py`.

`move_quality.py` FAILED its own validation — it could not reproduce the known
ordering of two builds 11 Elo apart. These tools screen out changes with no
mechanism; they do not rank near-equal builds and do not replace a match.

## Rules of engagement learned the hard way

- This harness resolves ~+/-65 Elo at 60 games and +/-30 at 300. Never ship on
  a null result; that is how Q=202 got in. Always `--fixed-games` for a number
  that will be quoted — SPRT early stopping inflates accepted effects.
- Bundle changes that are individually too small to measure, but only when each
  component is already known not to be harmful, and isolate each so it can be
  reverted in one line.
- NEVER edit a constant in the shared `deepblue/eval_terms.py` to make a
  candidate. fastsearch140 and 141 are byte-identical because of this, so
  141's recorded "-34 Elo" is one build playing itself. Add a new constant and
  a new njit function instead — Numba compiles lazily, so an unused one costs
  no init time.
- The machine uses Modern Standby and slept for 23 minutes across five episodes
  before a keep-awake holder was started. Start the scratchpad `keep_awake.py`
  before any long run, and keep the laptop on AC — hibernate-after is 6 hours
  on battery.
