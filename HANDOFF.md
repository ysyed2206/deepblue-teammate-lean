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

## What has been ruled out (do not re-litigate)

- **King-attack weight RATIOS do not matter.** Ours, Stockfish's, classic
  2/2/3/5 units, 20/20/40/80 and a free fit all land within 0.00005 of each
  other under our formula. The magnitude and the SHAPE of the input matter;
  the ratio does not.
- **Time management is not the problem.** Over 94 real games we spend 3.15s
  per move in moves 1-15 against opponents' 3.73s, and end games with 31.4s
  against their 32.4s. We are not the slower side. Running low correlates with
  LONG games (74 moves average vs 43), not with losing.
- **In-game vs cold divergence is not one component.** Clearing the TT, the
  history/killers, the continuation history or the correction history each
  recovers only about a quarter of the gap.
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
