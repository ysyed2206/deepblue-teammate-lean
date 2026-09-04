# Deep Blue - architecture

Current state: **S0 complete, hardened and shipping. S1 core complete. S1-search0 built and measured. Next: S1-search1 (Zobrist + transposition table + ordering).**

This document is kept accurate as the engine changes. It doubles as the
preparation for the finalist technical review, where the agent has to be
explained by a human.

---

## What ships

The submission is `agent.py` at the zip root plus the `deepblue/` package.
`harness/package.py` globs top-level `*.py` and named includes only, so the
package is included explicitly (`make zip` passes `--include deepblue`) and
`tools/fresh_process_test.py` builds the zip and runs from the extracted copy so
a broken submission cannot pass unnoticed.

```
agent.py                 entry point, fallback protection, per-game state
deepblue/constants.py    material values, generated piece-square tables, masks
deepblue/evaluation.py   tapered evaluation over a python-chess board
deepblue/time_manager.py clock allocation
deepblue/reference.py    S0: negamax search on python-chess
```

Not shipped, development only:

```
champion/                frozen copy of the current known-good build
tools/benchmark.py       movegen / search NPS on a fixed position suite
tools/arena_parallel.py  many concurrent games, with a confidence interval
tools/clock_sim.py       full-game clock discipline, worst-move reporting
tools/fresh_process_test.py  cold import + edge positions, from the packaged zip
```

## Two engines, on purpose

**S0 (`deepblue/reference.py`)** is built on `python-chess`. It is slow, and that
is accepted. It exists as the insurance policy (there is always a validated
submission), as the correctness oracle (every faster implementation is
differential-tested against it) and as the first real opponent.

**S1 (`deepblue/fastcore.py`, in progress)** is our own board representation,
designed so move generation, make/unmake, evaluation and search stay entirely
inside Numba nopython mode with no Python object crossing the boundary. S1
replaces S0 in the submission only when it is measurably both correct and
stronger.

## Board representation - S0

`chess.Board` from python-chess. Its move generation is the measured bottleneck:
9,600 positions/s at Kiwipete, which caps the search at 13,353 nodes/s there.

## Evaluation

Tapered. Two scores are accumulated from White's point of view - a middlegame
score and an endgame score - and interpolated on a phase that runs 24 (full
board) to 0 (bare kings), computed from non-pawn material so pawn trades do not
shift it. The result is flipped once at the end for the side to move.

Terms: material, piece-square tables, bishop pair, doubled / isolated / passed
pawns, rooks on open and half-open files, tempo, and mobility (off by default,
pending EXP-002).

Piece-square tables are **generated from interpretable rules**, not transcribed.
`deepblue/constants.py` holds a named parameter per effect - centre attraction,
rank advancement, back-rank penalty, king shelter versus king centralisation -
and builds the 64-entry tables from them. This keeps the tables explainable, and
makes the parameter vector the thing a tuner will later optimise.

The blend truncates toward zero rather than flooring, because floor division is
asymmetric about zero and made `evaluate(position) != evaluate(mirror)` on ~40%
of positions.

## Search - S0

Fail-soft negamax with alpha-beta, inside iterative deepening.

- **Transposition table** keyed on `board._transposition_key()`, storing depth,
  score, bound type and best move, with depth-preferred replacement and a hard
  entry cap so it cannot grow into the 2 GB limit.
- **Quiescence** at the leaves: captures and queen promotions, with full move
  generation when in check so the search cannot stand pat on a position where
  the king is attacked.
- **Move ordering**: transposition move, then promotions, then captures by
  MVV-LVA, then two killers per ply, then the history heuristic. The bands are
  widely separated so categories cannot interleave.
- **Repetition** is scored as a draw using the real game history, which persists
  across moves in the process. Without this the referee's automatic threefold
  claim can take a won game away.
- **Mate scores** are distance-adjusted, so the engine prefers the fastest mate
  and the slowest loss.

## Time management

Two budgets per move. **Soft** is the point at which no new iterative-deepening
pass is started; **hard** is the point at which the pass in progress is abandoned
and the last completed pass's move is returned. Abandoning is free by
construction, so the hard limit is 1.4x soft and acts as a real spending cap.

The engine measures its own iteration-to-iteration growth as it searches and
refuses to start a pass the estimate says cannot finish. Without that prediction
it overshot a 3.2 s budget by 5-9 s.

A fixed protocol overhead is subtracted from the clock before anything is
allocated, because the referee measures wall time around the whole call. Below
1.5 s remaining the engine stops trying to play chess and just moves.

## Reliability

`get_move` chooses a known-legal fallback *before* searching and returns it on
any exception, on an empty search result, and on the (unreachable) case of the
search returning a move that is not legal. An illegal move, a crash or a flag
loses the game outright, and qualification is a 13-round Swiss where one lost
point outweighs a large amount of nominal strength.

## Repetition and terminal handling

Two structures, deliberately separate. `game_counts` holds positions that have
ACTUALLY OCCURRED in the game and is permanent; `path_counts` holds positions
on the line currently being searched and is pushed and popped as the search
walks. Conflating them made the engine simultaneously miss real repetitions and
invent hypothetical ones. A FIDE threefold is three occurrences; the
twofold-in-search shortcut is retained but named, documented as a heuristic and
switchable.

`agent.py` records both the position it is handed and the position its own move
creates, because the referee claims threefold on either side's turn and we are
only ever shown one of the two.

Terminal precedence matches the referee's `outcome(claim_draw=True)`: checkmate
first, then insufficient material, then stalemate, then the claimable draws.
Getting this order wrong turns a mate into a draw or a stalemate into a win.

## The fused search path (S1)

The search does NOT build a legal move list. It generates pseudo-legal
candidates, makes each once, tests the mover's king, and either recurses on
that same made position or unmakes and skips. `fastcore.generate_legal` remains
exact and is still used by perft and the differential tests, but prefiltering
makes every searched move twice and is not what a search should do.

Interruption is a one-byte flag in an array, set by a Python timer thread and
read on **every** node entry. It was originally read only every 2,048 nodes,
which let the search run 2,090 ms past its deadline; the flag is cheap and the
mask was the bug.

## Known limitations

- Search NPS is 13k-25k. This is the dominant weakness and the reason S1 exists.
- Quiescence has no delta pruning and no SEE, so it explodes in capture-rich
  positions: depth 1 at Kiwipete costs 9,058 nodes (EXP-005).
- No null-move pruning, no late move reductions, no aspiration windows. These
  are deliberately absent so S1's search has a clean baseline to be measured
  against, one feature at a time.
- Mobility is implemented but disabled pending measurement (EXP-002).
- No pondering, no opening book, no tablebases, no learned evaluation. All are
  later decisions and none are started.

## Current measured bottleneck

For S1-search0: `make`/`unmake` (67.6% of modelled node cost) and the
`in_check` legality test after each make (27.7%). But the binding constraint on
*strength* is not node cost at all - it is that S1-search0 has no transposition
table, no killers and no history, and therefore reaches fewer plies than S0 at
the start position despite 42x the node rate.

## Next intended experiment

C6 / S1-search1: Zobrist hashing, an array-backed transposition table,
TT-move-first ordering, killers and history. Then re-profile.

EXP-006 (pin-mask legal generation) is **deferred on measured grounds**, not
skipped: in the fused search it would recover roughly 25-30% of node cost,
about +0.3 ply, which does not justify its correctness risk ahead of features
worth whole plies. See EXPERIMENTS.md EXP-014.
