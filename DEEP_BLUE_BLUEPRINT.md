# Deep Blue — compressed modern-engine blueprint

This document is a research map, not a source-code port.  Deep Blue's submitted
implementation remains our own Python/Numba code.  Public engines are used to
learn algorithms, ordering, failure modes and experiment design; no third-party
engine code is intended to ship in the Chessathon submission.

## Reference panel (2026-09-01)

### Stockfish 18 / current Stockfish
Use for: mature consensus search architecture, rule semantics, TT/mate handling,
modern pruning interactions, NNUE concepts, and Fishtest methodology.

### Coda
Use for: the clearest *engineering blueprint* for this project.  It is unusually
well-commented, was developed entirely through human direction + Claude Code,
and records failed experiments and live-game bugs.  Particularly valuable:
search-step order, aspiration/PVS, RFP->NMP->IIR ordering, staged MovePicker,
TT safety, history/correction-history architecture, qsearch design, time
management post-mortems and NNUE lazy-accumulator design.

### Reckless
Use for: cross-checking current elite Rust search choices.  Do not derive code
from it: current Reckless is AGPL.  Treat only public algorithmic descriptions
and independently-known techniques as research signals.

### PlentyChess
Use for: another current top-engine cross-check, especially threat-input NNUE,
fractional-depth search and modern search/eval direction.

### Pawnstar
Use for: unusually useful isolated SPRT evidence on a simpler engine.  Public
2026 results give us prioritisation signals:
- RFP (non-PV, <=7, static_eval - 80*depth >= beta): +114.8 +/-22.5 Elo there.
- IIR: +31.8 +/-16.2 Elo there.
- SEE-sorted/pruned qsearch redesign: +38.65 +/-13.78 Elo there.
- LMP: +18.6 +/-9.0 Elo there.
These numbers are *not transferable Elo*.  They only rank ideas worth testing.

## Deep Blue search ladder

Each search file is an isolated experiment.  A later number does not imply
promotion to champion.

- fastsearch1: C6 baseline — TT + killers/history + repetition.
- fastsearch2: + PVS.  Directionally KEEP; same fixed-depth results, fewer nodes.
- fastsearch3: + aspiration.  HOLD/REJECT for now; inconsistent local benefit and
  a timer-sensitive endgame score difference.  Do not stack from it.
- fastsearch4: fastsearch2 + conservative RFP (depth<=4, 100cp/depth).
  Strong local result: at fixed depth 6, 5/5 same best move and score as search2,
  median nodes about 54% of search2 on the five-position engineering suite.
  Needs paired games on pinned environment before promotion.
- fastsearch5: search4 + conservative NMP (R=3, pawn-only/zugzwang guard,
  non-PV, not in check, no consecutive null).
  Directionally promising; fixed-depth 6 returns the same move/score on 5/5
  suite positions and median node count is lower. Needs paired games.
- fastsearch6: search5 + simple IIR after NMP.  Locally nearly neutral because
  TT moves already exist at most revisited nodes. HOLD until proper paired test.
- fastsearch7: search5 + conservative LMR (not search6).
  Directionally strong: fixed-depth 6 gives identical move/score on 5/5 suite
  positions while reducing nodes on 4/5; short timed runs gain ~1 ply on several
  positions. Needs paired games.

## Near-term target architecture

iterative deepening
  -> PVS
  -> rule/repetition checks
  -> TT probe
  -> static eval
  -> RFP
  -> NMP
  -> optional IIR (only if our tests earn it)
  -> staged move ordering
       TT
       SEE-good captures/promotions
       killers / countermove
       history quiets
       SEE-bad captures
  -> LMP / futility
  -> LMR with full-depth re-search on surprise
  -> TT/history update

qsearch
  -> rule safety
  -> stand pat
  -> tactical-only generation
  -> SEE ordering/pruning
  -> delta pruning / capture cap only after measurement

## Features deliberately deferred

- Direct-legal/pin-mask generator: can recover some legality cost, but search
  selectivity is currently worth whole plies and dominates it.
- Aspiration windows: re-test later when score stability and TT are stronger.
- ProbCut / singular extensions / correction history: second wave.
- NNUE: only after classical search is stable enough that evaluator quality is
  the binding constraint.  If attempted, study threat inputs and lazy
  accumulator materialisation rather than blindly cloning HalfKP.
- Pondering: late reliability-gated experiment on one-core contention.

## Promotion gate for every feature

1. Compile / fresh-process reliability.
2. Permanent regression corpus.
3. Perft/differential/hash invariants where relevant.
4. Fixed-depth bench: compare best move, score, nodes.
5. Timed engineering suite: depth/nodes/time.
6. Paired balanced-opening games, colours reversed.
7. Candidate only becomes champion after playing evidence and zero reliability
   regressions.

No feature gets promoted because Coda, Stockfish or Pawnstar uses it.
