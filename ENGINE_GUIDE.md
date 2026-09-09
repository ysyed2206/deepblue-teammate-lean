# Deep Blue: How This Chess Engine Actually Works

A complete explanation, written for someone with no background in chess
programming. Nothing here assumes you know what a "transposition table" or
"null move" is. Read top to bottom and you should understand what every part
of this project does, why it exists, what we got right, and what we got
badly wrong.

Last updated: 2026-09-06.

---

## Part 1: What a chess engine actually is

A chess engine does one job: **given a position, output a move.** Everything
else is machinery in service of that.

Almost every strong engine, including ours, does it in two halves:

**1. Search** — "let me look ahead." Try a move, imagine the opponent's best
reply, imagine your best reply to that, and so on. This builds a tree of
possible futures.

**2. Evaluation** — "how good is this position?" At some point you have to
stop looking ahead and just judge a position with a number. Positive means
White is better, negative means Black is better, measured in *centipawns*
(100 centipawns = one pawn of advantage).

Search without evaluation is blind. Evaluation without search is shallow.
Strength comes from doing both well, and — critically — from **searching as
deeply as possible in the time available.**

That last point is the single most important idea in this whole document.
Nearly everything we did successfully was about searching deeper in the same
amount of time.

### Why depth matters so much

If you can only look 4 moves ahead and your opponent can look 11, they will
see traps you walk into. They see you hang a piece three moves before you do.
Every extra "ply" (one half-move — one player moving once) roughly doubles or
triples the work, but each one meaningfully increases strength.

Our engine went from depth 7 to depth 11-13 during one night of work. That is
an enormous difference in playing strength, and it came entirely from
searching *smarter*, not from faster hardware.

---

## Part 2: The competition we're building for

- **Format**: AI Chessathon. Teams submit an engine; engines play each other.
- **Time control**: 120 seconds per side, plus 0.5 seconds added per move.
  In practice our games average **1.7-2.9 seconds of thinking per move**.
- **Hardware limits**: 1 CPU core, 2GB RAM, **no GPU**, no internet access
  during play.
- **Startup budget**: 90 seconds to load before the clock starts.
- **A crash or an illegal move loses the entire game.** This is why the code
  is paranoid about safety in places that look over-engineered.

An unusual and important detail: **games do not start from move 1.** Each game
begins from a pre-set position roughly 6-8 moves in, supplied by the
organisers. This turned out to matter a lot (see Part 8).

---

## Part 3: The architecture — what each file does

```
agent.py                  The entry point. The competition calls this.
deepblue/
  fastcore.py             Board representation, move generation, make/unmake
  fastsearch.py           Base evaluation + shared tables
  fastsearchNN.py         Search variants (the numbered experiments)
  eval_terms.py           Individual evaluation features, each testable alone
  see.py                  Static Exchange Evaluation (is this capture good?)
  zobrist.py              Position hashing (for the transposition table)
  time_manager.py         How long to think about each move
  opening_book.json       Pre-computed answers for known starting positions
tools/
  sprt_gate.py            The testing harness (THE most important tool)
  regression_fast_variant.py   Correctness tests (32 positions)
  package_submission.py   Builds the .zip we upload
  build_opening_book.py   Pre-analyses tournament starting positions
nnue_lab/                 Neural network experiments (largely abandoned)
```

### agent.py — the front door

The competition imports this once and calls `get_move(fen, time_left)` for
every move. It does four things:

1. **Picks a guaranteed-legal fallback move before doing anything else.**
   If literally everything after this crashes, we still return a legal move
   rather than forfeiting. This is deliberate paranoia.
2. **Checks the opening book** — if we've seen this exact position before and
   pre-computed an answer, play it instantly (zero thinking time).
3. **Otherwise, runs the search** with a time budget.
4. **Records positions for repetition detection** — both the position we were
   given and the position our move creates, because the referee can declare a
   draw by repetition on either, and we'd otherwise draw won games without
   noticing.

### fastcore.py — the board

Chess positions here are stored as **bitboards**: a 64-bit number per piece
type, where each bit represents a square. "Where are all the white knights?"
is one integer. This makes operations like "which squares does this bishop
attack" fast bit arithmetic rather than loops.

The whole engine is compiled with **Numba**, which turns Python into machine
code. This is why startup takes 30-60 seconds — it's compiling. That cost is
paid inside the 90-second startup budget, deliberately, so it never eats into
game time.

---

## Part 4: The search, explained from scratch

### Step 1: Minimax — the basic idea

You want the best move. Your opponent wants the best move for them. So:

- I pick the move that **maximises** my score
- Assuming they then pick the move that **minimises** it
- Assuming I then maximise again... and so on.

This is called **minimax**. Our code uses an equivalent formulation called
**negamax**, which is the same thing written more compactly (it flips the sign
at each level instead of alternating between max and min logic).

The problem: the tree explodes. About 35 legal moves per position means
35 → 1,225 → 42,875 → 1.5 million positions by depth 4. You cannot search
deeply this way.

### Step 2: Alpha-beta pruning — the first big saving

Here's the key insight, in plain terms:

> If I'm considering move A, and I discover that move A lets my opponent reach
> a position better for them than something I've *already guaranteed* myself
> with move B, I can stop analysing move A entirely. It doesn't matter exactly
> how bad it is — I already know I won't choose it.

This is **alpha-beta pruning**. "Alpha" is the best score I've guaranteed so
far; "beta" is the best the opponent has guaranteed. When they cross, stop.

Alpha-beta can cut the work from 35^depth down to roughly 35^(depth/2) —
letting you search roughly **twice as deep** for the same effort. It is
mathematically guaranteed not to change the answer. It's pure free speed.

**Crucially, alpha-beta only pays off if you try good moves first.** If you
examine the best move first, you establish a high "alpha" immediately and can
cut everything else quickly. If you examine moves in a bad order, you prune
almost nothing. This is why *move ordering* is so important, and it's a theme
that comes back repeatedly below.

### Step 3: Quiescence search — don't stop mid-fight

Say you search 6 moves deep and stop. The last move in your line was "I
capture their queen with my knight." You evaluate: "I'm up a queen, fantastic!"

But you stopped one move too early. Their pawn recaptures your knight next
move. You're actually losing material. This is the **horizon effect** — you
can't see just past the edge of your search.

The fix: when you reach your depth limit, **don't stop if there are still
captures available.** Keep searching *only* captures and threats until the
position is "quiet." That's **quiescence search**, and our engine has had it
from early on. Without it, an engine hangs pieces constantly.

### Step 4: Iterative deepening — search depth 1, then 2, then 3...

Rather than committing to "search depth 10," the engine searches depth 1,
then depth 2, then 3, and so on until time runs out. This sounds wasteful
(you redo work) but it isn't, because:

- You always have a usable answer when the clock stops
- The shallow searches tell you which moves look good, so the deeper searches
  can try those first — which, per the alpha-beta point above, makes them
  dramatically faster

### Step 5: The transposition table — remember what you've seen

Different move orders reach the same position (1.e4 e5 2.Nf3 and 1.Nf3 e5
2.e4 are identical). Rather than re-analysing, we store results in a big hash
table keyed by a **Zobrist hash** (a near-unique 64-bit fingerprint of a
position). This is a large speedup and also improves move ordering, because
the table remembers which move was best last time.

### Step 6: PVS (Principal Variation Search)

A refinement of alpha-beta. Once you've searched the first move properly, you
*assume* it's the best and test the remaining moves with a deliberately narrow
window that can only answer "is this better than what I have — yes or no?"
That's much cheaper than computing exact scores. If one surprises you by
coming back "yes," you re-search it properly. Most of the time it doesn't.

---

## Part 5: Every feature we tested, and what happened

This is where the real story is. Each of these is a technique used by strong
engines. For each: what it does, and what actually happened when we tested it.

### SHIPPED — features that are in the engine now

**Null Move Pruning (NMP)** — *+64 Elo*

The idea sounds absurd at first: **let the opponent move twice in a row.** If
your position is still good enough to cause a cutoff even after handing them a
free move, then your position is so strong that you don't need to search this
branch carefully at all. Skip it.

Why this is safe: it's used only as a "this is definitely good enough" proof,
never at critical nodes, never when in check, and never in pawn-only endings
(where being forced to move is often *bad* — the "zugzwang" situation, where
passing would help you, breaks the logic).

Result: **4.3x fewer nodes needed to reach the same depth.**

**Late Move Reductions (LMR)** — *+45 Elo combined with NMP*

Move ordering means the moves we try late are probably bad. So: **search them
shallower.** If a late move unexpectedly comes back looking good, re-search it
at full depth to be sure.

The re-search is what makes this safe — a wrong reduction costs a little time,
but never loses a move.

Result: **11.5x fewer nodes at the same depth and the same evaluation.**

**Aspiration Windows** — *+78 Elo (our single biggest win)*

Normally each iterative-deepening pass searches with a wide-open window
(-infinity to +infinity). But we already know roughly what the score is from
the previous iteration. So search a narrow band around it (±25 centipawns).
Narrow windows prune vastly more.

If the true score falls outside the band, the search tells us, we widen and
redo it. The safety detail that matters: a score that lands exactly on the
window edge was *clamped* by our window rather than genuinely found, so we
never accept it — otherwise we'd feed a wrong score into time management.

**Mate distance pruning, check extensions, futility pruning, reverse futility
pruning** — smaller, earlier wins, all still in the engine.

**The opening book** — see Part 8.

### REJECTED — features that genuinely made it worse

**Late Move Pruning (LMP)** — *5.0% score, −511 Elo. Catastrophic.*

Superficially similar to LMR, but crucially different: instead of searching
late moves *shallower*, it **skips them entirely, permanently.**

Why it destroyed us: it only works if your move ordering is trustworthy. If a
genuinely good move happens to be sorted 5th, LMP never looks at it — ever. Our
ordering is weak (see below), so good moves land late, and LMP threw them away.

Evidence: at identical depth, LMP changed the engine's chosen move in **5 of 7
test positions** while using 4-5x fewer nodes. It wasn't searching the same
tree more efficiently — it was searching a *different, worse* tree.

**An important lesson from this**: LMP made the engine reach "depth 13" instead
of 11, and I initially reported that as progress. It was fake depth — the
number went up because the tree had been gutted. **Depth and node counts are
not proxies for strength.** Only games are.

**ProbCut** — *45%, −35 Elo*

Do a quick shallow search with a raised bar; if a capture clears it, assume it
would clear the real bar too and cut. It's redundant with NMP (which already
handles "this position is clearly good enough"), so it added cost without
adding information.

**Razoring** — *structurally inert*

At shallow depth, if the position looks hopeless, skip to quiescence. We
implemented it and measured a **0.1% change in nodes** — its trigger condition
essentially never fires here. Not worth testing in games.

**Texel tuning of the evaluation** — *27%, decisively bad*

We fitted our evaluation's parameters to millions of real Stockfish
evaluations using gradient descent. The result predicted Stockfish's judgments
**27% better** — and played dramatically **worse**.

The explanation is genuinely interesting: our search's pruning margins
(futility, razoring thresholds) were tuned around the old evaluation's scale.
Changing the evaluation's scale silently broke all of them. **Being more
accurate in isolation is not the same as being stronger in a system.**

**Singular extensions** — needs depth ≥8-10 to fire, which we couldn't
reliably reach. Genuinely not viable here yet.

### The one that explains a lot: SEE is not wired in

**Static Exchange Evaluation (SEE)** answers "if I capture here and we trade
everything on that square, do I come out ahead?" MVV-LVA — what we currently
use — only asks "is the captured piece valuable?" It cannot tell whether the
piece is *defended*.

So "rook takes knight" looks great to our ordering even when a pawn recaptures.

`deepblue/see.py` exists, is implemented, and was validated on 8,045 captures
with zero errors — **and nothing calls it.** This is very likely why LMP failed
and why several ordering-dependent techniques underperform. Testing it now.

---

## Part 6: The mistakes we made

Being honest about these matters more than the wins, because they're the
reason the wins took so long to find.

### Mistake 1: The 60-millisecond testing bug (by far the worst)

**Every test this project ever ran used 60 milliseconds per move.** That was
the default in the old testing script and nobody ever overrode it.

At 60ms, the engine reaches **depth 4.**

Now look at what that means for the features we were testing:

- **NMP** reduces the search by 3 plies. At depth 4, the "null move" search
  happens at depth 4−1−3 = **0**, i.e. it immediately drops into quiescence.
  The technique did nothing at all. It was recorded as "40.6%, rejected."
- **LMR** reduces late moves. At depth 4, reducing leaves depth 3 — no tree to
  save effort in, while paying the full re-search cost. Rejected 5-6 times.
- **LMP, razoring, SEE pruning, ProbCut, singular extensions** — all
  depth-gated. All effectively inert. All rejected.

**We spent an entire night concluding "these standard techniques don't work in
our engine," when in reality none of them had ever executed.** NMP and LMR are
two of the largest techniques in all of computer chess and were simply absent
from our engine as a result.

Fixing this one bug produced +45 and +78 Elo within hours.

The general lesson: **when many well-established techniques all measure
neutral, suspect your measurement, not the techniques.**

### Mistake 2: Sinking hours into NNUE

NNUE is a neural-network evaluation used by modern top engines. We spent a long
stretch trying to train one, and it went nowhere for reasons that were
foreseeable:

- Real NNUE training uses **hundreds of millions to billions** of positions.
  We could process about 4.6 million.
- Training needs a GPU. **This machine has Intel integrated graphics and a
  CPU-only PyTorch build** — no CUDA at all.
- We repeatedly ran out of RAM. One run silently thrashed to disk (29,000
  page-faults/second) and would never have finished; it looked like "slow
  training" rather than a dead run.
- Even a *successful* net would then have needed substantial engineering to
  integrate into the search, which never started.

Meanwhile the search work that eventually produced +123 Elo was sitting
untouched. The NNUE work was abandoned — correctly, but far too late.

### Mistake 3: Trusting depth and node counts as progress

Covered above under LMP. A change that makes the depth number go up can be
making the engine *much* weaker. Only game results count.

### Mistake 4: Contaminated tests

At one point a stray filesystem search process ran for **six hours**, competing
for CPU and corrupting the timing of several test runs. Separately, several
tests were run while NNUE training was saturating the machine. Timing-based
tests are only valid on an otherwise-idle machine — which is part of why the
new harness prefers deterministic fixed-depth testing where possible.

---

## Part 7: How we test now (and why the old way was broken)

Testing is the thing that decides what ships, so getting it right matters more
than any single feature.

### The problem with the old approach

- 60ms per move (depth 4) — as covered, this invalidated most conclusions
- Fixed number of games (240 or 480), then eyeball the percentage
- Single-threaded — one game at a time, which is why 60ms was chosen
- Results swung 20-30 percentage points between identical runs due to timing
  noise, which the project itself had documented but worked around rather than
  fixed

### The new harness: `tools/sprt_gate.py`

**1. It runs games in parallel** across CPU cores. This is what makes testing
at realistic speed affordable at all.

**2. It can test at fixed depth**, which is completely deterministic — the same
position always produces the same move, so re-running a test gives the same
answer instead of a different one. (For techniques whose whole purpose is
*saving time*, like NMP and LMR, we test on the clock instead, because at fixed
depth their benefit is invisible by construction.)

**3. It uses SPRT** (Sequential Probability Ratio Test) instead of a fixed
number of games. Rather than "play 240 games then look," it computes after
every game how strong the evidence is, and stops as soon as the answer is
clear either way. This is why LMP was caught in **20 games** instead of 240.

**4. It uses the real tournament starting positions** as test openings, so
we're measuring strength in the kind of position we're actually dealt.

**5. It was validated against itself.** Running the harness with the *same
engine on both sides* produces exactly 50.0% (+4 =8 −4, perfectly symmetric).
If it hadn't, every number it produced would be worthless.

### Current testing standard

- **2000 ms per move** — matching real competition pace (1.7-2.9s average)
- 100+ games minimum before shipping anything, with the score stable across
  checkpoints, or a formal SPRT decision
- 32/32 correctness regression must pass first, always
- Each new feature tested **against the current champion**, not an old
  baseline, so we measure the actual increment

---

## Part 8: The opening book discovery

Every competition game starts from a supplied position 6-8 moves in, not from
move 1. Reviewing 30 real games revealed something exploitable:

- Round 1 and round 14 started from **byte-identical positions**
- Round 10 and round 23 likewise
- Rounds 25→26 and 9→28 were each **one legal move apart**

The organisers draw starting positions from a **small, reused pool.** Since we
have unlimited time *before* a game but only seconds during one, we can
pre-analyse those positions offline far more deeply than we ever could at the
board, and store the answers.

`deepblue/opening_book.json` now holds 44 positions — 28 actually seen in real
games plus 16 predicted branches in the same opening families (the pool is
heavily weighted toward the Closed Sicilian). A book hit costs **zero seconds**
and plays a move analysed to greater depth than live play could reach.

The lookup is deliberately safe: any miss, or any stored move that isn't legal
in the current position, falls straight through to the normal search.

---

## Part 9: Where things stand

**Current engine**: `fastsearch71` = the previous engine + null move pruning +
late move reductions + aspiration windows, plus the opening book.

**Measured improvement tonight**: roughly **+123 Elo** combined, and depth 7 →
depth 11-13 at the same clock.

**Being tested**: SEE-based move ordering — likely important, since weak
ordering is the diagnosed root cause behind several failures.

**Known opportunity**: the **time manager is stopping early.** It predicts the
next iteration's cost and bails if it won't fit, but it assumes each iteration
costs at least 2x the previous one. With the new pruning, iterations grow more
slowly than that, so the engine is quitting with most of its clock unused — on
one test position it used 359ms of a 2000ms budget. That is free strength once
fixed.

**Still queued**: continuation history and capture history (both improve move
ordering), mobility evaluation (already written in `eval_terms.py` and never
called), king-safety attacker weighting, and a retest of ProbCut at proper
time control.

---

## Appendix: A glossary

- **Ply** — one player making one move. "Depth 10" = 10 plies = 5 moves each.
- **Centipawn** — 1/100th of a pawn, the unit of evaluation.
- **Node** — one position examined during search.
- **Cutoff** — abandoning a branch early because it can't affect the result.
- **Move ordering** — the sequence in which moves are examined. Good ordering
  makes alpha-beta pruning dramatically more effective.
- **Horizon effect** — misjudging a position because something decisive happens
  just past your search depth.
- **Zugzwang** — a position where being forced to move hurts you. It's why null
  move pruning is disabled in pawn endings.
- **Elo** — the rating scale. +100 Elo means winning roughly 64% of games.
- **SPRT** — a statistical test that stops as soon as evidence is conclusive.
- **Regression suite** — 32 fixed positions with known-correct answers, used to
  catch outright bugs before any strength testing.
