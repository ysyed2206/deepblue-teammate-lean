"""Deep Blue - the submission entrypoint.

The platform imports this file once per game and calls get_move for each of our
moves. Import time runs inside a 90 second budget before the clock starts, so
all table construction happens at import. (This file previously stated 60s in
one place and 90s in another; 90s is the published figure.)

WARM-UP BUDGET, measured against the platform's own reported INIT column
rather than assumed. An earlier note here claimed the competition runner was
faster than the dev laptop; the platform's numbers say otherwise:

    round 30   fastsearch57 (1.7)   platform 31.0s      laptop 33.6s
    rounds 31-35  fastsearch71 (1.8)  platform 40.4-46.9s  laptop 39.5s

So the runner is comparable to the dev laptop, marginally SLOWER for the
larger build -- not faster. About half the 90s budget is now consumed, and
every added feature costs compile time (NMP + LMR + aspiration together moved
this from ~33s to ~40s). Numba compile time scales with the size of the
recursive search functions, so this is a real budget to watch as features
accumulate, not a formality. If it approaches ~65s, stop adding search code
and start consolidating.

Two reliability commitments shape this file.

*   A known-legal fallback move is chosen before anything else can fail, and is
    returned on any exception. An illegal move or a crash loses the game
    outright, and qualification is a thirteen-game Swiss where one lost point
    outweighs a large amount of nominal strength.
*   Repetition bookkeeping records BOTH the position we are asked to move in
    and the position our own move creates. We are only ever shown the first,
    but the referee claims threefold automatically on either, so recording only
    our own turns loses won games to draws we never saw coming.

S1 engine, and why warm_up() is called eagerly below: FastEngine35's search/
quiescence/negamax are Numba-JIT-compiled on first use. That cost MUST land
inside this file's own import budget (90s per the published docs), not the
first move's clock -- so warm_up() runs here, at import time, rather than
being left to fire lazily on the first get_move() call.

fastsearch35 = the validated fastsearch18 search (PVS/RFP/TT/qsearch, see
EXPERIMENTS.md) with two evaluation terms added on top of the piece-square
tables: passed pawns and a king pawn-shield term. Measured 69.8% over 96
paired equal-time games against fastsearch18 (+51 =32 -13, every round above
50%), and 32/32 on the regression corpus. Both terms exist because three lost
qualification games traced directly to their absence -- two to mating attacks
after the engine pushed its own g/h pawns off its castled king, which a
piece-square table actively rewards, and one to a passed b-pawn walking to
b8 while the engine valued it at ~100cp.

fastsearch50 = fastsearch35 + futility pruning: at shallow depth (<=3 ply),
a quiet move that cannot raise alpha even after a generous per-depth margin
is skipped without being searched at all. This is the main-search sibling of
qsearch's delta pruning (which measured neutral). Nine other search
candidates were tried the night of 2026-09-04/05 -- gives-check-exempted
LMP+LMR, SEE-based capture ordering, aspiration windows, rook open-file/7th
rank, qsearch delta pruning, null move combined with the eval terms, and a
predictive time-management fix -- and all nine measured neutral or worse.
Futility pruning is the only one that measured positive, confirmed by TWO
independent 240-game runs (52.7% and 54.2%, 480 games combined, 53.4%
pooled, roughly +25 Elo) after the first run's result was treated as
provisional because a concurrent CPU load ran during it.

fastsearch52 = fastsearch50 + a fix for a foundational bug present in every
search variant since the project's first: iterative deepening trusted a
mate-range score to stop searching deeper at ANY depth, even depth 1, with
no minimum-depth or stability requirement. Confirmed directly from a real
qualification match's own log (round 20 vs "What the Fork", 2026-09-05):
from a completely winning king+queen+rook vs bare-king position, six
consecutive real moves showed the search completing at depth 1-3 in 0ms,
right as the engine cycled Rd3/Rd4/Rd3 into a threefold-repetition DRAW from
what should have been a trivial forced mate. Verified by replaying the exact
real position and clock: the old code reproduces the losing Rd4 move; the
fix (requiring depth >= 4 before trusting a mate score enough to stop early)
finds the correct winning Qc2 at the same position. Measured neutral (51.5%
over 240 games) against fastsearch50 in normal play, as expected -- the fix
only changes behaviour in the rare case a shallow false-early-stop would
otherwise have fired.

fastsearch57 = fastsearch52 + mate distance pruning + check extension, both
standard techniques used by essentially every strong engine (verified against
Ethereal's actual published search code, not just a description of it).
Tested individually first: mate distance pruning measured 50.6%, check
extension 50.4%, each over 240 games -- both within noise of exactly
neutral, matching a pattern where EVERY technique tried the night of
2026-09-05 past futility pruning and the mate-score fix landed within a
point of 50% at that sample size. That clustering looks like a
statistical-power limit (240 games cannot reliably resolve a true +10-20 Elo
effect) rather than genuine zero-effect from six well-established
techniques in a row. Bundling these two specific, already-individually-
tested, non-negative techniques was a deliberate exception to this
project's one-idea-per-candidate rule -- justified because neither
component is an unknown wildcard, so the risk that rule exists to prevent
(a bad interaction hidden behind a good average) is much lower here.
Measured 53.1% over 480 games (179-152-149, 19/30 rounds at or above 50%),
a clearer signal than either component alone.

fastsearch67 = fastsearch57 + null move pruning + late move reductions, and
the reason it exists is a measurement bug, not a new idea. Every paired test
this project ever ran used `paired_fast_variants*.py`, whose default is
60 ms/move -- at which this engine completes DEPTH 4. Every search technique
with a depth threshold therefore never fired in the test that rejected it.
NMP (NMP_MIN_DEPTH=4, R=3) searches its null line at depth 4-1-3 = 0, i.e.
straight into quiescence: its "40.6%" verdict measured nothing. LMR reducing
a depth-4 search leaves depth 3 with the re-search cost paid in full, which
is why five or six separate LMR attempts all "failed". Both techniques are
standard and large in every strong engine, and both were absent here.

Re-tested with a new gate (tools/sprt_gate.py: parallel, SPRT sequential
stopping, real tournament seed positions, and self-validated by scoring
identical engines at exactly 50.0%). Measured at 1000 ms/move -- still below
the competition's true 1.7-2.9 s/move average, which makes the estimate
conservative, since both techniques gain more with depth:

    NMP alone         59.2% over 60 games   (+64 Elo)
    NMP + LMR         55.6% over 80 games   (+39 Elo), 32/32 regression

Structurally, at a 2-second budget on the start position:

    fastsearch57   depth 7    1,096,265 nodes to reach depth 8
    fastsearch65   depth 8      256,116 nodes  (NMP, 4.3x fewer)
    fastsearch67   depth 11      60,348 nodes  (NMP+LMR, 18x fewer)

Four extra plies at the same clock, with identical scores at equal depth --
the node savings are not being bought with worse decisions. That depth is
the direct answer to the failure mode behind the round 28 and 29 losses,
both of which were slow positional slides invisible to a depth-7 search.

fastsearch71 = fastsearch67 + aspiration windows. Also previously dismissed
as "neutral", and for the same reason: at 60 ms/move there are only three or
four iterative-deepening iterations in an entire search, and a narrow window
around the previous score can save almost nothing. The technique only pays
once the iteration tree is large, which is precisely the regime NMP+LMR
opened up. Instead of searching every iteration on a full (-INF, +INF)
window, each iteration past depth 5 searches a +/-25cp band around the
previous score; a score landing on the window edge was clamped by the window
rather than found by the search, so it is never accepted -- the window widens
4x and the iteration is redone. That guard matters because a clamped score
would otherwise feed a wrong "best score" into both time management and the
mate-score early-stop rule.

    aspiration windows   61.0% over 100 games (+77.7 Elo), 32/32 regression

which is a larger gain than NMP+LMR itself, on top of NMP+LMR.

REJECTED on the same new gate, recorded so it is not retried blindly:
late move pruning measured 5.0% over 20 games (-511 Elo), decisively. This
engine tolerates techniques that REDUCE-then-verify (LMR re-searches anything
that beats alpha; NMP and ProbCut verify with a real search) but not ones
that DISCARD moves outright. The cause is move ordering: deepblue/see.py is
implemented and differentially validated on 8045 captures, and is not wired
into ordering anywhere, so "late" moves here are frequently good moves.
Razoring was implemented and measured structurally inert (0.1% node change).

fastsearch76 = fastsearch71 + a time-management fix, and it is the first
candidate in this project's history to reach a FORMAL SPRT decision rather
than being judged on a stable-looking percentage:

    222 games  +77 =102 -43   57.7%   +53.6 +/- 34 Elo   LLR +2.95  ACCEPT

The bug it fixes was introduced, indirectly, by the two features above.
The driver predicts the cost of the next iterative-deepening iteration and
stops if it will not fit in the soft budget:

    growth = min(6.0, max(2.0, iteration_ms / previous_ms))
    if elapsed + iteration_ms * growth > soft_ms:
        break

Two things then went wrong at once. The growth estimate is floored at 2.0,
which was fair for the old wide search but is far too pessimistic once NMP
and LMR cut the effective branching factor. And it is taken from a SINGLE
iteration ratio, while aspiration windows make individual iterations spiky:
one fail-high triggers a re-search, that iteration looks several times more
expensive than the trend, growth pins to its 6.0 ceiling, and the engine
quits immediately. Measured on a real tournament seed at a 2000 ms budget:
it used 359 ms, 18% of its clock, and stopped at depth 7.

The fix smooths the growth estimate instead of trusting one spiky sample,
lowers the floor to 1.3, and refuses to stop on the prediction alone while
less than half the budget has been spent. Over-running is bounded -- the
hard-time timer aborts and the last COMPLETED iteration's move is used --
whereas under-using the clock is a guaranteed loss of depth. Result: +1 ply
on exactly the positions that were quitting early, no regression elsewhere.

Worth recording as a general lesson: adding a feature can silently break a
heuristic elsewhere that was tuned around the old behaviour. Aspiration
windows were a large gain AND quietly cost some of it back through the time
manager, and nothing about the aspiration test itself would have revealed it.

fastsearch79 = fastsearch76 + a mobility evaluation term, and the notable
part is that the term was already written. deepblue/eval_terms.py has
contained mobility_white_relative -- pseudo-legal attack-square counts for
knights, bishops, rooks and queens, weighted 4/3/2/1 -- for the whole
project, validated, and NO shipped engine ever called it. The evaluation
until now was material + piece-square tables + passed pawns + a king pawn
shield, so "how many squares can my pieces actually reach" was a real gap.

    120 games  +45 =51 -24   58.8%   +61.4 +/- 48 Elo   32/32 regression

Worth recording how that number moved, because it is a lesson about reading
results early: it measured +117 Elo at 40 games and settled to +61 by 120.
Both figures are the same experiment; the first was simply noise-inflated.
Shipping on the 40-game number would have overstated the gain twofold.

Opening book (added 2026-09-05, on top of the search unchanged):
every real qualification PGN reviewed carries a [SetUp]/[FEN] header 6-8
moves deep rather than starting at move 1, and across the first 30 rated
games two pairs of seeds were BYTE-IDENTICAL across different rounds against
different opponents (round 1 == round 14, round 10 == round 23), plus two
more pairs one legal move apart (round 25 -> 26, round 9 -> 28) -- evidence
the tournament draws from a small, reused seed pool rather than a fresh
position every game. deepblue/opening_book.json holds all 28 distinct seeds
seen so far, each analyzed offline for 30-110s (reaching depth 9-11, versus
the depth 7-9 a live few-second move budget typically reaches) with a fresh
engine instance per position to avoid any TT/history cross-contamination
between unrelated seeds. get_move() checks this table before doing anything
else: a hit returns instantly (0ms, no search invoked, saving that time for
the moves that actually need it); a miss -- or a stale/illegal entry, which
this checks for explicitly -- falls straight through to the search exactly
as before. This is a pure additive lookup: it cannot make any position worse
than the unmodified search already would, only sometimes better plus free
clock time. Not a fix for any specific loss reviewed tonight (round 28 and
29's actual failures traced to a long positional slide starting 15-20 moves
later, not the opening choice) -- its value is the same small, compounding,
risk-free edge across every future game that lands on a known seed.

fastsearch86 = fastsearch79 + king_attack_danger_white_relative (added
2026-09-06). Round 36 (and the matching pattern in rounds 5, 8, 28, 29) was
level (+11) until the engine pushed its own kingside pawns (h5/h4/hxg3),
opening lines toward its own king, then collapsed to -127. The existing king
safety term only counts missing shield pawns; it has no notion of enemy
pieces actually massing near the king. king_attack_danger_white_relative
(deepblue/eval_terms.py, KING_DANGER_THRESHOLD=0, KING_DANGER_SCALE=512) adds
that: an attacker-weighted penalty once enemy piece pressure near the king
exceeds a threshold. Rebased directly onto fastsearch79 rather than the
originally-tested fastsearch84, which was built on fastsearch83 (the removed
puzzle-solving code) -- the eval addition itself is identical, only the base
differs.

Real-clock SPRT (fastsearch84 vs fastsearch83, 2000ms/move, 7 workers),
stopped at 180 games to prioritize the time-allocation fix rather than run to
full SPRT completion:
    20 games  47.5%   -17.4 Elo
    40 games  53.8%   +26.1 Elo
    60 games  57.5%   +52.5 Elo
    80 games  53.8%   +26.1 Elo
   100 games  52.5%   +17.4 Elo
   120 games  53.8%   +26.1 Elo
   140 games  52.9%   +19.9 Elo
   160 games  51.6%   +10.9 Elo
   180 games  52.8%   +19.3 Elo  +/-38, LLR +0.50 (bound +2.94, not reached)
Consistently above 50% at every checkpoint after game 20, never negative.
Honest caveat: the 95% interval still straddles zero, so this is a small
probable gain shipped on trend rather than a decisive SPRT accept.

fastsearch91 = fastsearch86 + gravity history (added 2026-09-06). The old
quiet-move history rule was `history[piece, to_square] += depth * depth`:
it only ever GROWS, and it never records that a quiet move was tried and
FAILED. Two consequences. It saturates -- once several quiets reach
HISTORY_CEILING they are indistinguishable, and the ordering information
that history exists to provide is gone. And a move that repeatedly fails to
cause a cutoff keeps whatever rank it earned once.

The gravity rule replaces it with a self-limiting update that also carries
penalties, which is the half that does most of the work in Ethereal and
Stockfish:

    bonus  = min(depth * depth, HISTORY_BONUS_MAX)      # 1536
    entry += bonus - (entry * bonus) // HISTORY_MAX     # 16384

so each update pulls the entry TOWARD +/-HISTORY_MAX instead of past it, and
every quiet already tried at that node is penalised by the same formula with
the sign flipped. Those already-tried quiets cost nothing to find: pick_best
swaps its chosen move into position `start`, so at a cutoff on order_index
the moves at pseudo[ply][0:order_index] are exactly the ones already tried,
in order -- no extra bookkeeping array was needed. HISTORY_MAX is bounded
far below ORDER_KILLER_2 (2**20 - 1), so no amount of accumulated history
can promote a quiet above a killer, capture or promotion.

Measured against fastsearch86, --mode time --move-ms 2000, 7 workers:

    20 games  +10 =7  -3   67.5%  +127.0 +/-129 Elo
    40 games  +17 =16 -7   62.5%   +88.7 +/- 85 Elo
    60 games  +25 =22 -13  60.0%   +70.4 +/- 71 Elo
    80 games  +32 =30 -18  58.8%   +61.4 +/- 61 Elo    32/32 regression

Fixed-time mode is the correct control here: this changes move ordering
only, never the time budget, so both sides search under an identical 2000 ms
per move. (A node-fraction TIME-management candidate tested the same week
looked like +107 Elo in this mode and was discarded precisely because that
mode was invalid for it -- it let the candidate spend up to 2520 ms against
the baseline's 2000 ms. That failure mode cannot apply here.)

The settling pattern matches the mobility term, which measured +117 Elo at
40 games and converged to +61 by 120; treat +61 as the estimate, not +127.
This targets the failure the games actually show -- losing gradually to less
accurate moves rather than to a single blunder -- because better ordering
means more cutoffs, which means more depth on EVERY move.


fastsearch97 = fastsearch91 + SEE pruning inside quiescence (added
2026-09-07), and it came from counting rather than guessing. Profiling the
counters on real positions from rounds 41-44 showed QUIESCENCE IS 59-71% OF
EVERY NODE THIS ENGINE SEARCHES -- and it was essentially unpruned. Every
search technique tested before this one (singular extensions, ProbCut, TT
sizing, LMR tuning) operates on the other 30-40%. The larger half of the
tree had never been touched.

Quiescence played out every capture in full, including ones that simply lose
material. A capture whose static exchange evaluation is negative loses
material by force and its subtree almost never raises the stand-pat score,
so the subtree is now skipped entirely.

This is NOT the SEE ORDERING that was rejected here at 41.7%. That paid the
SEE cost to re-sort the same set of nodes -- pure overhead against a
marginal ordering gain. This pays the same cost to DELETE nodes, in the
biggest part of the tree. Same primitive, opposite economics.

SEE must be evaluated on the pre-move position, so a pruned move never
increments legal_tactical and the any_legal_move stalemate probe can run
when it did not strictly need to. That probe is correct in every case and
correctness there is worth more than avoiding it.

Measured effect, 2000 ms budget, idle machine:

    position        depth        quiescence share
    r44 move 42     9 ->  9        62% -> 47%
    r41 midgame     8 ->  9        59% -> 45%
    r42 seed        9 -> 10        71% -> 52%

    fastsearch97 vs fastsearch91, --mode time --move-ms 2000, 7 workers:
     20 games  60.0%  +70.4 +/-110 Elo
     40 games  55.0%  +34.9 +/- 81 Elo
     60 games  51.7%  +11.6 +/- 65 Elo
     80 games  53.1%  +21.7 +/- 57 Elo
    100 games  53.5%  +24.4 +/- 51 Elo
    120 games  54.6%  +31.9 +/- 47 Elo    32/32 regression

Shipped on the trajectory rather than on formal significance, and the
distinction that justified it is worth recording: at 80 games this read
53.1%, the IDENTICAL figure the failed INCREMENT_FRACTION candidate showed
at 80 games. That one decayed to 50.5% by 100. This one held (53.5%) and
then rose (54.6%). Same reading, opposite behaviour under more evidence --
which is exactly why marginal results get more games instead of a decision.


fastsearch105 = fastsearch97 + continuation (counter-move) history (added
2026-09-07). How good a quiet move is depends on what the opponent JUST
played, not only on which piece moves where: retreating a knight is a
different move after it was attacked than in a quiet position. Alongside
history[piece][to] this keeps continuation_history[prev_piece][prev_to]
[piece][to] and adds the two when ordering quiets.

The move that led to each node is recorded in a per-ply array (prev_moves,
same shape as path_hashes) rather than threaded through every recursive call
as extra scalars -- order_moves already receives ply, so it can look it up.

The update rule deliberately MATCHES the gravity rule fastsearch91
introduced (entry += bonus - entry*bonus/HISTORY_MAX, with penalties for the
quiets already tried at that node). Giving the new table the old
`+= depth*depth` rule would have bundled two separate ideas into one
candidate, and gravity is the rule that measured +67 Elo.

Predicted from node counts BEFORE testing, at fixed depth 10:

    position        fastsearch97   fastsearch105    change
    r44 move 42          586,572         590,188     +0.6%
    r41 midgame        1,046,642         811,323    -22.5%
    r42 seed             608,643         413,998    -32.0%

Up to a third fewer nodes for the SAME depth is what better ordering looks
like -- and it is the exact inverse of singular extensions, which needed
43-55% MORE nodes and duly failed at 47.5%.

    fastsearch105 vs fastsearch97, --mode time --move-ms 2000, 7 workers:
    20 games  55.0%  +34.9 +/-131 Elo
    40 games  57.5%  +52.5 +/- 85 Elo
    60 games  56.7%  +46.6 +/- 67 Elo
    80 games  55.6%  +39.3 +/- 58 Elo    32/32 regression

Four checkpoints between 55 and 57.5 with no decay. Worth contrasting with
the eight time-management candidates tested the same night, which all either
declined from their opening reading or sat flat: ordering changes keep
winning here and clock changes keep not, because ordering buys depth on
every move while reallocation only moves it between moves.

ALSO in this build (agent.py, not the search): if the position has exactly
one legal move, it is played immediately without searching. Listed on the
Chess Programming Wiki as a standard early-termination trigger and absent
here until now -- a forced recapture or single check evasion was costing a
full move's budget to "decide" something with no alternative. Verified at
0 ms against a full search, with the same two repetition records the searched
path makes (it reuses _record_after rather than reimplementing it, since the
referee claims threefold on either position). Rounds 39, 41 and 44 all ended
in long forced sequences, which is where this pays.


fastsearch106 = fastsearch105 + a bishop-pair bonus (added 2026-09-07).
Two bishops cover both square colours and complement each other in a way no
other pair of minors does, and the advantage grows as the board opens. The
base evaluation prices every bishop identically, so until now trading one of
them off cost the engine precisely nothing. BISHOP_PAIR_BONUS is 28cp,
phase-flat: scaling it by phase is a separate idea and would need its own
measurement rather than being smuggled in alongside this one.

    fastsearch106 vs fastsearch105, --mode time --move-ms 2000, 7 workers:
     20 games  55.0%  +34.9 +/-99 Elo
     40 games  57.5%  +52.5 +/-73 Elo
     60 games  54.2%  +29.0 +/-66 Elo
     80 games  53.8%  +26.1 +/-54 Elo
    100 games  56.0%  +41.9 +/-49 Elo
    120 games  55.8%  +40.7 +/-45 Elo    32/32 regression

Deliberately run to 120 rather than shipped at the 80-game bar, because
53.8% at 80 games is a genuinely ambiguous reading in this project: rook
placement read 51.9% and was discarded, INCREMENT_FRACTION read 53.1% and
decayed to 50.5%, and SEE pruning read 53.1% and climbed to 54.6%. Same
neighbourhood, three different truths. Holding every candidate in that band
to the same 120 games is what keeps the series honest -- applying a looser
standard to a candidate one happens to like is how a test suite quietly
stops measuring anything.


fastsearch109 = fastsearch106 + doubled and isolated pawn penalties (added
2026-09-07). A piece-square table scores a pawn purely by the square it
stands on, so two pawns stacked on one file score exactly as well as two
abreast, and a pawn with no neighbours scores as well as one in a chain.
Both are long-term structural weaknesses a search cannot discover
tactically -- they cost material many moves beyond the horizon, which is
precisely the shape of losing gradually rather than to a single blunder,
the failure mode these games actually show.

DOUBLED_PAWN_PENALTY 12 per extra pawn on a file, ISOLATED_PAWN_PENALTY 15
per pawn with no friendly pawn on either adjacent file.

    fastsearch109 vs fastsearch106, --mode time --move-ms 2000, 7 workers:
    20 games  55.0%  +34.9 +/-131 Elo
    40 games  56.2%  +43.7 +/- 90 Elo
    60 games  55.0%  +34.9 +/- 69 Elo
    80 games  59.4%  +65.9 +/- 60 Elo    32/32 regression

Shipped at 80 rather than held to 120 like bishop pair and SEE pruning: the
120-game rule exists for readings in the ambiguous ~53% band, where three
separate candidates this session went on to 51.9% (discarded), 50.5%
(decayed to nothing) and 54.6% (shipped). At 59.4% the interval [+6, +126]
excludes zero outright, which is a different situation and does not need the
extra games.


fastsearch114 = fastsearch109 + rook placement + knight outposts + a 4M-entry
transposition table (added 2026-09-07), and the interesting part is that all
THREE had already been tested individually and discarded:

    rook open/semi-open file + seventh rank   51.9% over 80 games
    knight outposts                           51.2% over 80 games
    transposition table 20 -> 22 bits         50.8% over 60 games

Each was called neutral. Together they measure +61 Elo.

That is not a contradiction, it is a statement about the MEASUREMENT rather
than the features. This harness resolves roughly +/-30 Elo at 80-120 games,
so a change genuinely worth +10 is invisible on its own -- it reads 51% and
gets discarded. Three such changes stacked are worth +30, which is inside
what the gate can see. The features were never worthless; the instrument
could not read them.

    fastsearch114 vs fastsearch109, --mode time --move-ms 2000, 7 workers:
    20 games  60.0%  +70.4 +/-110 Elo
    40 games  62.5%  +88.7 +/- 77 Elo
    60 games  60.0%  +70.4 +/- 67 Elo
    80 games  58.8%  +61.4 +/- 61 Elo    32/32 regression

This deliberately breaks the one-idea-per-candidate rule, and the reason the
rule exists -- a harmful component hiding behind a good average -- is much
weaker here because every component had already been measured as not harmful
on its own. There is precedent in this engine: fastsearch57 bundled two
individually-inconclusive techniques and measured 53.1% over 480 games.

The consequence worth acting on: other candidates that read 50-52% with a
sound mechanism are NOT settled, and deserve a second bundle. Candidates that
read clearly negative (singular extensions 47.5%, delta pruning 38.8%, the
nine time-management attempts) are a different case -- those are not
invisible-small, they are harmful.


fastsearch116 = fastsearch114 + contempt (added 2026-09-07). The whole change
is one constant applied at two return sites:

    CONTEMPT = 20
    threefold repetition   return DRAW_SCORE -> return -CONTEMPT
    search-cycle heuristic return DRAW_SCORE -> return -CONTEMPT

DRAW_SCORE has always been 0, so a repetition scored exactly equal to an
unclear position and the engine accepted a draw whenever its own evaluation
was even slightly negative. Round 53 is the concrete cost: a threefold
repetition at MOVE 15 as White, from a position the engine rated -15. By its
own arithmetic the draw (0) beat -15, so it repeated deliberately. In a Swiss
where the team must climb, an early draw as White is nearly a loss.

Applied ONLY to repetition. Stalemate, insufficient material and the
fifty-move rule are genuinely drawn and still score exactly 0 -- a first
attempt replaced all thirteen DRAW_SCORE sites and failed four regression
positions that assert a terminal draw scores 0.

    fastsearch116 vs fastsearch114, --mode time --move-ms 2000, 7 workers:
     20 games  72.5%  +168.4 +/-131 Elo
     40 games  65.0%  +107.5 +/- 81 Elo
     60 games  59.2%   +64.4 +/- 68 Elo
     80 games  54.4%   +30.5 +/- 60 Elo
    100 games  53.5%   +24.4 +/- 54 Elo
    120 games  55.8%   +40.7 +/- 50 Elo    32/32 regression

TWO HONEST CAVEATS, recorded rather than buried:

* SELF-PLAY FLATTERS CONTEMPT. Both sides here are near-identical, so
  declining a draw from an equal position is close to a coin flip and the
  measurement is the friendliest available. In the real Swiss the opponents
  vary in strength, and refusing a draw against someone stronger converts a
  half point into zero. Strong engines scale contempt by opponent rating; we
  cannot, because we are not told who we are playing. Expect the tournament
  gain to be smaller than +41.
* THE VALUE 20 WAS CHOSEN, NOT MEASURED. It sits in the same category as the
  invented weights that failed on rook placement and knight outposts, with
  the difference that a single scalar can be tested directly. 10 and 35 are
  worth trying; 20 may be well off the peak.


fastsearch150 = fastsearch116 + razoring (added 2026-09-07). The mirror of
reverse futility pruning: RFP asks whether the static evaluation is far
enough ABOVE beta to assume a fail-high, razoring asks whether it is far
enough BELOW alpha that the node is hopeless. The difference that matters is
that razoring does not trust the static evaluation on its own -- quiescence
VERIFIES the node is still below alpha before anything is returned.

That verification puts it in the category this engine accepts. Reduce-then-
verify techniques have all worked here (null move +64, LMR +39, SEE
quiescence pruning +32); outright discards have not (LMP scored 5%, which is
a signature rather than noise).

Its recorded rejection is VOID, and this is the third time that has mattered.
The 60ms-era gate ran at depth 4, where a depth<=3 condition barely fires and
the quiescence verification dominates whatever remains. Null move and LMR
were recovered from the same broken gate and are worth +64 and +39.

    fastsearch150 vs fastsearch116, --mode time --move-ms 2000, 7 workers:
    20 games  45.0%  -34.9 +/-121 Elo
    40 games  55.0%  +34.9 +/- 88 Elo
    60 games  60.0%  +70.4 +/- 71 Elo
    80 games  60.0%  +70.4 +/- 58 Elo    32/32 regression

RAZOR_MAX_DEPTH 3, RAZOR_MARGIN_BASE 180, RAZOR_MARGIN_PER_DEPTH 120.

"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import chess

from deepblue import fastsearch180n
from deepblue.fastcore import from_fen
from deepblue.fastsearch180n import FastEngine180n
from deepblue.time_manager import allocate

# Opening prep: every real qualification PGN reviewed so far starts from a
# [SetUp]/[FEN] position 6-8 moves deep, not move 1 -- and across 30 real
# rated games, two pairs were BYTE-IDENTICAL seed positions from different
# rounds against different opponents, plus two more pairs one legal move
# apart. That means the tournament draws from a small, reused seed pool, and
# a position analyzed offline (minutes of search, no clock pressure) is
# available for instant, zero-clock-cost recall if we see it again. This is
# a pure lookup: a miss falls straight through to the normal search below,
# so it can only ever save time or improve the first move, never cost
# anything -- the existing fallback/repetition machinery is untouched.
try:
    _book_path = Path(__file__).resolve().parent / "deepblue" / "opening_book.json"
    _OPENING_BOOK = json.loads(_book_path.read_text(encoding="utf-8"))
except Exception:
    _OPENING_BOOK = {}


def _book_move(fen: str) -> str | None:
    try:
        key = " ".join(fen.split(" ")[:4])
        entry = _OPENING_BOOK.get(key)
        return entry["best_move"] if entry else None
    except Exception:
        return None

# The published time control is 120 s + 0.5 s/move, but the agent API is only
# told the clock, never the increment. Assuming a fixed 500 ms is a real
# hazard: if the true increment is smaller, the engine spends the difference
# every move, drains its clock and drops into panic mode playing unsearched
# moves. That was measured, at a 100 ms increment, as a 20-0 loss to an
# otherwise identical build. So the increment is OBSERVED from how the clock
# actually moves, and the published value is only the starting assumption.
DEFAULT_INCREMENT_MS = 500

_engine = FastEngine180n()
fastsearch180n.warm_up(_engine)  # forces the Numba JIT compile now, inside the import budget
_moves_played = 0
_increment_ms = float(DEFAULT_INCREMENT_MS)
_previous_clock_ms: float | None = None
_previous_spent_ms: float = 0.0


def _observe_increment(time_left_ms: int) -> None:
    """Infer the real increment from consecutive clock readings.

    Between two of our turns the referee subtracts what we spent and adds the
    increment, so the increment is what the clock gained beyond our spending.
    """
    global _increment_ms, _previous_clock_ms
    if _previous_clock_ms is not None:
        observed = time_left_ms - (_previous_clock_ms - _previous_spent_ms)
        if 0.0 <= observed <= 5000.0:
            # Track the smallest credible observation: budgeting for less
            # increment than we get is safe, budgeting for more is a flag.
            _increment_ms = min(_increment_ms, observed)
    _previous_clock_ms = float(time_left_ms)


def _fallback(board: chess.Board) -> str:
    """A legal move chosen without searching. Never returns an illegal move."""
    for move in board.legal_moves:
        return move.uci()
    # No legal moves: the game is over and the referee will not ask again.
    return "0000"


def _record_after(board: chess.Board, uci: str) -> None:
    """Record the position our own move creates.

    Bookkeeping must never cost us a game we could otherwise play, so every
    failure here is swallowed. A missed history entry weakens repetition
    detection; a raised exception would lose the game.
    """
    try:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            return
        board.push(move)
        try:
            _engine.record_game_position(from_fen(board.fen()))
        finally:
            board.pop()
    except Exception:
        pass


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation."""
    global _moves_played
    try:
        board = chess.Board(fen)
    except ValueError:
        return "0000"

    fallback = _fallback(board)
    if fallback == "0000":
        return fallback

    # Only one legal reply: play it without searching. Listed on the Chess
    # Programming Wiki as a standard early-termination trigger, and absent
    # here until now -- a forced recapture or a single check evasion was
    # costing a full move's budget to "decide" something that has no
    # alternative. The saved time stays on the clock for a move that can
    # actually use it, which matters most in the long forced sequences that
    # end our games (rounds 39, 41 and 44 all finished in one).
    #
    # Repetition bookkeeping still has to happen, so this records the position
    # exactly as the normal path does before returning.
    _one_reply = list(board.legal_moves)
    if len(_one_reply) == 1:
        only_uci = _one_reply[0].uci()
        try:
            _observe_increment(time_left_ms)
            # Same two records the searched path makes: the position we were
            # handed, and the one our move creates. _record_after is reused
            # rather than reimplemented -- the referee claims threefold on
            # either, and duplicating this logic is how it drifts out of sync.
            _engine.record_game_position(from_fen(fen))
            _record_after(board, only_uci)
        except Exception:
            traceback.print_exc(file=sys.stderr)
        _moves_played += 1
        return only_uci

    chosen = fallback
    started = time.monotonic()
    try:
        _observe_increment(time_left_ms)
        position = from_fen(fen)
        # Every real call records the position we were handed. There is no
        # de-duplication: a position that recurs is exactly the thing we need
        # to count, and suppressing the repeat defeated the whole mechanism.
        # The referee treats the supplied FEN as the start of a game, so no
        # history exists before our first call.
        _engine.record_game_position(position)

        book_uci = _book_move(fen)
        book_move = chess.Move.from_uci(book_uci) if book_uci is not None else None
        if book_move is not None and book_move in board.legal_moves:
            assert book_uci is not None
            chosen = book_uci
            _moves_played += 1
            print(f"deepblue: opening book move {book_uci}", file=sys.stderr)
        else:
            soft_ms, hard_ms = allocate(time_left_ms, int(_increment_ms), _moves_played)
            _moves_played += 1
            if hard_ms > 0.0:
                uci, score, depth, nodes, elapsed_ms = _engine.search(position, soft_ms, hard_ms)
                move = chess.Move.from_uci(uci) if uci is not None else None
                if move is not None and move in board.legal_moves:
                    chosen = uci
                    print(
                        f"depth {depth} score {score} nodes {nodes} "
                        f"time {elapsed_ms:.0f}ms "
                        f"nps {nodes / max(elapsed_ms, 1.0) * 1000:.0f}",
                        file=sys.stderr,
                    )
                elif uci is not None:
                    print(f"deepblue: search returned illegal {uci}", file=sys.stderr)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        chosen = fallback

    global _previous_spent_ms
    _previous_spent_ms = (time.monotonic() - started) * 1000.0
    _record_after(board, chosen)
    return chosen
