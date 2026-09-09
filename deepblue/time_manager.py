"""Clock allocation.

A flag is an immediate loss and is the most common self-inflicted one, so every
number here is chosen to fail toward returning early. Two budgets come out:

    soft  the point at which we stop *starting* a new iterative-deepening pass
    hard  the point at which we abandon the pass in progress and return the best
          move from the last completed one

The referee measures wall time around the whole call, including the JSON round
trip, so a fixed overhead is subtracted before anything else is computed.
"""

from __future__ import annotations

# Wall-clock cost of one move that is not search: protocol write, JSON encode,
# board construction, and the referee's own accounting. Measured, then doubled.
PROTOCOL_OVERHEAD_MS = 60.0

# Below this much clock we stop trying to play chess and just move.
PANIC_THRESHOLD_MS = 1_500.0
PANIC_BUDGET_MS = 30.0

# The clock is 120 s + 0.5 s/move. Assuming a fixed number of moves remaining is
# what makes engines flag in long games, so the estimate decays with the move
# number and never drops below a floor.
#
# Retuned from 42/18 after three lost qualification games (rounds 5, 7, 8) were
# all reached checkmate with 47-59% of the clock STILL UNSPENT -- 70 s, 56 s and
# 62 s in hand respectively. Simulating those games reproduces the old numbers
# exactly (42/18 spends 56.1% of the clock), and 28/12 lifts that to 76.2%,
# about +36% thinking time per move. The estimate is proportional to the clock
# that remains, so it asymptotes instead of flagging: simulated at 100 of our
# own moves it still never drops below ~9 s, against a 1.5 s panic threshold.
#
# 2026-09-05: a candidate fix to the iterative-deepening predictive stop
# (fastsearch49) would have made 28/12 genuinely unsafe -- real self-play (not
# the arithmetic simulation above) hit 1673 ms at its lowest point in a
# 140-move game, only 173 ms above the 1500 ms panic threshold. 34/15 was
# verified safe for that combination (2188 ms lowest). But fastsearch49 itself
# scored a neutral 47.5% over 240 paired games and was not shipped, so 34/15
# was reverted with it -- shipping only the more conservative clock value,
# with no matching depth benefit to justify giving back today's clock-usage
# gain, would have been change for its own sake. If fastsearch49 (or any other
# change to how much of the allocated budget the search consumes) is revisited,
# re-validate the safe MIN/BASE pair by REAL self-play again, not by reusing
# either of the two values recorded here -- they are safe only for the specific
# search behaviour they were each measured against.
#
# 2026-09-06: tried again, deliberately -- real games showed the champion
# spending 3-5s/move in the opening and only 0.70-0.81s/move in the late
# middlegame, and 28/12's own schedule explains why: it reaches its MIN floor
# by move ~32, then divides an already-shrinking clock by that same constant
# 12 for the rest of the game, which front-loads spending instead of easing
# into it. A 28->44 change was arithmetically verified to fix exactly this
# (simulated clock remaining at move 60: 3.51s at 28 vs 14.65s at 44). But
# real-clock SPRT (fastsearch86 vs itself, 28 vs 44, real game clock) came
# back flat: 47.5% at 20 games, 50.0% at 40 and 60 on the first run; a second
# run with per-game ply-length logging added (to rule out the test simply not
# reaching the range where the two schedules diverge) confirmed coverage was
# fine -- 19/20 games reached ply 60, averaging 105 plies -- and still opened
# at 47.5%. Two independent measurements, adequate depth confirmed, no
# effect. This is now the SECOND time changing this schedule alone has come
# back neutral (see 2026-09-05 above). The schedule is not the lever;
# whatever time is banked by a more conservative early-game budget is not
# translating into won games. Effort moved to fastsearch87 (stability-aware
# time management: read the search's own best-move/score stability instead
# of a move-count schedule) instead of a third attempt at this constant.
MIN_MOVES_REMAINING = 12
BASE_MOVES_REMAINING = 28

# Fraction of the increment we are willing to spend as if it were already banked.
INCREMENT_FRACTION = 0.75

# The hard limit is a multiple of the soft limit, capped as a fraction of the
# whole remaining clock so one hard position can never eat the game.
# Aborting a pass costs nothing - the last completed pass is always retained -
# so the hard limit can sit close to the soft one and act as a real spending
# cap rather than a distant safety net.
HARD_MULTIPLIER = 1.4
HARD_CLOCK_FRACTION = 0.30
ABSOLUTE_CLOCK_FRACTION = 0.90

# Each iterative-deepening pass costs some multiple of the one before. The
# multiple is measured as the search runs, because it varies with the position:
# a pass that opens a tactical line can cost five times its predecessor. These
# bounds keep a single wild estimate from either stalling or overrunning.
MIN_ITERATION_GROWTH = 2.0
MAX_ITERATION_GROWTH = 6.0
DEFAULT_ITERATION_GROWTH = 2.5


def allocate(time_left_ms: int, increment_ms: int = 500, moves_played: int = 0) -> tuple[float, float]:
    """Return (soft_ms, hard_ms) for this move."""
    usable = float(time_left_ms) - PROTOCOL_OVERHEAD_MS
    if usable <= 0.0:
        return 0.0, 0.0
    if time_left_ms < PANIC_THRESHOLD_MS:
        budget = min(PANIC_BUDGET_MS, usable * 0.5)
        return budget, budget

    moves_remaining = max(MIN_MOVES_REMAINING, BASE_MOVES_REMAINING - moves_played // 2)
    soft = usable / moves_remaining + increment_ms * INCREMENT_FRACTION
    hard = min(soft * HARD_MULTIPLIER, usable * HARD_CLOCK_FRACTION)

    ceiling = usable * ABSOLUTE_CLOCK_FRACTION
    soft = min(soft, ceiling)
    hard = min(max(hard, soft), ceiling)
    return soft, hard
