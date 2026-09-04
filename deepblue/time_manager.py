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
MIN_MOVES_REMAINING = 18
BASE_MOVES_REMAINING = 42

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
