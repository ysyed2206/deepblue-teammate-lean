"""Deep Blue - the submission entrypoint.

The platform imports this file once per game and calls get_move for each of our
moves. Import time runs inside a 60 second budget before the clock starts, so
all table construction happens at import.

Two reliability commitments shape this file.

*   A known-legal fallback move is chosen before anything else can fail, and is
    returned on any exception. An illegal move or a crash loses the game
    outright, and qualification is a thirteen-game Swiss where one lost point
    outweighs a large amount of nominal strength.
*   Repetition bookkeeping records BOTH the position we are asked to move in
    and the position our own move creates. We are only ever shown the first,
    but the referee claims threefold automatically on either, so recording only
    our own turns loses won games to draws we never saw coming.

S1 engine, and why warm_up() is called eagerly below: FastEngine18's search/
quiescence/negamax are Numba-JIT-compiled on first use (~50-53s measured
cold, after compile-time fixes -- see EXPERIMENTS.md). That cost MUST land
inside this file's own 60s import budget, not the first move's clock -- so
warm_up() runs here, at import time, rather than being left to fire lazily
on the first get_move() call.
"""

from __future__ import annotations

import sys
import time
import traceback

import chess

from deepblue import fastsearch18
from deepblue.fastcore import from_fen
from deepblue.fastsearch18 import FastEngine18
from deepblue.time_manager import allocate

# The published time control is 120 s + 0.5 s/move, but the agent API is only
# told the clock, never the increment. Assuming a fixed 500 ms is a real
# hazard: if the true increment is smaller, the engine spends the difference
# every move, drains its clock and drops into panic mode playing unsearched
# moves. That was measured, at a 100 ms increment, as a 20-0 loss to an
# otherwise identical build. So the increment is OBSERVED from how the clock
# actually moves, and the published value is only the starting assumption.
DEFAULT_INCREMENT_MS = 500

_engine = FastEngine18()
fastsearch18.warm_up()  # forces the Numba JIT compile now, inside the import budget
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
    except Exception:  # noqa: BLE001 - deliberately total
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
