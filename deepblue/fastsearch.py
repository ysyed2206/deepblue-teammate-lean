"""S1-search0: the first search built directly on our own core.

The point of this module is one measurement: what did replacing python-chess
actually buy in a *real* search? So it is deliberately the smallest correct
compiled search - negamax, alpha-beta, iterative deepening, quiescence, a
compiled evaluator with the same material and piece-square semantics as S0.
No transposition table, no pruning, no reductions. Those come after the
architecture has been measured, one at a time.

The hot loop is fused. It does NOT build a legal move list first:

    generate pseudo-legal
    for each candidate, best-ordered first:
        make once
        if the mover's king is attacked: unmake, skip
        recurse
        unmake once

`fastcore.generate_legal` still exists and is still exact, because perft and
the differential tests need it. It is simply not what a search should call:
prefiltering makes every searched move twice, and profiling that architecture
is what made the legality filter look like 96% of the cost.

Interruption is a shared one-byte flag set by a Python timer thread. The
jitted search is nogil, so the timer thread genuinely runs while the search is
running; the search polls the flag every few thousand nodes and unwinds. The
value of an interrupted pass is discarded entirely - only the last completed
iteration is ever returned.
"""

from __future__ import annotations

import threading
import time

import numpy as np
from numba import njit

from deepblue import constants as C
from deepblue.fastcore import (
    BK,
    MAX_MOVES,
    MAX_PLY,
    NO_PIECE,
    UNDO_STRIDE,
    WHITE,
    WK,
    generate_pseudo_legal,
    in_check,
    lsb,
    make_move,
    move_to_uci,
    popcount,
    unmake_move,
)

MATE_SCORE = C.MATE_SCORE
MATE_THRESHOLD = C.MATE_THRESHOLD
INFINITY = C.INFINITY
DRAW_SCORE = C.DRAW_SCORE
TOTAL_PHASE = C.TOTAL_PHASE
TEMPO_BONUS = C.TEMPO_BONUS

# The stop flag is read on EVERY node entry, not on a node-count interval.
#
# An earlier version gated the read behind `nodes & 2047 == 0`, reasoning that
# polling should be cheap. That was backwards: the mask is what makes a clock
# *syscall* affordable, but this flag is a single byte in an array the search
# already holds, so reading it costs almost nothing. Gating it meant only one
# node entry in 2048 ever noticed the flag, and every other node did its full
# work before returning - measured as a 2,090 ms unwind after the timer had
# already fired, on a 2,000 ms budget. That is a flag, and a flag is a loss.
#
# The flag is also re-checked inside the move loop, so a node with thirty
# candidates abandons the rest of them rather than finishing the list.

# Counter slots, so the jitted search can report without returning a tuple.
NODES, QNODES = 0, 1


def _build_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Signed, pre-mirrored material+PST tables indexed [piece, square].

    White entries are positive, black entries negative and vertically
    mirrored, so evaluation is one table read per piece with no branching on
    colour. The values are exactly S0's, so the two engines are being compared
    on architecture rather than on evaluation quality.
    """
    middlegame = np.zeros((12, 64), dtype=np.int32)
    endgame = np.zeros((12, 64), dtype=np.int32)
    phase = np.zeros(12, dtype=np.int32)
    order = (1, 2, 3, 4, 5, 6)  # chess.PAWN..KING
    for index, piece_type in enumerate(order):
        for square in range(64):
            middlegame[index, square] = C.WHITE_MG[piece_type][square]
            endgame[index, square] = C.WHITE_EG[piece_type][square]
            middlegame[index + 6, square] = -C.WHITE_MG[piece_type][square ^ 56]
            endgame[index + 6, square] = -C.WHITE_EG[piece_type][square ^ 56]
        phase[index] = C.PHASE_WEIGHT[piece_type]
        phase[index + 6] = C.PHASE_WEIGHT[piece_type]
    return middlegame, endgame, phase


MG_TABLE, EG_TABLE, PHASE_TABLE = _build_tables()

# MVV-LVA: victim value times sixteen minus attacker value, indexed by the
# piece codes fastcore uses.
PIECE_VALUE = np.array([1, 3, 3, 5, 9, 20, 1, 3, 3, 5, 9, 20], dtype=np.int32)


@njit(cache=True, nogil=True)
def evaluate(bb, st, mg_table, eg_table, phase_table):
    middlegame = 0
    endgame = 0
    phase = 0
    for piece in range(12):
        bits = bb[piece]
        while bits:
            square = lsb(bits)
            bits &= bits - np.uint64(1)
            middlegame += mg_table[piece, square]
            endgame += eg_table[piece, square]
            phase += phase_table[piece]
    if phase > TOTAL_PHASE:
        phase = TOTAL_PHASE
    blended = middlegame * phase + endgame * (TOTAL_PHASE - phase)
    # Truncate toward zero, so the evaluation is exactly colour-symmetric.
    if blended >= 0:
        score = blended // TOTAL_PHASE
    else:
        score = -((-blended) // TOTAL_PHASE)
    if st[0] != WHITE:
        score = -score
    return score + TEMPO_BONUS


# Square-colour masks, for the bishop case of the dead-position rule.
DARK_SQUARES = np.uint64(0xAA55AA55AA55AA55)
LIGHT_SQUARES = np.uint64(0x55AA55AA55AA55AA)


@njit(cache=True, nogil=True)
def has_insufficient_material(bb, occ, colour):
    """One side's half of the dead-position rule, matching the referee exactly.

    An earlier version tested "at most one minor each", which is far too broad:
    it declared K+B vs K+N, K+N vs K+N and opposite-coloured K+B vs K+B to be
    automatic draws. None of them are. The real rule is asymmetric and depends
    on what the OPPONENT has, so it is written out per side.
    """
    offset = 0 if colour == 0 else 6
    # Any pawn, rook or queen and mate is constructible.
    if bb[offset] or bb[offset + 3] or bb[offset + 4]:
        return False
    if bb[offset + 1]:
        # Knights: insufficient only with a single knight and nothing else, and
        # only if the opponent has nothing but king and queens.
        others = occ[1 - colour] & ~(bb[5] | bb[11]) & ~(bb[4] | bb[10])
        return popcount(occ[colour]) <= 2 and others == np.uint64(0)
    if bb[offset + 2]:
        # Bishops: insufficient only when every bishop ON THE BOARD sits on one
        # colour complex, and there are no pawns or knights anywhere.
        bishops = bb[2] | bb[8]
        same_complex = (bishops & DARK_SQUARES) == np.uint64(0) or (
            bishops & LIGHT_SQUARES
        ) == np.uint64(0)
        return same_complex and (bb[0] | bb[6]) == np.uint64(0) and (
            bb[1] | bb[7]
        ) == np.uint64(0)
    return True


@njit(cache=True, nogil=True)
def insufficient_material(bb, occ):
    """Dead position: neither side can possibly deliver mate."""
    return has_insufficient_material(bb, occ, 0) and has_insufficient_material(bb, occ, 1)


@njit(cache=True, nogil=True)
def any_legal_move(bb, occ, mail, st, buf, undo, ply):
    """Does the side to move have at least one legal move?

    Exits at the first survivor, so in an ordinary position this costs about
    one make, one king test and one unmake. Only a genuine stalemate makes it
    scan the whole candidate list.
    """
    side = st[0]
    count = generate_pseudo_legal(bb, occ, mail, st, buf)
    for index in range(count):
        move = buf[index]
        make_move(bb, occ, mail, st, move, undo, ply)
        legal = not in_check(bb, occ, st, side)
        unmake_move(bb, occ, mail, st, move, undo, ply)
        if legal:
            return True
    return False


@njit(cache=True, nogil=True)
def order_moves(bb, mail, moves, count, scores):
    """Score candidates for selection-sorted pick-best-next."""
    for index in range(count):
        move = moves[index]
        to_square = (move >> np.uint32(6)) & np.uint32(63)
        piece = (move >> np.uint32(12)) & np.uint32(15)
        captured = (move >> np.uint32(16)) & np.uint32(15)
        promotion = (move >> np.uint32(20)) & np.uint32(15)
        score = 0
        if captured != np.uint32(NO_PIECE):
            score = 100000 + PIECE_VALUE[captured] * 16 - PIECE_VALUE[piece]
        if promotion != np.uint32(NO_PIECE):
            score += 200000 + PIECE_VALUE[promotion] * 16
        scores[index] = score


@njit(cache=True, nogil=True)
def pick_best(moves, scores, count, start):
    """Selection sort one step: swap the best remaining candidate to `start`."""
    best = start
    for index in range(start + 1, count):
        if scores[index] > scores[best]:
            best = index
    if best != start:
        moves[start], moves[best] = moves[best], moves[start]
        scores[start], scores[best] = scores[best], scores[start]
    return moves[start]


@njit(cache=False, nogil=True)
def quiescence(bb, occ, mail, st, alpha, beta, ply, pseudo, scores, undo, stop, counters,
               mg_table, eg_table, phase_table, probe):
    counters[QNODES] += 1
    counters[NODES] += 1
    if stop[0]:
        return 0
    if ply >= MAX_PLY - 2:
        return evaluate(bb, st, mg_table, eg_table, phase_table)
    if insufficient_material(bb, occ):
        return DRAW_SCORE

    side = st[0]
    checked = in_check(bb, occ, st, side)
    count = generate_pseudo_legal(bb, occ, mail, st, pseudo[ply])
    order_moves(bb, mail, pseudo[ply], count, scores[ply])

    if checked:
        # In check every move is a candidate: standing pat here would evaluate
        # a position in which the king can be captured.
        best = -INFINITY
        legal = 0
        for index in range(count):
            move = pick_best(pseudo[ply], scores[ply], count, index)
            make_move(bb, occ, mail, st, move, undo, ply)
            if in_check(bb, occ, st, side):
                unmake_move(bb, occ, mail, st, move, undo, ply)
                continue
            legal += 1
            score = -quiescence(bb, occ, mail, st, -beta, -alpha, ply + 1, pseudo, scores,
                                undo, stop, counters, mg_table, eg_table, phase_table, probe)
            unmake_move(bb, occ, mail, st, move, undo, ply)
            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta or stop[0]:
                break
        if legal == 0:
            return -MATE_SCORE + ply
        return best

    # Not in check. Standing pat is only valid if a legal move actually
    # exists. Counting *pseudo* tactical moves is not enough: a position can
    # have pseudo-captures that are all illegal because they expose the king,
    # and no legal quiet move either, which is stalemate. So legality is
    # established directly, with an early exit that costs about one
    # make/test/unmake in an ordinary position.
    if not any_legal_move(bb, occ, mail, st, probe, undo, ply):
        return DRAW_SCORE
    # Only now, with mate and stalemate ruled out, does the fifty-move rule
    # apply - checkmate outranks it, exactly as it does for the referee.
    if st[3] >= 100:
        return DRAW_SCORE

    stand_pat = evaluate(bb, st, mg_table, eg_table, phase_table)
    if stand_pat >= beta:
        return stand_pat
    if stand_pat > alpha:
        alpha = stand_pat
    best = stand_pat

    for index in range(count):
        move = pick_best(pseudo[ply], scores[ply], count, index)
        captured = (move >> np.uint32(16)) & np.uint32(15)
        promotion = (move >> np.uint32(20)) & np.uint32(15)
        if captured == np.uint32(NO_PIECE) and promotion == np.uint32(NO_PIECE):
            continue
        make_move(bb, occ, mail, st, move, undo, ply)
        if in_check(bb, occ, st, side):
            unmake_move(bb, occ, mail, st, move, undo, ply)
            continue
        score = -quiescence(bb, occ, mail, st, -beta, -alpha, ply + 1, pseudo, scores,
                            undo, stop, counters, mg_table, eg_table, phase_table, probe)
        unmake_move(bb, occ, mail, st, move, undo, ply)
        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta or stop[0]:
            break
    return best


@njit(cache=False, nogil=True)
def negamax(bb, occ, mail, st, depth, alpha, beta, ply, pseudo, scores, undo, stop, counters,
            mg_table, eg_table, phase_table, probe):
    counters[NODES] += 1
    if stop[0]:
        return 0
    if insufficient_material(bb, occ):
        return DRAW_SCORE
    if depth <= 0:
        return quiescence(bb, occ, mail, st, alpha, beta, ply, pseudo, scores, undo, stop,
                          counters, mg_table, eg_table, phase_table, probe)

    side = st[0]
    checked = in_check(bb, occ, st, side)
    count = generate_pseudo_legal(bb, occ, mail, st, pseudo[ply])
    order_moves(bb, mail, pseudo[ply], count, scores[ply])

    best = -INFINITY
    legal = 0
    for index in range(count):
        move = pick_best(pseudo[ply], scores[ply], count, index)
        make_move(bb, occ, mail, st, move, undo, ply)
        if in_check(bb, occ, st, side):
            unmake_move(bb, occ, mail, st, move, undo, ply)
            continue
        legal += 1
        score = -negamax(bb, occ, mail, st, depth - 1, -beta, -alpha, ply + 1, pseudo, scores,
                         undo, stop, counters, mg_table, eg_table, phase_table, probe)
        unmake_move(bb, occ, mail, st, move, undo, ply)
        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta or stop[0]:
            break

    if legal == 0:
        # Terminal by construction: no pseudo-legal candidate survived the
        # king test. In check means mate, otherwise stalemate.
        return -MATE_SCORE + ply if checked else DRAW_SCORE
    # The fifty-move rule, applied only once mate has been ruled out, because
    # checkmate outranks it exactly as it does for the referee.
    if st[3] >= 100:
        return DRAW_SCORE
    return best


@njit(cache=False, nogil=True)
def search_root(bb, occ, mail, st, depth, pseudo, scores, undo, stop, counters,
                mg_table, eg_table, phase_table, root_moves, root_scores, probe):
    """One iterative-deepening pass. Returns the number of legal root moves;
    the caller reads root_moves[0] for the best move."""
    side = st[0]
    count = generate_pseudo_legal(bb, occ, mail, st, pseudo[0])
    order_moves(bb, mail, pseudo[0], count, scores[0])

    alpha = -INFINITY
    legal = 0
    for index in range(count):
        move = pick_best(pseudo[0], scores[0], count, index)
        make_move(bb, occ, mail, st, move, undo, 0)
        if in_check(bb, occ, st, side):
            unmake_move(bb, occ, mail, st, move, undo, 0)
            continue
        score = -negamax(bb, occ, mail, st, depth - 1, -INFINITY, -alpha, 1, pseudo, scores,
                         undo, stop, counters, mg_table, eg_table, phase_table, probe)
        unmake_move(bb, occ, mail, st, move, undo, 0)
        root_moves[legal] = move
        root_scores[legal] = score
        legal += 1
        if score > alpha:
            alpha = score
        if stop[0]:
            break
    # Bubble the best root move to the front.
    if legal > 0:
        best = 0
        for index in range(1, legal):
            if root_scores[index] > root_scores[best]:
                best = index
        if best != 0:
            root_moves[0], root_moves[best] = root_moves[best], root_moves[0]
            root_scores[0], root_scores[best] = root_scores[best], root_scores[0]
    return legal


class FastEngine:
    """Python driver: owns the buffers, the clock and the timer thread."""

    def __init__(self) -> None:
        self.pseudo = np.zeros((MAX_PLY, MAX_MOVES), dtype=np.uint32)
        self.scores = np.zeros((MAX_PLY, MAX_MOVES), dtype=np.int32)
        self.undo = np.zeros(MAX_PLY * UNDO_STRIDE, dtype=np.int64)
        self.stop = np.zeros(1, dtype=np.uint8)
        self.counters = np.zeros(4, dtype=np.int64)
        self.root_moves = np.zeros(MAX_MOVES, dtype=np.uint32)
        self.root_scores = np.zeros(MAX_MOVES, dtype=np.int32)
        # Scratch row for the legality probe, separate from the search stacks
        # so a probe at ply N cannot clobber ply N's own candidate list.
        self.probe = np.zeros(MAX_MOVES, dtype=np.uint32)

    def search(self, position, soft_ms: float, hard_ms: float, max_depth: int = 64):
        bb, occ, mail, st = position
        started = time.monotonic()

        # The referee calls outcome(claim_draw=True), which claims the
        # fifty-move draw before asking us to move. The engine must agree: a
        # position at the limit is drawn whatever material it can still win.
        # Without this the search reported +556 for a position the referee
        # scores as a draw. A move is still returned, because the referee may
        # not claim and we must never fail to move.
        fifty_move_draw = st[3] >= 100
        self.stop[0] = 0
        self.counters[:] = 0

        timer = threading.Timer(hard_ms / 1000.0, self._raise_stop)
        timer.daemon = True
        timer.start()
        try:
            best_move = 0
            best_score = 0
            completed = 0
            previous_ms = 0.0
            growth = 2.5
            for depth in range(1, max_depth + 1):
                iteration_started = time.monotonic()
                legal = search_root(
                    bb, occ, mail, st, depth, self.pseudo, self.scores, self.undo,
                    self.stop, self.counters, MG_TABLE, EG_TABLE, PHASE_TABLE,
                    self.root_moves, self.root_scores, self.probe,
                )
                if legal == 0:
                    break
                if self.stop[0]:
                    break  # the pass was interrupted; its result is discarded
                best_move = int(self.root_moves[0])
                best_score = int(self.root_scores[0])
                completed = depth

                now = time.monotonic()
                elapsed = (now - started) * 1000.0
                iteration_ms = (now - iteration_started) * 1000.0
                if elapsed >= soft_ms or abs(best_score) > MATE_THRESHOLD:
                    break
                if previous_ms > 1.0:
                    growth = min(6.0, max(2.0, iteration_ms / previous_ms))
                previous_ms = iteration_ms
                if elapsed + iteration_ms * growth > soft_ms:
                    break
            elapsed = (time.monotonic() - started) * 1000.0
            # Checkmate outranks the fifty-move rule, so a mate score is never
            # overwritten by it. The referee agrees: outcome() reports
            # CHECKMATE before it considers a claimable fifty-move draw.
            if fifty_move_draw and abs(best_score) <= MATE_THRESHOLD:
                best_score = DRAW_SCORE
            return (
                move_to_uci(best_move) if best_move else None,
                best_score,
                completed,
                int(self.counters[NODES]),
                elapsed,
            )
        finally:
            timer.cancel()

    def _raise_stop(self) -> None:
        self.stop[0] = 1


def warm_up() -> None:
    """Compile every signature the search uses, at import, inside the platform's
    initialisation budget rather than on the clock."""
    from deepblue.fastcore import from_fen

    engine = FastEngine()
    for fen in (
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ):
        engine.search(from_fen(fen), 30.0, 60.0, max_depth=3)
