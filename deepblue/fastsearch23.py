"""S1-search23: fastsearch20's NMP with Coda's actual remedy (clamp, not skip).

Parent: fastsearch20 (which itself parents fastsearch5, NMP v1). This is
explicitly framed, per independent donor re-verification (Agent E, see
CLAUDE_REMOTE_RESULTS.md EXP-R12), as the intended LAST candidate on the
NMP axis before parking it -- NMP has already consumed two prior
candidate-rounds (fastsearch5, fastsearch20) with no positive signal.

fastsearch20 fixed fastsearch5's null_depth=0-drops-into-quiescence issue
by SKIPPING NMP entirely whenever ``depth - 1 - NMP_REDUCTION < 1`` (i.e.
effectively raising the firing threshold from depth>=4 to depth>=5).
Fresh donor re-verification (re-fetched Stockfish/Ethereal/Viridithas/
Reckless/Coda source directly, not reused from prior notes) found this was
NOT the real donor remedy: four of five donors examined drop into
quiescence-depth null probes constantly and by design (Stockfish has NO
min-depth gate at all and R=7+depth/3, so depth-R<1 for every depth below
roughly 12 -- its "probe" IS a qsearch across most of its own tree). The
ONE donor that avoids it, Coda, does so by CLAMPING the reduction so the
null probe floors at depth 1 -- NMP still fires -- rather than skipping
the attempt outright:

    if depth - r < 1: r = depth - 1   # (Coda's form; null depth floors at 1)

fastsearch20's skip-based fix was demonstrably why it barely fires at Deep
Blue's real search depths: measured directly on 90 real game positions at
an 80ms/move budget, fastsearch20's NMP fired 6 times across 90 whole
searches (EXP-R06). This candidate restores NMP's original depth>=4 firing
threshold (fastsearch5's NMP_MIN_DEPTH, unchanged) but floors null_depth
at 1 instead of letting it reach 0 -- so a depth-4 trigger now gets a
shallow depth-1 alpha-beta probe (Coda's actual mechanism), never a full
quiescence search (fastsearch5's bug) and never a skip (fastsearch20's
undertested fix).

Per Agent E's explicit protocol: this candidate is gated on FIXED-DEPTH
node counts against fastsearch4 (the only channel through which NMP can
possibly help -- it cannot improve move quality at fixed depth, only
prune), not an 80ms paired gate, which EXP-R06 showed cannot exercise this
feature meaningfully. The diagnostic-only NMP_SKIPPED_SHALLOW counter from
fastsearch20 is removed (it no longer applies to this gate shape, and per
Agent E's finding, a candidate carrying instrumentation the baseline lacks
should not play a paired gate).

Every other guard (non-PV, not in check, non-pawn-material, no consecutive
null, static_eval >= beta, EP-aware null hash, decisive-score clamp on the
return) is unchanged from fastsearch5/fastsearch20. Not combined with the
TT-consistency guard or verification-search deltas (#1, #2 in
NMP_DELTA.md) -- those remain separate, later candidates, and per Agent
E's recommendation should likely not be pursued at all unless this
candidate's fixed-depth evidence is positive.

No aspiration windows, LMR, SEE, futility or IIR are present (inherited
unchanged from fastsearch5).
"""

from __future__ import annotations

import threading
import time

import numpy as np
from numba import njit

from deepblue import zobrist as Z
from deepblue.constants import DRAW_SCORE, INFINITY, MATE_SCORE, MATE_THRESHOLD
from deepblue.fastcore import (
    MAX_MOVES,
    MAX_PLY,
    NO_PIECE,
    WHITE,
    WN, WB, WR, WQ,
    BN, BB_, BR, BQ,
    UNDO_STRIDE,
    from_fen,
    generate_pseudo_legal,
    generate_pseudo_tactical,
    in_check,
    make_move,
    move_to_uci,
    unmake_move,
)
from deepblue.fastsearch import (
    EG_TABLE,
    MG_TABLE,
    PHASE_TABLE,
    PIECE_VALUE,
    any_legal_move,
    evaluate,
    insufficient_material,
)

EXACT, LOWER, UPPER = 0, 1, 2
# Counter slots. Counting calls and multiplying by separately measured unit
# costs disturbs the search far less than timing inside it, and the shares are
# what the next decision needs.
NODES, GEN, MAKE, CHECK, EVAL, TTPROBE, TTHIT, QNODES, ORDER, RFP_TRY, RFP_CUT, NMP_TRY, NMP_CUT = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12

# 2^20 entries is about 18 MB across the five arrays: comfortably inside the
# 2 GB runtime limit and small enough to stay friendly to one core's cache.
# Sizes get benchmarked later; this is not a maximise-RAM decision.
TT_BITS = 20
TT_SIZE = 1 << TT_BITS
TT_MASK = np.uint64(TT_SIZE - 1)

# Near the fifty-move boundary a position's value depends on the halfmove
# clock, which the hash does not encode. The table is simply not used there.
TT_HALFMOVE_SAFE_LIMIT = 80

ORDER_TT = 1 << 24
ORDER_PROMOTION = 1 << 22
ORDER_CAPTURE = 1 << 21
ORDER_KILLER_1 = 1 << 20
ORDER_KILLER_2 = (1 << 20) - 1
HISTORY_CEILING = (1 << 19) - 1

MAX_GAME_HISTORY = 1024

# RFP v1: deliberately conservative while the evaluator is still primitive.
RFP_MAX_DEPTH = 4
RFP_MARGIN_PER_DEPTH = 100  # centipawns
NMP_MIN_DEPTH = 4
NMP_REDUCTION = 3



@njit(cache=True, nogil=True)
def score_to_tt(score, ply):
    """A mate score means 'mate in N from here'. Stored unadjusted, the same
    position reached at another ply reads back the wrong distance."""
    if score > MATE_THRESHOLD:
        return score + ply
    if score < -MATE_THRESHOLD:
        return score - ply
    return score


@njit(cache=True, nogil=True)
def score_from_tt(score, ply):
    if score > MATE_THRESHOLD:
        return score - ply
    if score < -MATE_THRESHOLD:
        return score + ply
    return score


@njit(cache=True, nogil=True)
def repetition_info(value, path_hashes, ply, game_hashes, game_count, halfmove):
    """Return ``(prior_total, prior_on_search_path)`` for ``value``.

    ``path_hashes[0]`` is the real root position and, during actual games, is
    also normally the last item in ``game_hashes``.  Counting both silently
    turns the first search cycle into a fake threefold, so the overlapping root
    entry is excluded from the game-history scan.

    The returned counts exclude the current node.  Therefore ``prior_total >=
    2`` means the current node is the third occurrence under the real rule;
    ``prior_path >= 1`` is a separate optional search-cycle heuristic.
    """
    total = 0
    path_seen = 0
    limit = halfmove if halfmove < ply else ply
    index = ply - 1
    scanned = 0
    while index >= 0 and scanned < limit:
        if path_hashes[index] == value:
            total += 1
            path_seen += 1
        index -= 1
        scanned += 1
    remaining = halfmove - scanned
    index = game_count - 1
    # The current real root is represented by path_hashes[0] as well.  Skip the
    # duplicate copy if the caller recorded it before starting this search.
    if index >= 0 and game_hashes[index] == path_hashes[0]:
        index -= 1
    while index >= 0 and remaining > 0:
        if game_hashes[index] == value:
            total += 1
        index -= 1
        remaining -= 1
    return total, path_seen


@njit(cache=True, nogil=True)
def order_moves(moves, count, scores, mail, tt_move, killers, history, ply):
    for index in range(count):
        move = moves[index]
        to_square = (move >> np.uint32(6)) & np.uint32(63)
        piece = (move >> np.uint32(12)) & np.uint32(15)
        captured = (move >> np.uint32(16)) & np.uint32(15)
        promotion = (move >> np.uint32(20)) & np.uint32(15)
        if move == tt_move:
            scores[index] = ORDER_TT
        elif promotion != np.uint32(NO_PIECE):
            scores[index] = ORDER_PROMOTION + PIECE_VALUE[promotion] * 16
        elif captured != np.uint32(NO_PIECE):
            scores[index] = ORDER_CAPTURE + PIECE_VALUE[captured] * 16 - PIECE_VALUE[piece]
        elif move == killers[ply, 0]:
            scores[index] = ORDER_KILLER_1
        elif move == killers[ply, 1]:
            scores[index] = ORDER_KILLER_2
        else:
            value = history[piece, to_square]
            scores[index] = value if value < HISTORY_CEILING else HISTORY_CEILING


@njit(cache=True, nogil=True)
def order_qmoves(moves, count, scores):
    """Cheap qsearch ordering with no temporary arrays or history lookups."""
    for index in range(count):
        move = moves[index]
        piece = (move >> np.uint32(12)) & np.uint32(15)
        captured = (move >> np.uint32(16)) & np.uint32(15)
        promotion = (move >> np.uint32(20)) & np.uint32(15)
        score = 0
        if captured != np.uint32(NO_PIECE):
            score += ORDER_CAPTURE + PIECE_VALUE[captured] * 16 - PIECE_VALUE[piece]
        if promotion != np.uint32(NO_PIECE):
            score += ORDER_PROMOTION + PIECE_VALUE[promotion] * 16
        scores[index] = score


@njit(cache=True, nogil=True)
def pick_best(moves, scores, count, start):
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
               probe):
    counters[NODES] += 1
    counters[QNODES] += 1
    if stop[0]:
        return 0
    if ply >= MAX_PLY - 2:
        return evaluate(bb, st, MG_TABLE, EG_TABLE, PHASE_TABLE)
    if insufficient_material(bb, occ):
        return DRAW_SCORE

    side = st[0]
    checked = in_check(bb, occ, st, side)

    if checked:
        counters[GEN] += 1
        count = generate_pseudo_legal(bb, occ, mail, st, pseudo[ply])
        order_qmoves(pseudo[ply], count, scores[ply])
        best = -INFINITY
        legal = 0
        for index in range(count):
            move = pick_best(pseudo[ply], scores[ply], count, index)
            counters[MAKE] += 1
            make_move(bb, occ, mail, st, move, undo, ply)
            counters[CHECK] += 1
            if in_check(bb, occ, st, side):
                unmake_move(bb, occ, mail, st, move, undo, ply)
                continue
            legal += 1
            score = -quiescence(bb, occ, mail, st, -beta, -alpha, ply + 1, pseudo, scores,
                                undo, stop, counters, probe)
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

    counters[EVAL] += 1
    stand_pat = evaluate(bb, st, MG_TABLE, EG_TABLE, PHASE_TABLE)
    counters[GEN] += 1
    count = generate_pseudo_tactical(bb, occ, mail, st, pseudo[ply])
    order_qmoves(pseudo[ply], count, scores[ply])

    # A stand-pat cutoff is valid only if this is not stalemate.  Prove the
    # cheapest case first: if any tactical candidate is legal, a legal move
    # exists.  Only positions whose tactical list has no legal survivor pay for
    # a full early-exit legal-move probe.
    if stand_pat >= beta:
        has_legal_tactical = False
        for index in range(count):
            move = pseudo[ply, index]
            make_move(bb, occ, mail, st, move, undo, ply)
            legal = not in_check(bb, occ, st, side)
            unmake_move(bb, occ, mail, st, move, undo, ply)
            if legal:
                has_legal_tactical = True
                break
        if not has_legal_tactical and not any_legal_move(bb, occ, mail, st, probe, undo, ply):
            return DRAW_SCORE
        if st[3] >= 100:
            return DRAW_SCORE
        return stand_pat

    if stand_pat > alpha:
        alpha = stand_pat
    best = stand_pat
    legal_tactical = 0
    for index in range(count):
        move = pick_best(pseudo[ply], scores[ply], count, index)
        counters[MAKE] += 1
        make_move(bb, occ, mail, st, move, undo, ply)
        counters[CHECK] += 1
        if in_check(bb, occ, st, side):
            unmake_move(bb, occ, mail, st, move, undo, ply)
            continue
        legal_tactical += 1
        score = -quiescence(bb, occ, mail, st, -beta, -alpha, ply + 1, pseudo, scores,
                            undo, stop, counters, probe)
        unmake_move(bb, occ, mail, st, move, undo, ply)
        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta or stop[0]:
            break
    if legal_tactical == 0:
        if not any_legal_move(bb, occ, mail, st, probe, undo, ply):
            return DRAW_SCORE
    if st[3] >= 100:
        return DRAW_SCORE
    return best


@njit(cache=False, nogil=True)
def negamax(bb, occ, mail, st, depth, alpha, beta, ply, value, pseudo, scores, undo, stop,
            counters, probe, path_hashes, game_hashes, game_count,
            tt_key, tt_score, tt_depth, tt_bound, tt_move_arr,
            killers, history, piece_keys, side_key, castle_keys, ep_file_keys,
            suppress_repetition, can_null):
    counters[NODES] += 1
    if stop[0]:
        return 0

    path_hashes[ply] = value

    if ply > 0 and not suppress_repetition:
        prior_total, prior_path = repetition_info(
            value, path_hashes, ply, game_hashes, game_count, st[3]
        )
        # Rule-level threefold: two *prior* occurrences plus this node.
        if prior_total >= 2:
            return DRAW_SCORE
        # Search-only cycle heuristic.  Do not treat one previous occurrence in
        # the real game history as a rule-level draw.
        if prior_path >= 1:
            return DRAW_SCORE
    if insufficient_material(bb, occ):
        return DRAW_SCORE
    if depth <= 0:
        return quiescence(bb, occ, mail, st, alpha, beta, ply, pseudo, scores, undo, stop,
                          counters, probe)

    original_alpha = alpha
    tt_usable = st[3] < TT_HALFMOVE_SAFE_LIMIT
    index = np.int64(value & TT_MASK)
    tt_move = np.uint32(0)
    counters[TTPROBE] += 1
    if tt_usable and tt_key[index] == value:
        counters[TTHIT] += 1
        tt_move = tt_move_arr[index]
        if tt_depth[index] >= depth and ply > 0:
            stored = score_from_tt(tt_score[index], ply)
            bound = tt_bound[index]
            if bound == EXACT:
                return stored
            if bound == LOWER and stored > alpha:
                alpha = stored
            elif bound == UPPER and stored < beta:
                beta = stored
            if alpha >= beta:
                return stored

    side = st[0]
    counters[CHECK] += 1
    checked = in_check(bb, occ, st, side)

    # Reverse Futility Pruning (static null move).  This is intentionally
    # restricted to non-PV, non-check, interior, shallow nodes.  A quiet TT
    # move is evidence that static eval alone may be hiding the only good
    # defence, so mirror the conservative guard used by several modern
    # engines and skip RFP in that case.
    is_pv = beta - original_alpha > 1
    tt_move_is_quiet = False
    if tt_move != np.uint32(0):
        tt_cap = (tt_move >> np.uint32(16)) & np.uint32(15)
        tt_promo = (tt_move >> np.uint32(20)) & np.uint32(15)
        tt_move_is_quiet = (
            tt_cap == np.uint32(NO_PIECE) and tt_promo == np.uint32(NO_PIECE)
        )
    static_eval = 0
    static_valid = False
    if (
        ply > 0
        and not is_pv
        and not checked
        and depth <= RFP_MAX_DEPTH
        and not tt_move_is_quiet
        and abs(beta) < MATE_THRESHOLD
        and st[3] < 99
    ):
        counters[RFP_TRY] += 1
        counters[EVAL] += 1
        static_eval = evaluate(bb, st, MG_TABLE, EG_TABLE, PHASE_TABLE)
        static_valid = True
        margin = RFP_MARGIN_PER_DEPTH * depth
        if static_eval - margin >= beta:
            counters[RFP_CUT] += 1
            return static_eval - margin

    # Conservative null-move pruning.  The fictitious null line is searched
    # only as a fail-high proof; never at PV nodes/check nodes or pawn-only
    # endings where zugzwang risk is highest.  No consecutive null moves.
    # fastsearch23 / EXP-R12: the reduction is CLAMPED (Coda's actual
    # remedy) rather than gating the whole attempt on the unclamped
    # null_depth, so NMP fires at fastsearch5's original depth>=4
    # threshold again -- see the null_depth computation below.
    if (
        ply > 0
        and not is_pv
        and not checked
        and can_null
        and depth >= NMP_MIN_DEPTH
        and abs(beta) < MATE_THRESHOLD
        and st[3] < 99
    ):
        if side == WHITE:
            non_pawn = bb[WN] | bb[WB] | bb[WR] | bb[WQ]
        else:
            non_pawn = bb[BN] | bb[BB_] | bb[BR] | bb[BQ]
        if non_pawn != np.uint64(0):
            if not static_valid:
                counters[EVAL] += 1
                static_eval = evaluate(bb, st, MG_TABLE, EG_TABLE, PHASE_TABLE)
                static_valid = True
            if static_eval >= beta:
                counters[NMP_TRY] += 1
                old_side = st[0]
                old_ep = st[2]
                old_halfmove = st[3]
                old_fullmove = st[4]
                old_ep_file = Z.canonical_ep_file(bb, occ, st)
                st[0] = 1 - old_side
                st[2] = -1
                st[3] = old_halfmove + 1
                if old_side != WHITE:
                    st[4] = old_fullmove + 1
                null_value = value ^ side_key
                if old_ep_file >= 0:
                    null_value ^= ep_file_keys[old_ep_file]
                null_depth = depth - 1 - NMP_REDUCTION
                if null_depth < 1:
                    null_depth = 1
                null_score = -negamax(
                    bb, occ, mail, st, null_depth, -beta, -beta + 1, ply + 1,
                    null_value, pseudo, scores, undo, stop, counters, probe,
                    path_hashes, game_hashes, game_count,
                    tt_key, tt_score, tt_depth, tt_bound, tt_move_arr,
                    killers, history, piece_keys, side_key, castle_keys, ep_file_keys,
                    True, False,
                )
                st[0] = old_side
                st[2] = old_ep
                st[3] = old_halfmove
                st[4] = old_fullmove
                if stop[0]:
                    return 0
                if null_score >= beta:
                    counters[NMP_CUT] += 1
                    return beta if null_score > MATE_THRESHOLD else null_score

    counters[GEN] += 1
    count = generate_pseudo_legal(bb, occ, mail, st, pseudo[ply])
    counters[ORDER] += 1
    order_moves(pseudo[ply], count, scores[ply], mail, tt_move, killers, history, ply)

    best = -INFINITY
    best_move = np.uint32(0)
    legal = 0
    for order_index in range(count):
        move = pick_best(pseudo[ply], scores[ply], count, order_index)
        old_castle = st[1]
        old_ep_file = Z.canonical_ep_file(bb, occ, st)
        counters[MAKE] += 1
        make_move(bb, occ, mail, st, move, undo, ply)
        counters[CHECK] += 1
        if in_check(bb, occ, st, side):
            unmake_move(bb, occ, mail, st, move, undo, ply)
            continue
        legal += 1
        new_ep_file = Z.canonical_ep_file(bb, occ, st)
        child = Z.apply_move(value, np.uint64(move), side, old_castle, old_ep_file,
                             st[1], new_ep_file, piece_keys, side_key, castle_keys, ep_file_keys)
        # Principal Variation Search.  Good move ordering makes the first legal
        # move the best move surprisingly often.  Search it normally; probe
        # later moves with the minimal window that can answer "does this beat
        # alpha?".  Only a probe that lands strictly inside the full window is
        # re-searched for an exact score.
        if legal == 1:
            score = -negamax(bb, occ, mail, st, depth - 1, -beta, -alpha, ply + 1, child,
                             pseudo, scores, undo, stop, counters, probe, path_hashes,
                             game_hashes, game_count, tt_key, tt_score, tt_depth, tt_bound,
                             tt_move_arr, killers, history,
                             piece_keys, side_key, castle_keys, ep_file_keys,
                             suppress_repetition, True)
        else:
            score = -negamax(bb, occ, mail, st, depth - 1, -alpha - 1, -alpha, ply + 1, child,
                             pseudo, scores, undo, stop, counters, probe, path_hashes,
                             game_hashes, game_count, tt_key, tt_score, tt_depth, tt_bound,
                             tt_move_arr, killers, history,
                             piece_keys, side_key, castle_keys, ep_file_keys,
                             suppress_repetition, True)
            if score > alpha and score < beta and not stop[0]:
                score = -negamax(bb, occ, mail, st, depth - 1, -beta, -alpha, ply + 1, child,
                                 pseudo, scores, undo, stop, counters, probe, path_hashes,
                                 game_hashes, game_count, tt_key, tt_score, tt_depth, tt_bound,
                                 tt_move_arr, killers, history,
                                 piece_keys, side_key, castle_keys, ep_file_keys,
                             suppress_repetition, True)
        unmake_move(bb, occ, mail, st, move, undo, ply)
        if score > best:
            best = score
            best_move = move
        if best > alpha:
            alpha = best
        if alpha >= beta:
            captured = (move >> np.uint32(16)) & np.uint32(15)
            promotion = (move >> np.uint32(20)) & np.uint32(15)
            if captured == np.uint32(NO_PIECE) and promotion == np.uint32(NO_PIECE):
                if killers[ply, 0] != move:
                    killers[ply, 1] = killers[ply, 0]
                    killers[ply, 0] = move
                piece = (move >> np.uint32(12)) & np.uint32(15)
                to_square = (move >> np.uint32(6)) & np.uint32(63)
                history[piece, to_square] += depth * depth
            break
        if stop[0]:
            break

    if legal == 0:
        return -MATE_SCORE + ply if checked else DRAW_SCORE
    if st[3] >= 100:
        return DRAW_SCORE

    if tt_usable and not stop[0]:
        if best <= original_alpha:
            bound = UPPER
        elif best >= beta:
            bound = LOWER
        else:
            bound = EXACT
        if tt_depth[index] <= depth or tt_key[index] != value:
            tt_key[index] = value
            tt_score[index] = score_to_tt(best, ply)
            tt_depth[index] = depth
            tt_bound[index] = bound
            tt_move_arr[index] = best_move
    return best


@njit(cache=False, nogil=True)
def search_root(bb, occ, mail, st, depth, value, pseudo, scores, undo, stop, counters,
                probe, path_hashes, game_hashes, game_count,
                tt_key, tt_score, tt_depth, tt_bound, tt_move_arr,
                killers, history, piece_keys, side_key, castle_keys, ep_file_keys,
                root_moves, root_scores, previous_best):
    side = st[0]
    path_hashes[0] = value
    count = generate_pseudo_legal(bb, occ, mail, st, pseudo[0])
    order_moves(pseudo[0], count, scores[0], mail, previous_best, killers, history, 0)

    alpha = -INFINITY
    legal = 0
    for order_index in range(count):
        move = pick_best(pseudo[0], scores[0], count, order_index)
        old_castle = st[1]
        old_ep_file = Z.canonical_ep_file(bb, occ, st)
        make_move(bb, occ, mail, st, move, undo, 0)
        if in_check(bb, occ, st, side):
            unmake_move(bb, occ, mail, st, move, undo, 0)
            continue
        new_ep_file = Z.canonical_ep_file(bb, occ, st)
        child = Z.apply_move(value, np.uint64(move), side, old_castle, old_ep_file,
                             st[1], new_ep_file, piece_keys, side_key, castle_keys, ep_file_keys)
        if legal == 0:
            score = -negamax(bb, occ, mail, st, depth - 1, -INFINITY, -alpha, 1, child,
                             pseudo, scores, undo, stop, counters, probe, path_hashes,
                             game_hashes, game_count, tt_key, tt_score, tt_depth, tt_bound,
                             tt_move_arr, killers, history,
                             piece_keys, side_key, castle_keys, ep_file_keys,
                             False, True)
        else:
            score = -negamax(bb, occ, mail, st, depth - 1, -alpha - 1, -alpha, 1, child,
                             pseudo, scores, undo, stop, counters, probe, path_hashes,
                             game_hashes, game_count, tt_key, tt_score, tt_depth, tt_bound,
                             tt_move_arr, killers, history,
                             piece_keys, side_key, castle_keys, ep_file_keys,
                             False, True)
            if score > alpha and not stop[0]:
                score = -negamax(bb, occ, mail, st, depth - 1, -INFINITY, -alpha, 1, child,
                                 pseudo, scores, undo, stop, counters, probe, path_hashes,
                                 game_hashes, game_count, tt_key, tt_score, tt_depth, tt_bound,
                                 tt_move_arr, killers, history,
                                 piece_keys, side_key, castle_keys, ep_file_keys,
                             False, True)
        unmake_move(bb, occ, mail, st, move, undo, 0)
        root_moves[legal] = move
        root_scores[legal] = score
        legal += 1
        if score > alpha:
            alpha = score
        if stop[0]:
            break
    if legal > 0:
        best = 0
        for index in range(1, legal):
            if root_scores[index] > root_scores[best]:
                best = index
        if best != 0:
            root_moves[0], root_moves[best] = root_moves[best], root_moves[0]
            root_scores[0], root_scores[best] = root_scores[best], root_scores[0]
    return legal


class FastEngine23:
    """Driver: owns the buffers, the table, the clock and the game history."""

    def __init__(self) -> None:
        self.pseudo = np.zeros((MAX_PLY, MAX_MOVES), dtype=np.uint32)
        self.scores = np.zeros((MAX_PLY, MAX_MOVES), dtype=np.int32)
        self.undo = np.zeros(MAX_PLY * UNDO_STRIDE, dtype=np.int64)
        self.probe = np.zeros(MAX_MOVES, dtype=np.uint32)
        self.stop = np.zeros(1, dtype=np.uint8)
        self.counters = np.zeros(16, dtype=np.int64)
        self.root_moves = np.zeros(MAX_MOVES, dtype=np.uint32)
        self.root_scores = np.zeros(MAX_MOVES, dtype=np.int32)
        self.path_hashes = np.zeros(MAX_PLY, dtype=np.uint64)

        self.tt_key = np.zeros(TT_SIZE, dtype=np.uint64)
        self.tt_score = np.zeros(TT_SIZE, dtype=np.int32)
        self.tt_depth = np.full(TT_SIZE, -1, dtype=np.int16)
        self.tt_bound = np.zeros(TT_SIZE, dtype=np.int8)
        self.tt_move = np.zeros(TT_SIZE, dtype=np.uint32)

        self.killers = np.zeros((MAX_PLY, 2), dtype=np.uint32)
        self.history = np.zeros((12, 64), dtype=np.int32)

        # Positions that have ACTUALLY occurred in the game, kept across moves.
        self.game_hashes = np.zeros(MAX_GAME_HISTORY, dtype=np.uint64)
        self.game_count = 0

    def record_game_position(self, position) -> None:
        bb, occ, _, st = position
        if self.game_count < MAX_GAME_HISTORY:
            self.game_hashes[self.game_count] = np.uint64(
                Z.full_hash(bb, occ, st, Z.PIECE_KEYS, Z.SIDE_KEY, Z.CASTLE_KEYS, Z.EP_FILE_KEYS)
            )
            self.game_count += 1

    def search(self, position, soft_ms: float, hard_ms: float, max_depth: int = 64):
        bb, occ, mail, st = position
        started = time.monotonic()
        self.stop[0] = 0
        self.counters[:] = 0
        self.killers[:] = 0
        fifty_move_draw = st[3] >= 100
        # numba hands a uint64 back to Python as a plain int, and re-passing
        # one above 2^63 into a jitted call infers int64 and raises
        # "int too big to convert". It has to be re-boxed as uint64.
        value = np.uint64(
            Z.full_hash(bb, occ, st, Z.PIECE_KEYS, Z.SIDE_KEY, Z.CASTLE_KEYS, Z.EP_FILE_KEYS)
        )

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
                    bb, occ, mail, st, depth, value, self.pseudo, self.scores, self.undo,
                    self.stop, self.counters, self.probe, self.path_hashes,
                    self.game_hashes, self.game_count,
                    self.tt_key, self.tt_score, self.tt_depth, self.tt_bound, self.tt_move,
                    self.killers, self.history,
                    Z.PIECE_KEYS, np.uint64(Z.SIDE_KEY), Z.CASTLE_KEYS, Z.EP_FILE_KEYS,
                    self.root_moves, self.root_scores, np.uint32(best_move),
                )
                if legal == 0 or self.stop[0]:
                    break
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
            if fifty_move_draw and abs(best_score) <= MATE_THRESHOLD:
                best_score = DRAW_SCORE
            return (
                move_to_uci(best_move) if best_move else None,
                best_score, completed, int(self.counters[NODES]), elapsed,
            )
        finally:
            timer.cancel()

    def _raise_stop(self) -> None:
        self.stop[0] = 1


def warm_up() -> None:
    engine = FastEngine23()
    for fen in (
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ):
        engine.search(from_fen(fen), 40.0, 80.0, max_depth=3)
