"""S1-search29: fastsearch18 + LMP-7 (qsearch movecount cutoff).

Priority #5, the other immediately-testable LMP piece per BACKLOG.md,
built as its OWN separate candidate from fastsearch28 (LMP-1) -- they live
in different functions (quiescence vs. negamax) and this project's
established rule is one idea per candidate, so "listed together as this
round's priority" is not treated as license to combine them.

Per DONOR_NEXT_IDEAS.md: "LMP-7 qsearch movecount cutoff (``move_count>=3:
break``, fully independent, testable immediately without waiting on
anything else)". Quiescence's tactical (capture/promotion) move list is
already SEE/MVV-LVA ordered (best first, see ``order_qmoves``) before this
loop runs, so once the three best-looking tactical candidates have all been
tried and none produced a cutoff, the remaining, strictly worse-ordered
captures are judged not worth searching -- stop entirely rather than
working through the rest of the list.

Scope, deliberately the bare/literal donor form and nothing more:

- Applies ONLY in quiescence's non-check tactical-move loop (the branch that
  already has a sorted, generated tactical list). The in-check branch (full
  legal-move generation, since a king in check has no "captures only"
  luxury) is untouched -- cutting off evasions early is a correctness risk
  this candidate does not take.
- The counter is ``legal_tactical`` -- moves already confirmed legal (made,
  checked, and kept), matching how the existing loop already counts. The
  cutoff fires once the count would exceed 3, i.e. after the 3rd tactical
  move has been fully searched; a 4th candidate is neither made-and-searched
  nor even attempted.
- The move being cut off is unmade before the loop breaks, exactly like the
  existing beta-cutoff break just above it in the same loop, so this is not
  a new pattern for this file, just a second reason to reach it.
- ``legal_tactical`` is incremented before the break fires, so it correctly
  stays nonzero for the existing stalemate/draw detection immediately below
  the loop -- this candidate cannot accidentally manufacture a false
  "no legal tactical move" result.
- No change to the stand-pat cutoff path above this loop, no change to the
  in-check branch, no change to the TT store logic, no change to negamax.

Not combined with LMP-1 (fastsearch28), LMP-2 through LMP-6, or any LMR
piece (fastsearch27) -- each is its own separately-attributed candidate.

--- Inherited from fastsearch18 (qsearch TT), unchanged below ---

S1-search4: accepted PVS baseline plus conservative Reverse Futility Pruning.

Priority-list item #1 (donor roadmap: "qsearch TT"). fastsearch4's quiescence
never threaded the zobrist hash and never touched the table at all, so a
transposition that both the main search and quiescence visit paid the full
tactical-move-generation-and-search cost every single time.

Scope, deliberately narrow (one idea, mirroring this codebase's existing
"shortcuts don't write the TT" convention -- RFP in negamax never writes an
entry either):

- The hash is threaded into and through quiescence exactly the way negamax
  threads it into its own children.
- A probe happens once, near the top, shared by both the in-check and
  quiet branches, using the exact same EXACT/LOWER/UPPER cutoff logic as
  negamax's probe.
- A store happens only at the end of the normal move-loop path in each
  branch, with depth stored as 0. The replacement rule is
  ``tt_depth[index] <= depth OR tt_key[index] != value``.
- Move ordering in qsearch is deliberately NOT changed to prefer the TT
  move in this candidate.

No aspiration windows, null move, LMR, SEE, futility or IIR are present
(inherited unchanged from fastsearch4).
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
NODES, GEN, MAKE, CHECK, EVAL, TTPROBE, TTHIT, QNODES, ORDER, RFP_TRY, RFP_CUT = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
QTTPROBE, QTTHIT, QTT_EVICT_REAL, QTT_EVICT_QSEARCH = 11, 12, 13, 14
QMOVE_LIMIT_CUT = 15

TT_BITS = 20
TT_SIZE = 1 << TT_BITS
TT_MASK = np.uint64(TT_SIZE - 1)

TT_HALFMOVE_SAFE_LIMIT = 80

ORDER_TT = 1 << 24
ORDER_PROMOTION = 1 << 22
ORDER_CAPTURE = 1 << 21
ORDER_KILLER_1 = 1 << 20
ORDER_KILLER_2 = (1 << 20) - 1
HISTORY_CEILING = (1 << 19) - 1

MAX_GAME_HISTORY = 1024

RFP_MAX_DEPTH = 4
RFP_MARGIN_PER_DEPTH = 100  # centipawns

# LMP-7: qsearch movecount cutoff. Search at most this many tactical
# candidates per non-check qsearch node (already SEE/MVV-LVA sorted).
QSEARCH_MOVE_LIMIT = 3


@njit(cache=True, nogil=True)
def score_to_tt(score, ply):
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
def quiescence(bb, occ, mail, st, alpha, beta, ply, value, pseudo, scores, undo, stop, counters,
               probe, tt_key, tt_score, tt_depth, tt_bound, tt_move_arr,
               piece_keys, side_key, castle_keys, ep_file_keys):
    counters[NODES] += 1
    counters[QNODES] += 1
    if stop[0]:
        return 0
    if ply >= MAX_PLY - 2:
        return evaluate(bb, st, MG_TABLE, EG_TABLE, PHASE_TABLE)
    if insufficient_material(bb, occ):
        return DRAW_SCORE

    original_alpha = alpha
    tt_usable = st[3] < TT_HALFMOVE_SAFE_LIMIT
    tt_index = np.int64(value & TT_MASK)
    counters[QTTPROBE] += 1
    if tt_usable and tt_key[tt_index] == value:
        counters[QTTHIT] += 1
        stored = score_from_tt(tt_score[tt_index], ply)
        bound = tt_bound[tt_index]
        if bound == EXACT:
            return stored
        if bound == LOWER and stored > alpha:
            alpha = stored
        elif bound == UPPER and stored < beta:
            beta = stored
        if alpha >= beta:
            return stored

    side = st[0]
    checked = in_check(bb, occ, st, side)

    if checked:
        counters[GEN] += 1
        count = generate_pseudo_legal(bb, occ, mail, st, pseudo[ply])
        order_qmoves(pseudo[ply], count, scores[ply])
        best = -INFINITY
        best_move = np.uint32(0)
        legal = 0
        for index in range(count):
            move = pick_best(pseudo[ply], scores[ply], count, index)
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
            score = -quiescence(bb, occ, mail, st, -beta, -alpha, ply + 1, child, pseudo, scores,
                                undo, stop, counters, probe,
                                tt_key, tt_score, tt_depth, tt_bound, tt_move_arr,
                                piece_keys, side_key, castle_keys, ep_file_keys)
            unmake_move(bb, occ, mail, st, move, undo, ply)
            if score > best:
                best = score
                best_move = move
            if best > alpha:
                alpha = best
            if alpha >= beta or stop[0]:
                break
        if legal == 0:
            return -MATE_SCORE + ply
        if tt_usable and not stop[0]:
            if best <= original_alpha:
                bound = UPPER
            elif best >= beta:
                bound = LOWER
            else:
                bound = EXACT
            if tt_depth[tt_index] <= 0 or tt_key[tt_index] != value:
                if tt_key[tt_index] != value and tt_depth[tt_index] >= 1:
                    counters[QTT_EVICT_REAL] += 1
                elif tt_key[tt_index] == value and tt_depth[tt_index] == 0:
                    counters[QTT_EVICT_QSEARCH] += 1
                tt_key[tt_index] = value
                tt_score[tt_index] = score_to_tt(best, ply)
                tt_depth[tt_index] = 0
                tt_bound[tt_index] = bound
                tt_move_arr[tt_index] = best_move
        return best

    counters[EVAL] += 1
    stand_pat = evaluate(bb, st, MG_TABLE, EG_TABLE, PHASE_TABLE)
    counters[GEN] += 1
    count = generate_pseudo_tactical(bb, occ, mail, st, pseudo[ply])
    order_qmoves(pseudo[ply], count, scores[ply])

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
    best_move = np.uint32(0)
    legal_tactical = 0
    for index in range(count):
        move = pick_best(pseudo[ply], scores[ply], count, index)
        old_castle = st[1]
        old_ep_file = Z.canonical_ep_file(bb, occ, st)
        counters[MAKE] += 1
        make_move(bb, occ, mail, st, move, undo, ply)
        counters[CHECK] += 1
        if in_check(bb, occ, st, side):
            unmake_move(bb, occ, mail, st, move, undo, ply)
            continue
        legal_tactical += 1

        # LMP-7: already-sorted (SEE/MVV-LVA) tactical list -- once the best
        # QSEARCH_MOVE_LIMIT candidates have been fully searched with no
        # cutoff, stop trying the (strictly worse-ordered) rest entirely.
        if legal_tactical > QSEARCH_MOVE_LIMIT:
            counters[QMOVE_LIMIT_CUT] += 1
            unmake_move(bb, occ, mail, st, move, undo, ply)
            break

        new_ep_file = Z.canonical_ep_file(bb, occ, st)
        child = Z.apply_move(value, np.uint64(move), side, old_castle, old_ep_file,
                             st[1], new_ep_file, piece_keys, side_key, castle_keys, ep_file_keys)
        score = -quiescence(bb, occ, mail, st, -beta, -alpha, ply + 1, child, pseudo, scores,
                            undo, stop, counters, probe,
                            tt_key, tt_score, tt_depth, tt_bound, tt_move_arr,
                            piece_keys, side_key, castle_keys, ep_file_keys)
        unmake_move(bb, occ, mail, st, move, undo, ply)
        if score > best:
            best = score
            best_move = move
        if best > alpha:
            alpha = best
        if alpha >= beta or stop[0]:
            break
    if legal_tactical == 0:
        if not any_legal_move(bb, occ, mail, st, probe, undo, ply):
            return DRAW_SCORE
    if st[3] >= 100:
        return DRAW_SCORE
    if tt_usable and not stop[0]:
        if best <= original_alpha:
            bound = UPPER
        elif best >= beta:
            bound = LOWER
        else:
            bound = EXACT
        if tt_depth[tt_index] <= 0 or tt_key[tt_index] != value:
            if tt_key[tt_index] != value and tt_depth[tt_index] >= 1:
                counters[QTT_EVICT_REAL] += 1
            elif tt_key[tt_index] == value and tt_depth[tt_index] == 0:
                counters[QTT_EVICT_QSEARCH] += 1
            tt_key[tt_index] = value
            tt_score[tt_index] = score_to_tt(best, ply)
            tt_depth[tt_index] = 0
            tt_bound[tt_index] = bound
            tt_move_arr[tt_index] = best_move
    return best


@njit(cache=False, nogil=True)
def negamax(bb, occ, mail, st, depth, alpha, beta, ply, value, pseudo, scores, undo, stop,
            counters, probe, path_hashes, game_hashes, game_count,
            tt_key, tt_score, tt_depth, tt_bound, tt_move_arr,
            killers, history, piece_keys, side_key, castle_keys, ep_file_keys):
    counters[NODES] += 1
    if stop[0]:
        return 0

    path_hashes[ply] = value

    if ply > 0:
        prior_total, prior_path = repetition_info(
            value, path_hashes, ply, game_hashes, game_count, st[3]
        )
        if prior_total >= 2:
            return DRAW_SCORE
        if prior_path >= 1:
            return DRAW_SCORE
    if insufficient_material(bb, occ):
        return DRAW_SCORE
    if depth <= 0:
        return quiescence(bb, occ, mail, st, alpha, beta, ply, value, pseudo, scores, undo, stop,
                          counters, probe, tt_key, tt_score, tt_depth, tt_bound, tt_move_arr,
                          piece_keys, side_key, castle_keys, ep_file_keys)

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

    is_pv = beta - original_alpha > 1
    tt_move_is_quiet = False
    if tt_move != np.uint32(0):
        tt_cap = (tt_move >> np.uint32(16)) & np.uint32(15)
        tt_promo = (tt_move >> np.uint32(20)) & np.uint32(15)
        tt_move_is_quiet = (
            tt_cap == np.uint32(NO_PIECE) and tt_promo == np.uint32(NO_PIECE)
        )
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
        margin = RFP_MARGIN_PER_DEPTH * depth
        if static_eval - margin >= beta:
            counters[RFP_CUT] += 1
            return static_eval - margin

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
        if legal == 1:
            score = -negamax(bb, occ, mail, st, depth - 1, -beta, -alpha, ply + 1, child,
                             pseudo, scores, undo, stop, counters, probe, path_hashes,
                             game_hashes, game_count, tt_key, tt_score, tt_depth, tt_bound,
                             tt_move_arr, killers, history,
                             piece_keys, side_key, castle_keys, ep_file_keys)
        else:
            score = -negamax(bb, occ, mail, st, depth - 1, -alpha - 1, -alpha, ply + 1, child,
                             pseudo, scores, undo, stop, counters, probe, path_hashes,
                             game_hashes, game_count, tt_key, tt_score, tt_depth, tt_bound,
                             tt_move_arr, killers, history,
                             piece_keys, side_key, castle_keys, ep_file_keys)
            if score > alpha and score < beta and not stop[0]:
                score = -negamax(bb, occ, mail, st, depth - 1, -beta, -alpha, ply + 1, child,
                                 pseudo, scores, undo, stop, counters, probe, path_hashes,
                                 game_hashes, game_count, tt_key, tt_score, tt_depth, tt_bound,
                                 tt_move_arr, killers, history,
                                 piece_keys, side_key, castle_keys, ep_file_keys)
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
            score = -negamax(bb, occ, mail, st, depth - 1, np.int64(-INFINITY), -alpha, np.int64(1), child,
                             pseudo, scores, undo, stop, counters, probe, path_hashes,
                             game_hashes, game_count, tt_key, tt_score, tt_depth, tt_bound,
                             tt_move_arr, killers, history,
                             piece_keys, side_key, castle_keys, ep_file_keys)
        else:
            score = -negamax(bb, occ, mail, st, depth - 1, -alpha - 1, -alpha, np.int64(1), child,
                             pseudo, scores, undo, stop, counters, probe, path_hashes,
                             game_hashes, game_count, tt_key, tt_score, tt_depth, tt_bound,
                             tt_move_arr, killers, history,
                             piece_keys, side_key, castle_keys, ep_file_keys)
            if score > alpha and not stop[0]:
                score = -negamax(bb, occ, mail, st, depth - 1, np.int64(-INFINITY), -alpha, np.int64(1), child,
                                 pseudo, scores, undo, stop, counters, probe, path_hashes,
                                 game_hashes, game_count, tt_key, tt_score, tt_depth, tt_bound,
                                 tt_move_arr, killers, history,
                                 piece_keys, side_key, castle_keys, ep_file_keys)
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


class FastEngine29:
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
    engine = FastEngine29()
    for fen in (
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ):
        engine.search(from_fen(fen), 40.0, 80.0, max_depth=3)
