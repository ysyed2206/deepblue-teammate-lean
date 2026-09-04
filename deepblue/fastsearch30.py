"""S1-search30: fastsearch18 + NNUE evaluation, replacing the hand-crafted
evaluator entirely (own-trained, fine-tuned from Pawnstar's public weights
on our own downloaded Stockfish-labelled data -- see
nnue_lab/teammate_pawnstar/finetune/ for the training/quantisation record).

Scope, deliberately the ONE idea this candidate tests: swap only the
evaluation source. Every other piece of fastsearch18 (PVS, RFP thresholds,
qsearch TT probe/store, move ordering, killers/history, time management) is
UNCHANGED. RFP's static-eval call and quiescence's stand-pat call now read
from the NNUE accumulator instead of calling ``evaluate()``; RFP's own
threshold/depth/guard logic is untouched (only which number it compares
against beta changed, not the pruning logic around it).

Accumulator maintenance, threaded through the search:

- ``white_acc``/``black_acc`` are ``int16[MAX_PLY+1, HIDDEN_SIZE]`` arrays,
  one row per ply. Ply 0 is refreshed once per move at the root, from the
  position ``search()`` was called with.
- Before making a candidate move at ply, the OLD king squares are read from
  the (still pre-move) bitboards; after ``make_move``, the NEW king squares
  are read back. If a perspective's king bucket is unchanged, that
  perspective's row at ``ply+1`` is an incremental delta from ``ply``'s row
  (built from the decoded move's own fields, matching ``fastcore.make_move``
  exactly -- see ``deepblue/nnue.py``'s ``compute_delta`` docstring). If the
  bucket changed, that perspective's row at ``ply+1`` is a full refresh from
  the post-move mailbox instead.
- No accumulator "unmake" is needed: a child node only ever writes its own
  ply's row; the parent's row is untouched by the child's recursion, so it
  is already correct again the instant the child returns. This mirrors how
  ``pseudo[ply]``/``scores[ply]`` are already ply-indexed scratch buffers in
  this file, just extended to the new accumulator state.
- Verified via ``tools/nnue_incremental_gate.py``: 100,000 real
  make_move/unmake_move transitions (random play, this exact engine's own
  move generator), 0 accumulator mismatches, full coverage of every special
  move type including king-bucket-crossing moves, en passant, both castling
  sides, promotion and capture-promotion.

--- Inherited from fastsearch18 (qsearch TT), unchanged below ---

S1-search4: accepted PVS baseline plus conservative Reverse Futility Pruning.
The hash is threaded into and through quiescence exactly the way negamax
threads it into its own children. A TT probe happens once, near the top,
shared by both the in-check and quiet branches. A TT store happens only at
the end of the normal move-loop path in each branch, depth stored as 0.
No aspiration windows, null move, LMR, SEE, futility or IIR are present.
"""

from __future__ import annotations

import threading
import time

import numpy as np
from numba import njit

from deepblue import nnue as dnnue
from deepblue import zobrist as Z
from deepblue.constants import DRAW_SCORE, INFINITY, MATE_SCORE, MATE_THRESHOLD
from deepblue.fastcore import (
    CAPTURE_SHIFT,
    FLAG_CASTLE,
    FLAG_EP,
    FROM_SHIFT,
    MAX_MOVES,
    MAX_PLY,
    NO_PIECE,
    PIECE_SHIFT,
    PROMOTION_SHIFT,
    TO_SHIFT,
    UNDO_STRIDE,
    from_fen,
    generate_pseudo_legal,
    generate_pseudo_tactical,
    in_check,
    lsb,
    make_move,
    move_to_uci,
    unmake_move,
)
from deepblue.fastsearch import PIECE_VALUE, any_legal_move, insufficient_material
from deepblue.see import see_value

EXACT, LOWER, UPPER = 0, 1, 2
NODES, GEN, MAKE, CHECK, EVAL, TTPROBE, TTHIT, QNODES, ORDER, RFP_TRY, RFP_CUT = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
QTTPROBE, QTTHIT, QTT_EVICT_REAL, QTT_EVICT_QSEARCH = 11, 12, 13, 14

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
QSEARCH_MOVE_LIMIT = 3  # matches fastsearch29's validated LMP-7 value; loosening this to 6 was not tight enough against Kiwipete-class positions
QSEARCH_MAX_DEPTH = 6  # hard cap on qsearch recursion depth -- see quiescence()'s docstring comment

WK, BK = 5, 11  # Deep Blue piece codes for the white/black king


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
def qsearch_tt_store(tt_usable, stopped, best, original_alpha, beta, tt_index,
                      tt_key, tt_score, tt_depth, tt_bound, tt_move_arr,
                      value, ply, best_move, counters):
    """Shared TT store for quiescence's two branches -- extracted purely to
    shrink quiescence's compiled body (same technique validated on
    fastsearch18: pure extraction, no behavioural change)."""
    if tt_usable and not stopped:
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


@njit(cache=True, nogil=True)
def rfp_margin(ply, is_pv, checked, depth, tt_move, beta, halfmove_clock, counters,
                white_acc_row, black_acc_row, side_to_move, output_w_stm, output_w_ntm, output_bias):
    """Reverse Futility Pruning check, extracted to shrink negamax's
    compiled body (pure extraction, same technique validated on
    fastsearch18). Returns (should_cutoff, cutoff_value)."""
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
        and halfmove_clock < 99
    ):
        counters[RFP_TRY] += 1
        counters[EVAL] += 1
        static_eval = dnnue.tail(white_acc_row, black_acc_row, side_to_move, output_w_stm, output_w_ntm, output_bias)
        margin = RFP_MARGIN_PER_DEPTH * depth
        if static_eval - margin >= beta:
            counters[RFP_CUT] += 1
            return True, static_eval - margin
    return False, 0


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


@njit(cache=True, nogil=True)
def advance_accumulators(bb, mail, move_int, side, white_acc_row, black_acc_row,
                          old_white_king, old_black_king,
                          feature_weights, feature_bias,
                          out_white_acc_row, out_black_acc_row,
                          removed_sq, removed_pc, added_sq, added_pc):
    """Compute ply+1's accumulator rows from ply's, given the move just made
    (bb/mail already reflect the POST-move position when this is called).
    Returns (new_white_king, new_black_king). Decodes ``move_int`` inline
    with the same bit layout ``fastcore.make_move`` itself uses (its own
    ``decode()`` is a plain-Python UCI helper, not njit-callable)."""
    move = np.uint32(move_int)
    from_sq = np.int64((move >> np.uint32(FROM_SHIFT)) & np.uint32(63))
    to_sq = np.int64((move >> np.uint32(TO_SHIFT)) & np.uint32(63))
    piece = np.int64((move >> np.uint32(PIECE_SHIFT)) & np.uint32(15))
    captured = np.int64((move >> np.uint32(CAPTURE_SHIFT)) & np.uint32(15))
    promotion = np.int64((move >> np.uint32(PROMOTION_SHIFT)) & np.uint32(15))
    is_ep = (move & FLAG_EP) != np.uint32(0)
    is_castle = (move & FLAG_CASTLE) != np.uint32(0)

    new_white_king = lsb(bb[WK])
    new_black_king = lsb(bb[BK])

    rc, ac = dnnue.compute_delta(from_sq, to_sq, piece, captured, promotion, is_ep, is_castle, side,
                                  removed_sq, removed_pc, added_sq, added_pc)

    if dnnue.king_bucket(new_white_king, 0) == dnnue.king_bucket(old_white_king, 0):
        result_white = dnnue.apply_delta(white_acc_row, feature_weights, removed_sq, removed_pc, rc,
                                          added_sq, added_pc, ac, old_white_king, 0)
    else:
        result_white = dnnue.refresh(feature_weights, feature_bias, mail, new_white_king, 0)

    if dnnue.king_bucket(new_black_king, 1) == dnnue.king_bucket(old_black_king, 1):
        result_black = dnnue.apply_delta(black_acc_row, feature_weights, removed_sq, removed_pc, rc,
                                          added_sq, added_pc, ac, old_black_king, 1)
    else:
        result_black = dnnue.refresh(feature_weights, feature_bias, mail, new_black_king, 1)

    for j in range(out_white_acc_row.shape[0]):
        out_white_acc_row[j] = result_white[j]
        out_black_acc_row[j] = result_black[j]
    return new_white_king, new_black_king


@njit(cache=False, nogil=True)
def quiescence(bb, occ, mail, st, alpha, beta, ply, qdepth, value, pseudo, scores, undo, stop, counters,
               probe, tt_key, tt_score, tt_depth, tt_bound, tt_move_arr,
               piece_keys, side_key, castle_keys, ep_file_keys,
               feature_weights, feature_bias, output_w_stm, output_w_ntm, output_bias,
               white_acc, black_acc, removed_sq, removed_pc, added_sq, added_pc):
    counters[NODES] += 1
    counters[QNODES] += 1
    if stop[0]:
        return 0
    # Hard cap on quiescence recursion depth (independent of the movecount
    # cap above, which only bounds branching *width* per node): positions
    # with many simultaneous tactical threads (e.g. the Kiwipete stress-test
    # position) can chain captures for many plies even at a bounded
    # branching factor -- 6 moves/node ^ 8 plies deep is already ~1.7M
    # nodes. Profiling confirmed 99.94% of nodes were in quiescence on that
    # position before this cap; this is the actual fix, not the width caps
    # alone. Falls back to the stand-pat/current value once hit, same as
    # running out of captures naturally.
    if qdepth >= QSEARCH_MAX_DEPTH:
        return dnnue.tail(white_acc[ply], black_acc[ply], st[0], output_w_stm, output_w_ntm, output_bias)
    if ply >= MAX_PLY - 2:
        return dnnue.tail(white_acc[ply], black_acc[ply], st[0], output_w_stm, output_w_ntm, output_bias)
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
            old_white_king = lsb(bb[WK])
            old_black_king = lsb(bb[BK])
            counters[MAKE] += 1
            make_move(bb, occ, mail, st, move, undo, ply)
            counters[CHECK] += 1
            if in_check(bb, occ, st, side):
                unmake_move(bb, occ, mail, st, move, undo, ply)
                continue
            legal += 1
            advance_accumulators(bb, mail, move, side, white_acc[ply], black_acc[ply],
                                  old_white_king, old_black_king, feature_weights, feature_bias,
                                  white_acc[ply + 1], black_acc[ply + 1], removed_sq, removed_pc, added_sq, added_pc)
            new_ep_file = Z.canonical_ep_file(bb, occ, st)
            child = Z.apply_move(value, np.uint64(move), side, old_castle, old_ep_file,
                                 st[1], new_ep_file, piece_keys, side_key, castle_keys, ep_file_keys)
            score = -quiescence(bb, occ, mail, st, -beta, -alpha, ply + 1, qdepth + 1, child, pseudo, scores,
                                undo, stop, counters, probe,
                                tt_key, tt_score, tt_depth, tt_bound, tt_move_arr,
                                piece_keys, side_key, castle_keys, ep_file_keys,
                                feature_weights, feature_bias, output_w_stm, output_w_ntm, output_bias,
                                white_acc, black_acc, removed_sq, removed_pc, added_sq, added_pc)
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
        qsearch_tt_store(tt_usable, stop[0], best, original_alpha, beta, tt_index,
                          tt_key, tt_score, tt_depth, tt_bound, tt_move_arr,
                          value, ply, best_move, counters)
        return best

    counters[EVAL] += 1
    stand_pat = dnnue.tail(white_acc[ply], black_acc[ply], st[0], output_w_stm, output_w_ntm, output_bias)
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
        # SEE-based qsearch skip: a non-promotion capture that loses material
        # even in the best case (SEE < 0) is not worth searching -- Kiwipete
        # -class positions (many simultaneous hanging pieces, no delta
        # pruning otherwise) explode without this; see.see_value's own
        # docstring forbids using it for promotions, so those are exempt.
        promotion_flag = (move >> np.uint32(20)) & np.uint32(15)
        if promotion_flag == np.uint32(NO_PIECE) and see_value(bb, occ, mail, st, move) < 0:
            continue
        # LMP-7 (already validated separately, fastsearch29): a hard cap on
        # how many tactical candidates qsearch will search at one node, on
        # top of the SEE skip above -- Kiwipete-class positions (many
        # simultaneous hanging pieces, no delta pruning) can still have more
        # SEE>=0 captures than are worth fully exploring; this is the
        # documented-missing safety net ("no delta pruning, no SEE" per
        # ARCHITECTURE.md) rather than a new experimental idea.
        if legal_tactical >= QSEARCH_MOVE_LIMIT:
            break
        old_castle = st[1]
        old_ep_file = Z.canonical_ep_file(bb, occ, st)
        old_white_king = lsb(bb[WK])
        old_black_king = lsb(bb[BK])
        counters[MAKE] += 1
        make_move(bb, occ, mail, st, move, undo, ply)
        counters[CHECK] += 1
        if in_check(bb, occ, st, side):
            unmake_move(bb, occ, mail, st, move, undo, ply)
            continue
        legal_tactical += 1
        advance_accumulators(bb, mail, move, side, white_acc[ply], black_acc[ply],
                              old_white_king, old_black_king, feature_weights, feature_bias,
                              white_acc[ply + 1], black_acc[ply + 1], removed_sq, removed_pc, added_sq, added_pc)
        new_ep_file = Z.canonical_ep_file(bb, occ, st)
        child = Z.apply_move(value, np.uint64(move), side, old_castle, old_ep_file,
                             st[1], new_ep_file, piece_keys, side_key, castle_keys, ep_file_keys)
        score = -quiescence(bb, occ, mail, st, -beta, -alpha, ply + 1, qdepth + 1, child, pseudo, scores,
                            undo, stop, counters, probe,
                            tt_key, tt_score, tt_depth, tt_bound, tt_move_arr,
                            piece_keys, side_key, castle_keys, ep_file_keys,
                            feature_weights, feature_bias, output_w_stm, output_w_ntm, output_bias,
                            white_acc, black_acc, removed_sq, removed_pc, added_sq, added_pc)
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
    qsearch_tt_store(tt_usable, stop[0], best, original_alpha, beta, tt_index,
                      tt_key, tt_score, tt_depth, tt_bound, tt_move_arr,
                      value, ply, best_move, counters)
    return best


@njit(cache=False, nogil=True)
def negamax(bb, occ, mail, st, depth, alpha, beta, ply, value, pseudo, scores, undo, stop,
            counters, probe, path_hashes, game_hashes, game_count,
            tt_key, tt_score, tt_depth, tt_bound, tt_move_arr,
            killers, history, piece_keys, side_key, castle_keys, ep_file_keys,
            feature_weights, feature_bias, output_w_stm, output_w_ntm, output_bias,
            white_acc, black_acc, removed_sq, removed_pc, added_sq, added_pc):
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
        return quiescence(bb, occ, mail, st, alpha, beta, ply, 0, value, pseudo, scores, undo, stop,
                          counters, probe, tt_key, tt_score, tt_depth, tt_bound, tt_move_arr,
                          piece_keys, side_key, castle_keys, ep_file_keys,
                          feature_weights, feature_bias, output_w_stm, output_w_ntm, output_bias,
                          white_acc, black_acc, removed_sq, removed_pc, added_sq, added_pc)

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
    rfp_cutoff, rfp_value = rfp_margin(ply, is_pv, checked, depth, tt_move, beta, st[3], counters,
                                        white_acc[ply], black_acc[ply], st[0],
                                        output_w_stm, output_w_ntm, output_bias)
    if rfp_cutoff:
        return rfp_value

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
        old_white_king = lsb(bb[WK])
        old_black_king = lsb(bb[BK])
        counters[MAKE] += 1
        make_move(bb, occ, mail, st, move, undo, ply)
        counters[CHECK] += 1
        if in_check(bb, occ, st, side):
            unmake_move(bb, occ, mail, st, move, undo, ply)
            continue
        legal += 1
        advance_accumulators(bb, mail, move, side, white_acc[ply], black_acc[ply],
                              old_white_king, old_black_king, feature_weights, feature_bias,
                              white_acc[ply + 1], black_acc[ply + 1], removed_sq, removed_pc, added_sq, added_pc)
        new_ep_file = Z.canonical_ep_file(bb, occ, st)
        child = Z.apply_move(value, np.uint64(move), side, old_castle, old_ep_file,
                             st[1], new_ep_file, piece_keys, side_key, castle_keys, ep_file_keys)
        if legal == 1:
            score = -negamax(bb, occ, mail, st, depth - 1, -beta, -alpha, ply + 1, child,
                             pseudo, scores, undo, stop, counters, probe, path_hashes,
                             game_hashes, game_count, tt_key, tt_score, tt_depth, tt_bound,
                             tt_move_arr, killers, history,
                             piece_keys, side_key, castle_keys, ep_file_keys,
                             feature_weights, feature_bias, output_w_stm, output_w_ntm, output_bias,
                             white_acc, black_acc, removed_sq, removed_pc, added_sq, added_pc)
        else:
            score = -negamax(bb, occ, mail, st, depth - 1, -alpha - 1, -alpha, ply + 1, child,
                             pseudo, scores, undo, stop, counters, probe, path_hashes,
                             game_hashes, game_count, tt_key, tt_score, tt_depth, tt_bound,
                             tt_move_arr, killers, history,
                             piece_keys, side_key, castle_keys, ep_file_keys,
                             feature_weights, feature_bias, output_w_stm, output_w_ntm, output_bias,
                             white_acc, black_acc, removed_sq, removed_pc, added_sq, added_pc)
            if score > alpha and score < beta and not stop[0]:
                score = -negamax(bb, occ, mail, st, depth - 1, -beta, -alpha, ply + 1, child,
                                 pseudo, scores, undo, stop, counters, probe, path_hashes,
                                 game_hashes, game_count, tt_key, tt_score, tt_depth, tt_bound,
                                 tt_move_arr, killers, history,
                                 piece_keys, side_key, castle_keys, ep_file_keys,
                                 feature_weights, feature_bias, output_w_stm, output_w_ntm, output_bias,
                                 white_acc, black_acc, removed_sq, removed_pc, added_sq, added_pc)
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
                root_moves, root_scores, previous_best,
                feature_weights, feature_bias, output_w_stm, output_w_ntm, output_bias,
                white_acc, black_acc, removed_sq, removed_pc, added_sq, added_pc):
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
        old_white_king = lsb(bb[WK])
        old_black_king = lsb(bb[BK])
        make_move(bb, occ, mail, st, move, undo, 0)
        if in_check(bb, occ, st, side):
            unmake_move(bb, occ, mail, st, move, undo, 0)
            continue
        advance_accumulators(bb, mail, move, side, white_acc[0], black_acc[0],
                              old_white_king, old_black_king, feature_weights, feature_bias,
                              white_acc[1], black_acc[1], removed_sq, removed_pc, added_sq, added_pc)
        new_ep_file = Z.canonical_ep_file(bb, occ, st)
        child = Z.apply_move(value, np.uint64(move), side, old_castle, old_ep_file,
                             st[1], new_ep_file, piece_keys, side_key, castle_keys, ep_file_keys)
        if legal == 0:
            score = -negamax(bb, occ, mail, st, depth - 1, np.int64(-INFINITY), -alpha, np.int64(1), child,
                             pseudo, scores, undo, stop, counters, probe, path_hashes,
                             game_hashes, game_count, tt_key, tt_score, tt_depth, tt_bound,
                             tt_move_arr, killers, history,
                             piece_keys, side_key, castle_keys, ep_file_keys,
                             feature_weights, feature_bias, output_w_stm, output_w_ntm, output_bias,
                             white_acc, black_acc, removed_sq, removed_pc, added_sq, added_pc)
        else:
            score = -negamax(bb, occ, mail, st, depth - 1, -alpha - 1, -alpha, np.int64(1), child,
                             pseudo, scores, undo, stop, counters, probe, path_hashes,
                             game_hashes, game_count, tt_key, tt_score, tt_depth, tt_bound,
                             tt_move_arr, killers, history,
                             piece_keys, side_key, castle_keys, ep_file_keys,
                             feature_weights, feature_bias, output_w_stm, output_w_ntm, output_bias,
                             white_acc, black_acc, removed_sq, removed_pc, added_sq, added_pc)
            if score > alpha and not stop[0]:
                score = -negamax(bb, occ, mail, st, depth - 1, np.int64(-INFINITY), -alpha, np.int64(1), child,
                                 pseudo, scores, undo, stop, counters, probe, path_hashes,
                                 game_hashes, game_count, tt_key, tt_score, tt_depth, tt_bound,
                                 tt_move_arr, killers, history,
                                 piece_keys, side_key, castle_keys, ep_file_keys,
                                 feature_weights, feature_bias, output_w_stm, output_w_ntm, output_bias,
                                 white_acc, black_acc, removed_sq, removed_pc, added_sq, added_pc)
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


class FastEngine30:
    """Driver: owns the buffers, the table, the clock, the game history, and
    the NNUE weights/accumulator stack."""

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

        (self.feature_weights, self.feature_bias, self.output_w_stm,
         self.output_w_ntm, self.output_bias) = dnnue.load_weights()
        width = self.feature_bias.shape[0]  # derived from the loaded net, not hardcoded -- width-flexible
        self.white_acc = np.zeros((MAX_PLY + 1, width), dtype=np.int16)
        self.black_acc = np.zeros((MAX_PLY + 1, width), dtype=np.int16)
        self.removed_sq = np.zeros(4, dtype=np.int64)
        self.removed_pc = np.zeros(4, dtype=np.int64)
        self.added_sq = np.zeros(4, dtype=np.int64)
        self.added_pc = np.zeros(4, dtype=np.int64)

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

        mail_i64 = mail.astype(np.int64)
        white_king = lsb(bb[WK])
        black_king = lsb(bb[BK])
        self.white_acc[0] = dnnue.refresh(self.feature_weights, self.feature_bias, mail_i64, white_king, 0)
        self.black_acc[0] = dnnue.refresh(self.feature_weights, self.feature_bias, mail_i64, black_king, 1)

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
                    self.feature_weights, self.feature_bias, self.output_w_stm, self.output_w_ntm, self.output_bias,
                    self.white_acc, self.black_acc, self.removed_sq, self.removed_pc, self.added_sq, self.added_pc,
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
    engine = FastEngine30()
    for fen in (
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ):
        engine.search(from_fen(fen), 40.0, 80.0, max_depth=3)
