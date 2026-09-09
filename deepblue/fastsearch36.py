"""S1-search36: fastsearch4 + a transposition table probe/store inside qsearch.

S1-search4: accepted PVS baseline plus conservative Reverse Futility Pruning.

Priority-list item #1 (donor roadmap: "qsearch TT"). fastsearch4's quiescence
never threaded the zobrist hash and never touched the table at all, so a
transposition that both the main search and quiescence visit paid the full
tactical-move-generation-and-search cost every single time.

Scope, deliberately narrow (one idea, mirroring this codebase's existing
"shortcuts don't write the TT" convention -- RFP in negamax never writes an
entry either):

- The hash is threaded into and through quiescence exactly the way negamax
  threads it into its own children: via ``Z.apply_move`` with the same
  ``canonical_ep_file`` before/after bookkeeping, so the "pinned pseudo-EP
  does not affect the canonical hash" invariant is reused verbatim rather
  than re-derived.
- A probe happens once, near the top, shared by both the in-check and
  quiet branches, using the exact same EXACT/LOWER/UPPER cutoff logic as
  negamax's probe -- no PV/non-PV restriction, matching fastsearch4's own
  baseline (that restriction is fastsearch15/P1b's separate, unpromoted
  experiment; this candidate does not stack on it).
- A store happens only at the end of the normal move-loop path in each
  branch, with depth stored as 0. CORRECTED (independent red-team audit,
  see CLAUDE_REMOTE_RESULTS.md EXP-R04 Finding A1): the replacement rule
  is ``tt_depth[index] <= depth OR tt_key[index] != value`` -- the second
  disjunct means a qsearch write DOES unconditionally replace a
  *different-key* entry at any depth, including a real search entry. This
  is not a new failure mode (fastsearch4's own negamax-to-negamax writes
  use the identical disjunct), but an earlier draft of this docstring
  claimed a same-key-only guarantee that the code does not actually make;
  do not design against that guarantee. Measured eviction rate: ~0.02%-0.1%
  of nodes (52-248 evictions per 100k-450k nodes on the 5-position suite).
- The probe (not the store) is the actual source of every behavioural
  divergence from fastsearch4: it has NO depth condition, so a shallow
  quiescence node can consume a deep (depth>=1) negamax entry's bound. The
  audit ablated this precisely: a store-only variant (writes but never
  probes) was identical to fastsearch4 on every case that diverges under
  the full candidate; a probe restricted to depth==0 entries was also
  identical on those cases. Kept as coded (the probe is the entire point
  of this candidate -- a probe that only ever reads qsearch's own depth-0
  entries would forgo the actual transposition-hit benefit), but recorded
  here so the mechanism is attributed correctly.
- The stand-pat beta cutoff, the checkmate/stalemate-in-qsearch returns, and
  the fifty-move-rule draw return are all unchanged and, like negamax's own
  early returns, never reach the TT write -- only a result that actually
  came from the move loop is cached.
- Move ordering in qsearch is deliberately NOT changed to prefer the TT
  move in this candidate, to keep the experiment to one isolated idea
  (staged move ordering is donor-roadmap item separate from this one).

No aspiration windows, null move, LMR, SEE, futility or IIR are present
(inherited unchanged from fastsearch4).

CORRECTION (independent red-team audit, see EXP-R04 Finding A2):
"bit-identical to fastsearch4 at fixed depth" does NOT generalise beyond
the small 5-position engineering suite this was first measured on. A
broader sweep (57 positions x depths 1-7) found 8/399 pairs diverge,
including one actual BEST-MOVE change (not just a score/node-count
difference): ``8/8/4k3/8/8/3NKN2/8/8 w - - 0 1`` at depth 5, fs4 plays
e3d4 (652), this candidate plays e3e4 (657). The fixed-depth-identity
argument is therefore NOT available as a promotion shortcut for this
candidate -- paired games are the real gate, as the project's test
philosophy already requires for every candidate regardless.
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
    evaluate as base_evaluate,
    insufficient_material,
)

from deepblue.constants import TOTAL_PHASE as _TOTAL_PHASE
from deepblue.eval_terms import game_phase, king_safety_white_relative

_WHITE = 0


@njit(cache=True, nogil=True)
def evaluate(bb, st, mg_table, eg_table, phase_table):
    """Base PST evaluation plus the king-safety term.

    Two separately validated changes stacked: null-move pruning (fastsearch34,
    59.4% over 96 games, every round above 50%) and the king-safety/pawn-shield
    term (fastsearch33, 56.8%). Stacking is NOT assumed to be additive -- an
    earlier bundle of two individually-positive eval terms measured 44.3% --
    so this combination is its own candidate with its own evidence.
    """
    score = base_evaluate(bb, st, mg_table, eg_table, phase_table)
    phase = game_phase(bb, phase_table, _TOTAL_PHASE)
    extra = king_safety_white_relative(bb, phase, _TOTAL_PHASE)
    if st[0] != _WHITE:
        extra = -extra
    return score + extra


EXACT, LOWER, UPPER = 0, 1, 2
# Counter slots. Counting calls and multiplying by separately measured unit
# costs disturbs the search far less than timing inside it, and the shares are
# what the next decision needs.
NODES, GEN, MAKE, CHECK, EVAL, TTPROBE, TTHIT, QNODES, ORDER, RFP_TRY, RFP_CUT = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
QTTPROBE, QTTHIT, QTT_EVICT_REAL, QTT_EVICT_QSEARCH = 11, 12, 13, 14
NULL_TRY, NULL_CUT = 15, 16

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

# Null move pruning: minimum depth, base reduction, and the depth at which
# the reduction grows by one.
NULL_MIN_DEPTH = 3
NULL_BASE_R = 2
NULL_DEEP_DEPTH = 6



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
def qsearch_tt_store(tt_usable, stopped, best, original_alpha, beta, tt_index,
                      tt_key, tt_score, tt_depth, tt_bound, tt_move_arr,
                      value, ply, best_move, counters):
    """Shared TT store for quiescence's two branches (in-check and quiet) --
    byte-identical logic previously duplicated in both; extracted purely to
    shrink the compiled size of ``quiescence`` itself (Numba compile time
    scales with function body size for these large recursive functions; this
    refactor is a pure extraction with no behavioural change, verified via
    node-identical fixed-depth comparison against the pre-refactor code:
    8/8 exact move/score/node-count matches across 4 positions x 2 depths)."""
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
def rfp_margin(bb, st, ply, is_pv, checked, depth, tt_move, original_alpha, beta,
                counters, mg_table, eg_table, phase_table):
    """Reverse Futility Pruning check, extracted verbatim from negamax to
    shrink negamax's own compiled body (pure extraction, no behavioural
    change -- verified via node-identical fixed-depth comparison against the
    pre-extraction code: 8/8 exact move/score/node-count matches). Returns
    (should_cutoff, cutoff_value); the caller returns cutoff_value
    immediately when should_cutoff is True, exactly where the inline check
    used to ``return`` directly."""
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
        static_eval = evaluate(bb, st, mg_table, eg_table, phase_table)
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
        qsearch_tt_store(tt_usable, stop[0], best, original_alpha, beta, tt_index,
                          tt_key, tt_score, tt_depth, tt_bound, tt_move_arr,
                          value, ply, best_move, counters)
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
    qsearch_tt_store(tt_usable, stop[0], best, original_alpha, beta, tt_index,
                      tt_key, tt_score, tt_depth, tt_bound, tt_move_arr,
                      value, ply, best_move, counters)
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

    # Reverse Futility Pruning (static null move).  This is intentionally
    # restricted to non-PV, non-check, interior, shallow nodes.  A quiet TT
    # move is evidence that static eval alone may be hiding the only good
    # defence, so mirror the conservative guard used by several modern
    # engines and skip RFP in that case.
    is_pv = beta - original_alpha > 1
    rfp_cutoff, rfp_value = rfp_margin(bb, st, ply, is_pv, checked, depth, tt_move,
                                        original_alpha, beta, counters,
                                        MG_TABLE, EG_TABLE, PHASE_TABLE)
    if rfp_cutoff:
        return rfp_value

    # Null Move Pruning.  Hand the opponent a completely free move; if the
    # position still fails high at reduced depth even after that gift, the
    # real move list is almost certain to fail high too, so cut the node.
    #
    # Every guard below is load-bearing:
    #   not is_pv        - never gamble inside the principal variation.
    #   not checked      - "pass" is not a legal option when in check, and the
    #                      reduced search would be scored from an illegal premise.
    #   depth >= 3       - below that the reduction saves nothing worth the risk.
    #   non-pawn material for the side to move - ZUGZWANG. In a king-and-pawn
    #                      endgame passing is frequently the BEST move available,
    #                      so the null move's whole premise (passing is worse
    #                      than anything real) inverts precisely there, and
    #                      pruning on it throws away won pawn endgames.
    #   abs(beta) < MATE_THRESHOLD - never prune inside a mate score.
    #
    # A fail-high proved only by a null move is also never allowed to escape as
    # a mate score: passing cannot prove a forced mate.
    piece_offset = 0 if side == 0 else 6
    side_has_pieces = (
        bb[piece_offset + 1] | bb[piece_offset + 2]
        | bb[piece_offset + 3] | bb[piece_offset + 4]
    ) != np.uint64(0)
    if (
        ply > 0
        and not is_pv
        and not checked
        and depth >= NULL_MIN_DEPTH
        and abs(beta) < MATE_THRESHOLD
        and side_has_pieces
    ):
        counters[EVAL] += 1
        null_static = evaluate(bb, st, MG_TABLE, EG_TABLE, PHASE_TABLE)
        if null_static >= beta:
            counters[NULL_TRY] += 1
            reduction = NULL_BASE_R
            if depth >= NULL_DEEP_DEPTH:
                reduction += 1
            null_depth = depth - 1 - reduction
            if null_depth < 0:
                null_depth = 0
            old_ep = st[2]
            old_ep_file = Z.canonical_ep_file(bb, occ, st)
            null_value = value ^ side_key
            if old_ep_file >= 0:
                null_value ^= ep_file_keys[old_ep_file]
            st[0] = 1 - side
            st[2] = np.int64(-1)
            null_score = -negamax(bb, occ, mail, st, null_depth,
                                  -beta, -beta + np.int64(1), ply + 1, null_value,
                                  pseudo, scores, undo, stop, counters, probe, path_hashes,
                                  game_hashes, game_count, tt_key, tt_score, tt_depth,
                                  tt_bound, tt_move_arr, killers, history,
                                  piece_keys, side_key, castle_keys, ep_file_keys)
            st[0] = side
            st[2] = old_ep
            if not stop[0] and null_score >= beta:
                if null_score > MATE_THRESHOLD:
                    null_score = beta
                counters[NULL_CUT] += 1
                return null_score

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
            score = -negamax(bb, occ, mail, st, depth - 1, np.int64(-INFINITY), -alpha,
                             np.int64(1), child,
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
                score = -negamax(bb, occ, mail, st, depth - 1, np.int64(-INFINITY), -alpha,
                             np.int64(1), child,
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


class FastEngine36:
    """Driver: owns the buffers, the table, the clock and the game history."""

    def __init__(self) -> None:
        self.pseudo = np.zeros((MAX_PLY, MAX_MOVES), dtype=np.uint32)
        self.scores = np.zeros((MAX_PLY, MAX_MOVES), dtype=np.int32)
        self.undo = np.zeros(MAX_PLY * UNDO_STRIDE, dtype=np.int64)
        self.probe = np.zeros(MAX_MOVES, dtype=np.uint32)
        self.stop = np.zeros(1, dtype=np.uint8)
        self.counters = np.zeros(20, dtype=np.int64)  # 17 slots used; NULL_TRY/NULL_CUT added
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
    engine = FastEngine36()
    for fen in (
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ):
        engine.search(from_fen(fen), 40.0, 80.0, max_depth=3)
