with open("deepblue/fastsearch180.py", "r", encoding="utf-8") as f:
    text = f.read()

# 1. Update docstring
idx_future = text.find('from __future__ import annotations')
assert idx_future != -1, "future import not found"
doc_new = '''"""S1-search180n: fastsearch180 with incremental NNUE evaluation (own-trained, 14.8M positions, PSN1 H128).

Threaded through the search with incremental accumulator maintenance (apply_delta_into + tail_fast).
Weights are passed as parameters rather than global array closures, preserving fast cold-start compilation.
"""

'''
text = doc_new + text[idx_future:]

# 2. Update imports and remove classical evaluate block
old_block_start = text.find('from deepblue.fastsearch import (')
old_block_end = text.find('EXACT, LOWER, UPPER = 0, 1, 2')
assert old_block_start != -1 and old_block_end != -1, "old import / eval block not found"

new_import = '''from deepblue.fastsearch import (
    PIECE_VALUE,
    any_legal_move,
    insufficient_material,
)

from deepblue.see import SEE_VALUE, see_ge_zero
from deepblue import nnue as _dbnnue
from deepblue.nnue_fast import advance_accumulators, tail_fast
import os as _os
from pathlib import Path as _Path

WK, BK = 5, 11
_WHITE = 0
_NNUE_PATH = _os.environ.get("DEEPBLUE_NNUE_WEIGHTS") or str(
    _Path(__file__).resolve().parent / "deepblue_nnue_v3.bin")
_NNUE_FW, _NNUE_FB, _NNUE_WSTM, _NNUE_WNTM, _NNUE_OB = _dbnnue.load_weights(_NNUE_PATH)


'''

text = text[:old_block_start] + new_import + text[old_block_end:]

# 3. Update rfp_margin to remove unused classical tables
old_rfp_def = '''def rfp_margin(bb, st, ply, is_pv, checked, depth, tt_move, original_alpha, beta,
                counters, mg_table, eg_table, phase_table, static_eval, have_eval):'''
new_rfp_def = '''def rfp_margin(bb, st, ply, is_pv, checked, depth, tt_move, original_alpha, beta,
                counters, static_eval, have_eval):'''
assert old_rfp_def in text, "old_rfp_def not found"
text = text.replace(old_rfp_def, new_rfp_def, 1)

old_rfp_call = '''    rfp_cutoff, rfp_value = rfp_margin(bb, st, ply, is_pv, checked, depth, tt_move,
                                        original_alpha, beta, counters,
                                        MG_TABLE, EG_TABLE, PHASE_TABLE,
                                        node_eval, have_eval)'''
new_rfp_call = '''    rfp_cutoff, rfp_value = rfp_margin(bb, st, ply, is_pv, checked, depth, tt_move,
                                        original_alpha, beta, counters,
                                        node_eval, have_eval)'''
assert old_rfp_call in text, "old_rfp_call not found"
text = text.replace(old_rfp_call, new_rfp_call, 1)

# 4. Quiescence
# 4a. Signature
old_q_sig = '''def quiescence(bb, occ, mail, st, alpha, beta, ply, value, pseudo, scores, undo, stop, counters,
               probe, tt_key, tt_score, tt_depth, tt_bound, tt_move_arr,
               piece_keys, side_key, castle_keys, ep_file_keys):'''

new_q_sig = '''def quiescence(bb, occ, mail, st, alpha, beta, ply, value, pseudo, scores, undo, stop, counters,
               probe, tt_key, tt_score, tt_depth, tt_bound, tt_move_arr,
               piece_keys, side_key, castle_keys, ep_file_keys,
               feature_weights, feature_bias, output_w_stm, output_w_ntm, output_bias,
               white_acc, black_acc, removed_sq, removed_pc, added_sq, added_pc):'''

assert old_q_sig in text, "old_q_sig not found"
text = text.replace(old_q_sig, new_q_sig, 1)

# 4b. Max ply check
old_q_maxply = '''    if ply >= MAX_PLY - 2:
        return evaluate(bb, st, MG_TABLE, EG_TABLE, PHASE_TABLE)'''
new_q_maxply = '''    if ply >= MAX_PLY - 2:
        return tail_fast(white_acc[ply], black_acc[ply], st[0], output_w_stm, output_w_ntm, output_bias)'''
assert old_q_maxply in text, "old_q_maxply not found"
text = text.replace(old_q_maxply, new_q_maxply, 1)

# 4c. Checked move loop
old_q_checked_loop = '''            old_castle = st[1]
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
                                piece_keys, side_key, castle_keys, ep_file_keys)'''

new_q_checked_loop = '''            old_castle = st[1]
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
                                 old_white_king, old_black_king,
                                 feature_weights, feature_bias,
                                 white_acc[ply + 1], black_acc[ply + 1],
                                 removed_sq, removed_pc, added_sq, added_pc)
            new_ep_file = Z.canonical_ep_file(bb, occ, st)
            child = Z.apply_move(value, np.uint64(move), side, old_castle, old_ep_file,
                                 st[1], new_ep_file, piece_keys, side_key, castle_keys, ep_file_keys)
            score = -quiescence(bb, occ, mail, st, -beta, -alpha, ply + 1, child, pseudo, scores,
                                undo, stop, counters, probe,
                                tt_key, tt_score, tt_depth, tt_bound, tt_move_arr,
                                piece_keys, side_key, castle_keys, ep_file_keys,
                                feature_weights, feature_bias, output_w_stm, output_w_ntm, output_bias,
                                white_acc, black_acc, removed_sq, removed_pc, added_sq, added_pc)'''

assert old_q_checked_loop in text, "old_q_checked_loop not found"
text = text.replace(old_q_checked_loop, new_q_checked_loop, 1)

# 4d. Stand pat
old_q_standpat = '''    counters[EVAL] += 1
    stand_pat = evaluate(bb, st, MG_TABLE, EG_TABLE, PHASE_TABLE)'''
new_q_standpat = '''    counters[EVAL] += 1
    stand_pat = tail_fast(white_acc[ply], black_acc[ply], st[0], output_w_stm, output_w_ntm, output_bias)'''
assert old_q_standpat in text, "old_q_standpat not found"
text = text.replace(old_q_standpat, new_q_standpat, 1)

# 4e. Tactical move loop
old_q_tactical_loop = '''        old_castle = st[1]
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
                            piece_keys, side_key, castle_keys, ep_file_keys)'''

new_q_tactical_loop = '''        old_castle = st[1]
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
                             old_white_king, old_black_king,
                             feature_weights, feature_bias,
                             white_acc[ply + 1], black_acc[ply + 1],
                             removed_sq, removed_pc, added_sq, added_pc)
        new_ep_file = Z.canonical_ep_file(bb, occ, st)
        child = Z.apply_move(value, np.uint64(move), side, old_castle, old_ep_file,
                             st[1], new_ep_file, piece_keys, side_key, castle_keys, ep_file_keys)
        score = -quiescence(bb, occ, mail, st, -beta, -alpha, ply + 1, child, pseudo, scores,
                            undo, stop, counters, probe,
                            tt_key, tt_score, tt_depth, tt_bound, tt_move_arr,
                            piece_keys, side_key, castle_keys, ep_file_keys,
                            feature_weights, feature_bias, output_w_stm, output_w_ntm, output_bias,
                            white_acc, black_acc, removed_sq, removed_pc, added_sq, added_pc)'''

assert old_q_tactical_loop in text, "old_q_tactical_loop not found"
text = text.replace(old_q_tactical_loop, new_q_tactical_loop, 1)

# 5. Negamax
# 5a. Signature
old_n_sig = '''def negamax(bb, occ, mail, st, depth, alpha, beta, ply, value, pseudo, scores, undo, stop,
            counters, probe, path_hashes, game_hashes, game_count,
            tt_key, tt_score, tt_depth, tt_bound, tt_move_arr,
            killers, history, continuation_history, prev_moves, corr_hist, excluded,
            piece_keys, side_key, castle_keys, ep_file_keys,
            suppress_repetition, can_null):'''

new_n_sig = '''def negamax(bb, occ, mail, st, depth, alpha, beta, ply, value, pseudo, scores, undo, stop,
            counters, probe, path_hashes, game_hashes, game_count,
            tt_key, tt_score, tt_depth, tt_bound, tt_move_arr,
            killers, history, continuation_history, prev_moves, corr_hist, excluded,
            piece_keys, side_key, castle_keys, ep_file_keys,
            suppress_repetition, can_null,
            feature_weights, feature_bias, output_w_stm, output_w_ntm, output_bias,
            white_acc, black_acc, removed_sq, removed_pc, added_sq, added_pc):'''

assert old_n_sig in text, "old_n_sig not found"
text = text.replace(old_n_sig, new_n_sig, 1)

# 5b. Drop to quiescence at depth <= 0
old_n_drop_q = '''    if depth <= 0:
        return quiescence(bb, occ, mail, st, alpha, beta, ply, value, pseudo, scores, undo, stop,
                          counters, probe, tt_key, tt_score, tt_depth, tt_bound, tt_move_arr,
                          piece_keys, side_key, castle_keys, ep_file_keys)'''

new_n_drop_q = '''    if depth <= 0:
        return quiescence(
            bb, occ, mail, st, alpha, beta, ply, value, pseudo, scores, undo,
            stop, counters, probe, tt_key, tt_score, tt_depth, tt_bound, tt_move_arr,
            piece_keys, side_key, castle_keys, ep_file_keys,
            feature_weights, feature_bias, output_w_stm, output_w_ntm, output_bias,
            white_acc, black_acc, removed_sq, removed_pc, added_sq, added_pc)'''

assert old_n_drop_q in text, "old_n_drop_q not found"
text = text.replace(old_n_drop_q, new_n_drop_q, 1)

# 5c. Static eval in negamax
old_n_eval = '''    if not checked and not is_pv:
        counters[EVAL] += 1
        node_pawn_key = pawn_structure_key(bb, piece_keys)
        node_eval = corrected_eval(
            np.int64(evaluate(bb, st, MG_TABLE, EG_TABLE, PHASE_TABLE)),
            corr_hist, side, node_pawn_key)
        have_eval = True'''

new_n_eval = '''    if not checked and not is_pv:
        counters[EVAL] += 1
        node_pawn_key = pawn_structure_key(bb, piece_keys)
        static_eval = tail_fast(white_acc[ply], black_acc[ply], side,
                                output_w_stm, output_w_ntm, output_bias)
        node_eval = corrected_eval(
            static_eval,
            corr_hist, side, node_pawn_key)
        have_eval = True'''

assert old_n_eval in text, "old_n_eval not found"
text = text.replace(old_n_eval, new_n_eval, 1)

# 5d. Razoring call to quiescence
old_n_razor = '''        if razor_eval + RAZOR_MARGIN_BASE + RAZOR_MARGIN_PER_DEPTH * depth <= alpha:
            razor_score = quiescence(
                bb, occ, mail, st, alpha, alpha + 1, ply, value, pseudo, scores, undo,
                stop, counters, probe, tt_key, tt_score, tt_depth, tt_bound, tt_move_arr,
                piece_keys, side_key, castle_keys, ep_file_keys)'''

new_n_razor = '''        if razor_eval + RAZOR_MARGIN_BASE + RAZOR_MARGIN_PER_DEPTH * depth <= alpha:
            razor_score = quiescence(
                bb, occ, mail, st, alpha, alpha + 1, ply, value, pseudo, scores, undo,
                stop, counters, probe, tt_key, tt_score, tt_depth, tt_bound, tt_move_arr,
                piece_keys, side_key, castle_keys, ep_file_keys,
                feature_weights, feature_bias, output_w_stm, output_w_ntm, output_bias,
                white_acc, black_acc, removed_sq, removed_pc, added_sq, added_pc)'''

assert old_n_razor in text, "old_n_razor not found"
text = text.replace(old_n_razor, new_n_razor, 1)

# 5e. Null move pruning
old_n_nmp = '''                null_score = -negamax(
                    bb, occ, mail, st, null_depth, -beta, -beta + 1, ply + 1,
                    null_value, pseudo, scores, undo, stop, counters, probe,
                    path_hashes, game_hashes, game_count,
                    tt_key, tt_score, tt_depth, tt_bound, tt_move_arr,
                    killers, history, continuation_history, prev_moves, corr_hist, excluded,
                    piece_keys, side_key, castle_keys, ep_file_keys,
                    True, False)'''

new_n_nmp = '''                for j in range(white_acc.shape[1]):
                    white_acc[ply + 1, j] = white_acc[ply, j]
                    black_acc[ply + 1, j] = black_acc[ply, j]
                null_score = -negamax(
                    bb, occ, mail, st, null_depth, -beta, -beta + 1, ply + 1,
                    null_value, pseudo, scores, undo, stop, counters, probe,
                    path_hashes, game_hashes, game_count,
                    tt_key, tt_score, tt_depth, tt_bound, tt_move_arr,
                    killers, history, continuation_history, prev_moves, corr_hist, excluded,
                    piece_keys, side_key, castle_keys, ep_file_keys,
                    True, False,
                    feature_weights, feature_bias, output_w_stm, output_w_ntm, output_bias,
                    white_acc, black_acc, removed_sq, removed_pc, added_sq, added_pc)'''

assert old_n_nmp in text, "old_n_nmp not found"
text = text.replace(old_n_nmp, new_n_nmp, 1)

# 5f. Singular search
old_n_sing = '''            singular_score = negamax(
                bb, occ, mail, st, singular_depth,
                singular_beta - 1, singular_beta, ply, value,
                pseudo, scores, undo, stop, counters, probe, path_hashes,
                game_hashes, game_count, tt_key, tt_score, tt_depth, tt_bound,
                tt_move_arr, killers, history, continuation_history, prev_moves,
                corr_hist, excluded, piece_keys, side_key, castle_keys, ep_file_keys,
                True, False)'''

new_n_sing = '''            singular_score = negamax(
                bb, occ, mail, st, singular_depth,
                singular_beta - 1, singular_beta, ply, value,
                pseudo, scores, undo, stop, counters, probe, path_hashes,
                game_hashes, game_count, tt_key, tt_score, tt_depth, tt_bound,
                tt_move_arr, killers, history, continuation_history, prev_moves,
                corr_hist, excluded, piece_keys, side_key, castle_keys, ep_file_keys,
                True, False,
                feature_weights, feature_bias, output_w_stm, output_w_ntm, output_bias,
                white_acc, black_acc, removed_sq, removed_pc, added_sq, added_pc)'''

assert old_n_sing in text, "old_n_sing not found"
text = text.replace(old_n_sing, new_n_sing, 1)

# 5g. Multicut
old_n_mc = '''                mc_old_castle = st[1]
                mc_old_ep = Z.canonical_ep_file(bb, occ, st)
                counters[MAKE] += 1
                make_move(bb, occ, mail, st, mc_move, undo, ply)
                counters[CHECK] += 1
                if in_check(bb, occ, st, side):
                    unmake_move(bb, occ, mail, st, mc_move, undo, ply)
                    continue
                mc_tried += 1
                mc_new_ep = Z.canonical_ep_file(bb, occ, st)
                mc_child = Z.apply_move(value, np.uint64(mc_move), side, mc_old_castle,
                                        mc_old_ep, st[1], mc_new_ep,
                                        piece_keys, side_key, castle_keys, ep_file_keys)
                prev_moves[ply + 1] = mc_move
                mc_score = -negamax(
                    bb, occ, mail, st, mc_reduced - 1, -beta, -beta + 1, ply + 1, mc_child,
                    pseudo, scores, undo, stop, counters, probe, path_hashes,
                    game_hashes, game_count, tt_key, tt_score, tt_depth, tt_bound,
                    tt_move_arr, killers, history, continuation_history, prev_moves,
                    corr_hist, excluded, piece_keys, side_key, castle_keys, ep_file_keys,
                    False, True)'''

new_n_mc = '''                mc_old_castle = st[1]
                mc_old_ep = Z.canonical_ep_file(bb, occ, st)
                old_white_king = lsb(bb[WK])
                old_black_king = lsb(bb[BK])
                counters[MAKE] += 1
                make_move(bb, occ, mail, st, mc_move, undo, ply)
                counters[CHECK] += 1
                if in_check(bb, occ, st, side):
                    unmake_move(bb, occ, mail, st, mc_move, undo, ply)
                    continue
                mc_tried += 1
                advance_accumulators(bb, mail, mc_move, side, white_acc[ply], black_acc[ply],
                                     old_white_king, old_black_king,
                                     feature_weights, feature_bias,
                                     white_acc[ply + 1], black_acc[ply + 1],
                                     removed_sq, removed_pc, added_sq, added_pc)
                mc_new_ep = Z.canonical_ep_file(bb, occ, st)
                mc_child = Z.apply_move(value, np.uint64(mc_move), side, mc_old_castle,
                                        mc_old_ep, st[1], mc_new_ep,
                                        piece_keys, side_key, castle_keys, ep_file_keys)
                prev_moves[ply + 1] = mc_move
                mc_score = -negamax(
                    bb, occ, mail, st, mc_reduced - 1, -beta, -beta + 1, ply + 1, mc_child,
                    pseudo, scores, undo, stop, counters, probe, path_hashes,
                    game_hashes, game_count, tt_key, tt_score, tt_depth, tt_bound,
                    tt_move_arr, killers, history, continuation_history, prev_moves,
                    corr_hist, excluded, piece_keys, side_key, castle_keys, ep_file_keys,
                    False, True,
                    feature_weights, feature_bias, output_w_stm, output_w_ntm, output_bias,
                    white_acc, black_acc, removed_sq, removed_pc, added_sq, added_pc)'''

assert old_n_mc in text, "old_n_mc not found"
text = text.replace(old_n_mc, new_n_mc, 1)

# 5h. Negamax main move loop
old_n_main_loop = '''        old_castle = st[1]
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
        prev_moves[ply + 1] = move
        child_depth = depth - 1 + CHECK_EXTENSION if checked else depth - 1
        if singular_extension != 0 and move == tt_move:
            child_depth += singular_extension
        if legal == 1:
            score = -negamax(bb, occ, mail, st, child_depth, -beta, -alpha, ply + 1, child,
                             pseudo, scores, undo, stop, counters, probe, path_hashes,
                             game_hashes, game_count, tt_key, tt_score, tt_depth, tt_bound,
                             tt_move_arr, killers, history, continuation_history, prev_moves, corr_hist, excluded,
                             piece_keys, side_key, castle_keys, ep_file_keys,
                             False, True)
        else:
            reduction = 0
            if (
                depth >= LMR_MIN_DEPTH
                and legal >= LMR_MIN_MOVES
                and not checked
                and move_captured0 == np.uint32(NO_PIECE)
                and move_promotion0 == np.uint32(NO_PIECE)
                and move != killers[ply, 0]
                and move != killers[ply, 1]
            ):
                depth_index = depth if depth < _LMR_MAX else _LMR_MAX - 1
                move_index = legal if legal < _LMR_MAX else _LMR_MAX - 1
                reduction = LMR_TABLE[depth_index, move_index]
                lmr_piece = (move >> np.uint32(12)) & np.uint32(15)
                lmr_to = (move >> np.uint32(6)) & np.uint32(63)
                lmr_hist = np.int64(history[lmr_piece, lmr_to])
                lmr_prev = prev_moves[ply]
                if lmr_prev != np.uint32(0):
                    lmr_pp = (lmr_prev >> np.uint32(12)) & np.uint32(15)
                    lmr_ps = (lmr_prev >> np.uint32(6)) & np.uint32(63)
                    if lmr_pp < np.uint32(12):
                        lmr_hist += np.int64(
                            continuation_history[lmr_pp, lmr_ps, lmr_piece, lmr_to])
                lmr_adj = lmr_hist // LMR_HISTORY_DIV
                if lmr_adj > LMR_HISTORY_MAX_ADJ:
                    lmr_adj = LMR_HISTORY_MAX_ADJ
                elif lmr_adj < -LMR_HISTORY_MAX_ADJ:
                    lmr_adj = -LMR_HISTORY_MAX_ADJ
                reduction -= lmr_adj
                if reduction > child_depth - 1:
                    reduction = child_depth - 1
                if reduction < 0:
                    reduction = 0
            score = -negamax(bb, occ, mail, st, child_depth - reduction, -alpha - 1, -alpha,
                             ply + 1, child,
                             pseudo, scores, undo, stop, counters, probe, path_hashes,
                             game_hashes, game_count, tt_key, tt_score, tt_depth, tt_bound,
                             tt_move_arr, killers, history, continuation_history, prev_moves, corr_hist, excluded,
                             piece_keys, side_key, castle_keys, ep_file_keys,
                             False, True)
            if reduction > 0 and score > alpha and not stop[0]:
                # Reduced probe beat alpha: the reduction was wrong here, so
                # verify at full depth before trusting the score.
                score = -negamax(bb, occ, mail, st, child_depth, -alpha - 1, -alpha,
                                 ply + 1, child,
                                 pseudo, scores, undo, stop, counters, probe, path_hashes,
                                 game_hashes, game_count, tt_key, tt_score, tt_depth, tt_bound,
                                 tt_move_arr, killers, history, continuation_history, prev_moves, corr_hist, excluded,
                                 piece_keys, side_key, castle_keys, ep_file_keys,
                                 False, True)
            if score > alpha and score < beta and not stop[0]:
                score = -negamax(bb, occ, mail, st, child_depth, -beta, -alpha, ply + 1, child,
                                 pseudo, scores, undo, stop, counters, probe, path_hashes,
                                 game_hashes, game_count, tt_key, tt_score, tt_depth, tt_bound,
                                 tt_move_arr, killers, history, continuation_history, prev_moves, corr_hist, excluded,
                                 piece_keys, side_key, castle_keys, ep_file_keys,
                                 False, True)'''

new_n_main_loop = '''        old_castle = st[1]
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
                             old_white_king, old_black_king,
                             feature_weights, feature_bias,
                             white_acc[ply + 1], black_acc[ply + 1],
                             removed_sq, removed_pc, added_sq, added_pc)
        new_ep_file = Z.canonical_ep_file(bb, occ, st)
        child = Z.apply_move(value, np.uint64(move), side, old_castle, old_ep_file,
                             st[1], new_ep_file, piece_keys, side_key, castle_keys, ep_file_keys)
        # Principal Variation Search.  Good move ordering makes the first legal
        # move the best move surprisingly often.  Search it normally; probe
        # later moves with the minimal window that can answer "does this beat
        # alpha?".  Only a probe that lands strictly inside the full window is
        # re-searched for an exact score.
        prev_moves[ply + 1] = move
        child_depth = depth - 1 + CHECK_EXTENSION if checked else depth - 1
        if singular_extension != 0 and move == tt_move:
            child_depth += singular_extension
        if legal == 1:
            score = -negamax(bb, occ, mail, st, child_depth, -beta, -alpha, ply + 1, child,
                             pseudo, scores, undo, stop, counters, probe, path_hashes,
                             game_hashes, game_count, tt_key, tt_score, tt_depth, tt_bound,
                             tt_move_arr, killers, history, continuation_history, prev_moves, corr_hist, excluded,
                             piece_keys, side_key, castle_keys, ep_file_keys,
                             False, True,
                             feature_weights, feature_bias, output_w_stm, output_w_ntm, output_bias,
                             white_acc, black_acc, removed_sq, removed_pc, added_sq, added_pc)
        else:
            reduction = 0
            if (
                depth >= LMR_MIN_DEPTH
                and legal >= LMR_MIN_MOVES
                and not checked
                and move_captured0 == np.uint32(NO_PIECE)
                and move_promotion0 == np.uint32(NO_PIECE)
                and move != killers[ply, 0]
                and move != killers[ply, 1]
            ):
                depth_index = depth if depth < _LMR_MAX else _LMR_MAX - 1
                move_index = legal if legal < _LMR_MAX else _LMR_MAX - 1
                reduction = LMR_TABLE[depth_index, move_index]
                lmr_piece = (move >> np.uint32(12)) & np.uint32(15)
                lmr_to = (move >> np.uint32(6)) & np.uint32(63)
                lmr_hist = np.int64(history[lmr_piece, lmr_to])
                lmr_prev = prev_moves[ply]
                if lmr_prev != np.uint32(0):
                    lmr_pp = (lmr_prev >> np.uint32(12)) & np.uint32(15)
                    lmr_ps = (lmr_prev >> np.uint32(6)) & np.uint32(63)
                    if lmr_pp < np.uint32(12):
                        lmr_hist += np.int64(
                            continuation_history[lmr_pp, lmr_ps, lmr_piece, lmr_to])
                lmr_adj = lmr_hist // LMR_HISTORY_DIV
                if lmr_adj > LMR_HISTORY_MAX_ADJ:
                    lmr_adj = LMR_HISTORY_MAX_ADJ
                elif lmr_adj < -LMR_HISTORY_MAX_ADJ:
                    lmr_adj = -LMR_HISTORY_MAX_ADJ
                reduction -= lmr_adj
                if reduction > child_depth - 1:
                    reduction = child_depth - 1
                if reduction < 0:
                    reduction = 0
            score = -negamax(bb, occ, mail, st, child_depth - reduction, -alpha - 1, -alpha,
                             ply + 1, child,
                             pseudo, scores, undo, stop, counters, probe, path_hashes,
                             game_hashes, game_count, tt_key, tt_score, tt_depth, tt_bound,
                             tt_move_arr, killers, history, continuation_history, prev_moves, corr_hist, excluded,
                             piece_keys, side_key, castle_keys, ep_file_keys,
                             False, True,
                             feature_weights, feature_bias, output_w_stm, output_w_ntm, output_bias,
                             white_acc, black_acc, removed_sq, removed_pc, added_sq, added_pc)
            if reduction > 0 and score > alpha and not stop[0]:
                # Reduced probe beat alpha: the reduction was wrong here, so
                # verify at full depth before trusting the score.
                score = -negamax(bb, occ, mail, st, child_depth, -alpha - 1, -alpha,
                                 ply + 1, child,
                                 pseudo, scores, undo, stop, counters, probe, path_hashes,
                                 game_hashes, game_count, tt_key, tt_score, tt_depth, tt_bound,
                                 tt_move_arr, killers, history, continuation_history, prev_moves, corr_hist, excluded,
                                 piece_keys, side_key, castle_keys, ep_file_keys,
                                 False, True,
                                 feature_weights, feature_bias, output_w_stm, output_w_ntm, output_bias,
                                 white_acc, black_acc, removed_sq, removed_pc, added_sq, added_pc)
            if score > alpha and score < beta and not stop[0]:
                score = -negamax(bb, occ, mail, st, child_depth, -beta, -alpha, ply + 1, child,
                                 pseudo, scores, undo, stop, counters, probe, path_hashes,
                                 game_hashes, game_count, tt_key, tt_score, tt_depth, tt_bound,
                                 tt_move_arr, killers, history, continuation_history, prev_moves, corr_hist, excluded,
                                 piece_keys, side_key, castle_keys, ep_file_keys,
                                 False, True,
                                 feature_weights, feature_bias, output_w_stm, output_w_ntm, output_bias,
                                 white_acc, black_acc, removed_sq, removed_pc, added_sq, added_pc)'''

assert old_n_main_loop in text, "old_n_main_loop not found"
text = text.replace(old_n_main_loop, new_n_main_loop, 1)

# 6. Search Root
# 6a. Signature
old_root_sig = '''def search_root(bb, occ, mail, st, depth, value, pseudo, scores, undo, stop, counters,
                probe, path_hashes, game_hashes, game_count,
                tt_key, tt_score, tt_depth, tt_bound, tt_move_arr,
                killers, history, continuation_history, prev_moves, corr_hist, excluded,
                piece_keys, side_key, castle_keys, ep_file_keys,
                root_moves, root_scores, previous_best, root_alpha, root_beta):'''

new_root_sig = '''def search_root(bb, occ, mail, st, depth, value, pseudo, scores, undo, stop, counters,
                probe, path_hashes, game_hashes, game_count,
                tt_key, tt_score, tt_depth, tt_bound, tt_move_arr,
                killers, history, continuation_history, prev_moves, corr_hist, excluded,
                piece_keys, side_key, castle_keys, ep_file_keys,
                root_moves, root_scores, previous_best, root_alpha, root_beta,
                feature_weights, feature_bias, output_w_stm, output_w_ntm, output_bias,
                white_acc, black_acc, removed_sq, removed_pc, added_sq, added_pc):'''

assert old_root_sig in text, "old_root_sig not found"
text = text.replace(old_root_sig, new_root_sig, 1)

# 6b. Root initialization of accumulator at ply 0
old_root_init = '''    side = st[0]
    path_hashes[0] = value
    count = generate_pseudo_legal(bb, occ, mail, st, pseudo[0])'''

new_root_init = '''    side = st[0]
    path_hashes[0] = value
    white_king = lsb(bb[WK])
    black_king = lsb(bb[BK])
    res_w = _dbnnue.refresh(feature_weights, feature_bias, mail, white_king, 0)
    res_b = _dbnnue.refresh(feature_weights, feature_bias, mail, black_king, 1)
    for j in range(white_acc.shape[1]):
        white_acc[0, j] = res_w[j]
        black_acc[0, j] = res_b[j]
    count = generate_pseudo_legal(bb, occ, mail, st, pseudo[0])'''

assert old_root_init in text, "old_root_init not found"
text = text.replace(old_root_init, new_root_init, 1)

# 6c. Root move loop
old_root_loop = '''        old_castle = st[1]
        old_ep_file = Z.canonical_ep_file(bb, occ, st)
        make_move(bb, occ, mail, st, move, undo, 0)
        if in_check(bb, occ, st, side):
            unmake_move(bb, occ, mail, st, move, undo, 0)
            continue
        new_ep_file = Z.canonical_ep_file(bb, occ, st)
        child = Z.apply_move(value, np.uint64(move), side, old_castle, old_ep_file,
                             st[1], new_ep_file, piece_keys, side_key, castle_keys, ep_file_keys)
        prev_moves[1] = move
        if legal == 0:
            score = -negamax(bb, occ, mail, st, depth - 1, -root_beta, -alpha,
                             np.int64(1), child,
                             pseudo, scores, undo, stop, counters, probe, path_hashes,
                             game_hashes, game_count, tt_key, tt_score, tt_depth, tt_bound,
                             tt_move_arr, killers, history, continuation_history, prev_moves, corr_hist, excluded,
                             piece_keys, side_key, castle_keys, ep_file_keys,
                             False, True)
        else:
            score = -negamax(bb, occ, mail, st, depth - 1, -alpha - 1, -alpha, np.int64(1), child,
                             pseudo, scores, undo, stop, counters, probe, path_hashes,
                             game_hashes, game_count, tt_key, tt_score, tt_depth, tt_bound,
                             tt_move_arr, killers, history, continuation_history, prev_moves, corr_hist, excluded,
                             piece_keys, side_key, castle_keys, ep_file_keys,
                             False, True)
            if score > alpha and not stop[0]:
                score = -negamax(bb, occ, mail, st, depth - 1, -root_beta, -alpha,
                             np.int64(1), child,
                                 pseudo, scores, undo, stop, counters, probe, path_hashes,
                                 game_hashes, game_count, tt_key, tt_score, tt_depth, tt_bound,
                                 tt_move_arr, killers, history, continuation_history, prev_moves, corr_hist, excluded,
                                 piece_keys, side_key, castle_keys, ep_file_keys,
                                 False, True)'''

new_root_loop = '''        old_castle = st[1]
        old_ep_file = Z.canonical_ep_file(bb, occ, st)
        old_white_king = lsb(bb[WK])
        old_black_king = lsb(bb[BK])
        make_move(bb, occ, mail, st, move, undo, 0)
        if in_check(bb, occ, st, side):
            unmake_move(bb, occ, mail, st, move, undo, 0)
            continue
        advance_accumulators(bb, mail, move, side, white_acc[0], black_acc[0],
                             old_white_king, old_black_king,
                             feature_weights, feature_bias,
                             white_acc[1], black_acc[1],
                             removed_sq, removed_pc, added_sq, added_pc)
        new_ep_file = Z.canonical_ep_file(bb, occ, st)
        child = Z.apply_move(value, np.uint64(move), side, old_castle, old_ep_file,
                             st[1], new_ep_file, piece_keys, side_key, castle_keys, ep_file_keys)
        prev_moves[1] = move
        if legal == 0:
            score = -negamax(bb, occ, mail, st, depth - 1, -root_beta, -alpha,
                             np.int64(1), child,
                             pseudo, scores, undo, stop, counters, probe, path_hashes,
                             game_hashes, game_count, tt_key, tt_score, tt_depth, tt_bound,
                             tt_move_arr, killers, history, continuation_history, prev_moves, corr_hist, excluded,
                             piece_keys, side_key, castle_keys, ep_file_keys,
                             False, True,
                             feature_weights, feature_bias, output_w_stm, output_w_ntm, output_bias,
                             white_acc, black_acc, removed_sq, removed_pc, added_sq, added_pc)
        else:
            score = -negamax(bb, occ, mail, st, depth - 1, -alpha - 1, -alpha, np.int64(1), child,
                             pseudo, scores, undo, stop, counters, probe, path_hashes,
                             game_hashes, game_count, tt_key, tt_score, tt_depth, tt_bound,
                             tt_move_arr, killers, history, continuation_history, prev_moves, corr_hist, excluded,
                             piece_keys, side_key, castle_keys, ep_file_keys,
                             False, True,
                             feature_weights, feature_bias, output_w_stm, output_w_ntm, output_bias,
                             white_acc, black_acc, removed_sq, removed_pc, added_sq, added_pc)
            if score > alpha and not stop[0]:
                score = -negamax(bb, occ, mail, st, depth - 1, -root_beta, -alpha,
                             np.int64(1), child,
                                 pseudo, scores, undo, stop, counters, probe, path_hashes,
                                 game_hashes, game_count, tt_key, tt_score, tt_depth, tt_bound,
                                 tt_move_arr, killers, history, continuation_history, prev_moves, corr_hist, excluded,
                                 piece_keys, side_key, castle_keys, ep_file_keys,
                                 False, True,
                                 feature_weights, feature_bias, output_w_stm, output_w_ntm, output_bias,
                                 white_acc, black_acc, removed_sq, removed_pc, added_sq, added_pc)'''

assert old_root_loop in text, "old_root_loop not found"
text = text.replace(old_root_loop, new_root_loop, 1)

# 7. FastEngine180 -> FastEngine180n
text = text.replace("class FastEngine180:", "class FastEngine180n:", 1)

# In FastEngine180n.__init__
old_engine_init = '''        self.excluded = np.zeros(MAX_PLY + 2, dtype=np.uint32)'''
new_engine_init = '''        self.excluded = np.zeros(MAX_PLY + 2, dtype=np.uint32)
        self.fw, self.fb, self.w_stm, self.w_ntm, self.ob = _NNUE_FW, _NNUE_FB, _NNUE_WSTM, _NNUE_WNTM, _NNUE_OB
        hidden_size = self.fb.shape[0]
        self.white_acc = np.zeros((MAX_PLY + 2, hidden_size), dtype=np.int16)
        self.black_acc = np.zeros((MAX_PLY + 2, hidden_size), dtype=np.int16)
        self.removed_sq = np.zeros(4, dtype=np.int64)
        self.removed_pc = np.zeros(4, dtype=np.int64)
        self.added_sq = np.zeros(4, dtype=np.int64)
        self.added_pc = np.zeros(4, dtype=np.int64)'''

assert old_engine_init in text, "old_engine_init not found"
text = text.replace(old_engine_init, new_engine_init, 1)

# In FastEngine180n.search, pass NNUE args to search_root
old_search_call = '''                    legal = search_root(
                        bb, occ, mail, st, depth, value, self.pseudo, self.scores, self.undo,
                        self.stop, self.counters, self.probe, self.path_hashes,
                        self.game_hashes, self.game_count,
                        self.tt_key, self.tt_score, self.tt_depth, self.tt_bound, self.tt_move,
                        self.killers, self.history,
                        self.continuation_history, self.prev_moves, self.corr_hist,
                        self.excluded,
                        Z.PIECE_KEYS, np.uint64(Z.SIDE_KEY), Z.CASTLE_KEYS, Z.EP_FILE_KEYS,
                        self.root_moves, self.root_scores, np.uint32(best_move),
                        root_alpha, root_beta,
                    )'''

new_search_call = '''                    legal = search_root(
                        bb, occ, mail, st, depth, value, self.pseudo, self.scores, self.undo,
                        self.stop, self.counters, self.probe, self.path_hashes,
                        self.game_hashes, self.game_count,
                        self.tt_key, self.tt_score, self.tt_depth, self.tt_bound, self.tt_move,
                        self.killers, self.history,
                        self.continuation_history, self.prev_moves, self.corr_hist,
                        self.excluded,
                        Z.PIECE_KEYS, np.uint64(Z.SIDE_KEY), Z.CASTLE_KEYS, Z.EP_FILE_KEYS,
                        self.root_moves, self.root_scores, np.uint32(best_move),
                        root_alpha, root_beta,
                        self.fw, self.fb, self.w_stm, self.w_ntm, self.ob,
                        self.white_acc, self.black_acc, self.removed_sq, self.removed_pc, self.added_sq, self.added_pc,
                    )'''

assert old_search_call in text, "old_search_call not found"
text = text.replace(old_search_call, new_search_call, 1)

# 8. warm_up()
old_warmup = '''def warm_up() -> None:
    engine = FastEngine180()
    for fen in (
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ):
        engine.search(from_fen(fen), 40.0, 80.0, max_depth=3)'''

new_warmup = '''def warm_up() -> None:
    engine = FastEngine180n()
    for fen in (
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ):
        engine.search(from_fen(fen), 40.0, 80.0, max_depth=2)'''

assert old_warmup in text, "old_warmup not found"
text = text.replace(old_warmup, new_warmup, 1)

with open("deepblue/fastsearch180n.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Generated deepblue/fastsearch180n.py successfully! Total lines:", len(text.splitlines()))
