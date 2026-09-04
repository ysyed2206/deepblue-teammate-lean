"""Correctness gate: NNUE incremental accumulator vs. full refresh, driven by
Deep Blue's own fastcore make_move/unmake_move (not a chess.Board prototype).
Ad-hoc verification tool for the fastsearch30 NNUE integration; not part of
the permanent regression suite.
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepblue import nnue as dnnue
from deepblue.fastcore import (
    NO_PIECE,
    UNDO_STRIDE,
    decode,
    from_fen,
    generate_pseudo_legal,
    in_check,
    make_move,
    unmake_move,
)

MAX_MOVES = 320


def king_squares(mail):
    wk = int(np.flatnonzero(mail == 5)[0])
    bk = int(np.flatnonzero(mail == 11)[0])
    return wk, bk


def run_gate(num_transitions: int, seed: int) -> dict:
    fw, fb, w_stm, w_ntm, ob = dnnue.load_weights("weights/deepblue_nnue_v1.bin")
    dnnue.warm_up(fw, fb, w_stm, w_ntm, ob)

    rng = random.Random(seed)
    removed_sq = np.zeros(4, dtype=np.int64)
    removed_pc = np.zeros(4, dtype=np.int64)
    added_sq = np.zeros(4, dtype=np.int64)
    added_pc = np.zeros(4, dtype=np.int64)

    categories = {
        "quiet": 0, "capture": 0, "en_passant": 0, "king_same_bucket": 0,
        "king_bucket_change": 0, "castle_k": 0, "castle_q": 0,
        "promotion": 0, "capture_promotion": 0, "underpromotion": 0,
    }
    acc_mismatches = 0
    checked = 0

    def fresh_state():
        board = chess.Board()
        bb, occ, mail, st = from_fen(board.fen())
        wk, bk = king_squares(mail)
        white_acc = dnnue.refresh(fw, fb, mail.astype(np.int64), wk, 0)
        black_acc = dnnue.refresh(fw, fb, mail.astype(np.int64), bk, 1)
        return board, bb, occ, mail, st, wk, bk, white_acc, black_acc

    board, bb, occ, mail, st, white_king, black_king, white_acc, black_acc = fresh_state()
    pseudo = np.zeros(MAX_MOVES, dtype=np.uint32)
    undo = np.zeros(256 * UNDO_STRIDE, dtype=np.int64)

    def classify(from_sq, to_sq, piece, captured, promotion, is_ep, is_castle):
        if is_ep:
            categories["en_passant"] += 1
        elif is_castle:
            if to_sq in (6, 62):
                categories["castle_k"] += 1
            else:
                categories["castle_q"] += 1
        elif promotion != NO_PIECE:
            if captured != NO_PIECE:
                categories["capture_promotion"] += 1
            else:
                categories["promotion"] += 1
            if promotion % 6 != 4:  # not queen
                categories["underpromotion"] += 1
        elif captured != NO_PIECE:
            categories["capture"] += 1
        else:
            categories["quiet"] += 1

    ply = 0
    while checked < num_transitions:
        count = generate_pseudo_legal(bb, occ, mail, st, pseudo)
        legal_moves = []
        for i in range(count):
            move = int(pseudo[i])
            side = int(st[0])
            make_move(bb, occ, mail, st, move, undo, ply)
            if not in_check(bb, occ, st, side):
                legal_moves.append(move)
            unmake_move(bb, occ, mail, st, move, undo, ply)
        if not legal_moves or ply >= 200:
            board, bb, occ, mail, st, white_king, black_king, white_acc, black_acc = fresh_state()
            ply = 0
            continue

        move = rng.choice(legal_moves)
        from_sq, to_sq, piece, captured, promotion, is_ep, is_castle, is_double = decode(move)
        side = int(st[0])
        classify(from_sq, to_sq, piece, captured, promotion, is_ep, is_castle)

        make_move(bb, occ, mail, st, move, undo, ply)
        ply += 1
        checked += 1

        new_white_king, new_black_king = king_squares(mail)
        new_white_bucket = dnnue.king_bucket(new_white_king, 0)
        old_white_bucket = dnnue.king_bucket(white_king, 0)
        new_black_bucket = dnnue.king_bucket(new_black_king, 1)
        old_black_bucket = dnnue.king_bucket(black_king, 1)

        if piece == 5 or piece == 11:  # a king moved
            if new_white_bucket == old_white_bucket and new_black_bucket == old_black_bucket:
                categories["king_same_bucket"] += 1
            else:
                categories["king_bucket_change"] += 1

        rc, ac = dnnue.compute_delta(from_sq, to_sq, piece, captured, promotion, is_ep, is_castle, side,
                                      removed_sq, removed_pc, added_sq, added_pc)

        if new_white_bucket == old_white_bucket:
            white_acc = dnnue.apply_delta(white_acc, fw, removed_sq, removed_pc, rc, added_sq, added_pc, ac, white_king, 0)
        else:
            white_acc = dnnue.refresh(fw, fb, mail.astype(np.int64), new_white_king, 0)

        if new_black_bucket == old_black_bucket:
            black_acc = dnnue.apply_delta(black_acc, fw, removed_sq, removed_pc, rc, added_sq, added_pc, ac, black_king, 1)
        else:
            black_acc = dnnue.refresh(fw, fb, mail.astype(np.int64), new_black_king, 1)

        white_king, black_king = new_white_king, new_black_king

        ref_white = dnnue.refresh(fw, fb, mail.astype(np.int64), white_king, 0)
        ref_black = dnnue.refresh(fw, fb, mail.astype(np.int64), black_king, 1)
        if not np.array_equal(white_acc, ref_white) or not np.array_equal(black_acc, ref_black):
            acc_mismatches += 1
            if acc_mismatches <= 3:
                print(f"MISMATCH at transition {checked}: from={from_sq} to={to_sq} piece={piece} "
                      f"captured={captured} promo={promotion} ep={is_ep} castle={is_castle}")

    return {"checked": checked, "acc_mismatches": acc_mismatches, "categories": categories}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    t0 = time.perf_counter()
    result = run_gate(n, seed=20260904)
    elapsed = time.perf_counter() - t0
    print(f"\ntransitions checked: {result['checked']}")
    print(f"accumulator mismatches: {result['acc_mismatches']}")
    print(f"categories: {result['categories']}")
    print(f"elapsed: {elapsed:.1f}s")
    raise SystemExit(1 if result["acc_mismatches"] else 0)
