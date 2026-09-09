"""Production NNUE evaluator: loading, feature mapping, incremental
accumulator maintenance, and the SCReLU tail -- built directly on Deep
Blue's own board representation (``deepblue.fastcore``'s ``bb``/``mail``/
``decode(move)``), not the ``chess.Board``-based research prototype in
``nnue_lab/teammate_pawnstar/``.

Architecture (1024-wide, 8 king buckets, verified against the Pawnstar-v12
format -- see ``nnue_lab/teammate_pawnstar/PAWNSTAR_FORMAT.md``): the weight
file is the same ``PSN1``-stamped binary format loaded there; only the
*calling convention* differs here, since this module is called from inside
Deep Blue's own njit-compiled search rather than from Python.

Deep Blue piece codes: ``WP,WN,WB,WR,WQ,WK=0..5``, ``BP,BN,BB_,BR,BQ,BK=6..11``,
``NO_PIECE=15``. ``colour = piece // 6``, ``piece_type = piece % 6`` -- this
is the exact same split used by ``nnue_lab/features.py`` and verified there
to be mathematically identical to the donor's own feature formula.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from numba import njit

from deepblue.fastcore import NO_PIECE

INPUT_SIZE = 768
NUM_KING_BUCKETS = 8
FEATURE_ROWS = INPUT_SIZE * NUM_KING_BUCKETS  # 6144
HIDDEN_SIZE = 1024
QA = 255
QB = 64
SCALE = 400
RANK_FLIP = 0x38

NET_MAGIC = b"PSN1"
HEADER_STRUCT = struct.Struct("<4sHHHHhhh14s")
_PAYLOAD_INT16_COUNT = FEATURE_ROWS * HIDDEN_SIZE + HIDDEN_SIZE + 2 * HIDDEN_SIZE + 1
PAYLOAD_BYTES = _PAYLOAD_INT16_COUNT * 2

DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "weights" / "deepblue_nnue_v1.bin"


def load_weights(path: Path | str = DEFAULT_WEIGHTS_PATH):
    """Load and validate the stamped net; returns arrays ready for the njit
    kernels below (Numba functions receive plain NumPy arrays, not a
    dataclass, since dataclasses of arrays are awkward to pass into njit).

    ``hidden_size`` (the network width) is read from the file's own header,
    not hardcoded -- the feature/bucket scheme (``INPUT_SIZE``,
    ``NUM_KING_BUCKETS``) and the quantisation scale (``QA``/``QB``/``SCALE``,
    a numeric convention, not anyone's trained weights) are the only things
    validated as fixed, so this loader works unchanged whether the trained
    network is width 128, 1024, or anything else -- every njit kernel below
    already sizes its loops from ``acc.shape[0]``, not a module constant."""
    data = Path(path).read_bytes()
    if len(data) < HEADER_STRUCT.size or data[:4] != NET_MAGIC:
        raise ValueError(f"'{path}' is not a stamped NNUE net (missing 'PSN1' header)")
    magic, version, input_size, king_buckets, hidden_size, qa, qb, scale, _reserved = HEADER_STRUCT.unpack_from(
        data, 0
    )
    if (input_size, king_buckets, qa, qb, scale) != (INPUT_SIZE, NUM_KING_BUCKETS, QA, QB, SCALE):
        raise ValueError(
            f"'{path}' architecture mismatch: in{input_size}/buckets{king_buckets}/"
            f"qa{qa}/qb{qb}/scale{scale}, expected in{INPUT_SIZE}/buckets{NUM_KING_BUCKETS}/"
            f"qa{QA}/qb{QB}/scale{SCALE} (hidden_size is flexible: file says {hidden_size})"
        )
    feature_rows = INPUT_SIZE * NUM_KING_BUCKETS
    payload_bytes = (feature_rows * hidden_size + hidden_size + 2 * hidden_size + 1) * 2
    payload = data[HEADER_STRUCT.size :]
    if len(payload) < payload_bytes:
        raise ValueError(f"'{path}' truncated payload")

    offset = 0
    fw_count = feature_rows * hidden_size
    feature_weights = np.frombuffer(payload, dtype="<i2", count=fw_count, offset=offset).reshape(
        feature_rows, hidden_size
    ).copy()
    offset += fw_count * 2
    feature_bias = np.frombuffer(payload, dtype="<i2", count=hidden_size, offset=offset).copy()
    offset += hidden_size * 2
    output_weights_flat = np.frombuffer(payload, dtype="<i2", count=2 * hidden_size, offset=offset)
    output_weights = output_weights_flat.reshape(2, hidden_size).copy()
    offset += 2 * hidden_size * 2
    (output_bias,) = struct.unpack_from("<h", payload, offset)

    return feature_weights, feature_bias, output_weights[0].copy(), output_weights[1].copy(), np.int64(output_bias)


@njit(cache=True, nogil=True)
def king_bucket(king_square, perspective):
    oriented = king_square if perspective == 0 else (king_square ^ 56)
    file_pair = (oriented & 7) >> 1
    advanced_half = 4 if (oriented >> 3) >= 4 else 0
    return file_pair + advanced_half


@njit(cache=True, nogil=True)
def feature_row(piece_code, square, king_square, perspective):
    colour = piece_code // 6
    piece_type = piece_code % 6
    bucket = king_bucket(king_square, perspective)
    if perspective == 0:
        slot_colour = colour
        oriented_square = square
    else:
        slot_colour = 1 - colour
        oriented_square = square ^ 56
    return bucket * 768 + slot_colour * 384 + piece_type * 64 + oriented_square


@njit(cache=True, nogil=True)
def refresh(feature_weights, feature_bias, mail, king_square, perspective):
    acc = feature_bias.copy()
    for square in range(64):
        piece = mail[square]
        if piece != NO_PIECE:
            row = feature_row(piece, square, king_square, perspective)
            for j in range(acc.shape[0]):
                acc[j] = np.int16(acc[j] + feature_weights[row, j])
    return acc


@njit(cache=True, nogil=True)
def apply_delta(acc, feature_weights, removed_squares, removed_pieces, removed_count,
                 added_squares, added_pieces, added_count, king_square, perspective):
    """Apply a piece-placement delta to one perspective's accumulator, using
    a SINGLE king square/bucket for every row in this call -- correct only
    when that perspective's king bucket is unchanged across the move (the
    caller is responsible for choosing full ``refresh`` instead when it
    isn't)."""
    out = acc.copy()
    for i in range(removed_count):
        row = feature_row(removed_pieces[i], removed_squares[i], king_square, perspective)
        for j in range(out.shape[0]):
            out[j] = np.int16(out[j] - feature_weights[row, j])
    for i in range(added_count):
        row = feature_row(added_pieces[i], added_squares[i], king_square, perspective)
        for j in range(out.shape[0]):
            out[j] = np.int16(out[j] + feature_weights[row, j])
    return out


@njit(cache=True, nogil=True)
def compute_delta(from_square, to_square, piece, captured, promotion, is_ep, is_castle, side,
                   removed_squares, removed_pieces, added_squares, added_pieces):
    """Fill the (size-4, preallocated by the caller) removed/added arrays
    from one decoded move's fields -- mirrors ``fastcore.make_move`` exactly
    (same en passant capture-square offset, same castling rook squares) so
    every move type (quiet, capture, en passant, castling, promotion,
    capture-promotion) falls out without special-casing beyond what's here.
    Returns (removed_count, added_count)."""
    removed_count = 0
    added_count = 0

    removed_squares[removed_count] = from_square
    removed_pieces[removed_count] = piece
    removed_count += 1

    if captured != NO_PIECE:
        if is_ep:
            capture_square = to_square - 8 if side == 0 else to_square + 8
        else:
            capture_square = to_square
        removed_squares[removed_count] = capture_square
        removed_pieces[removed_count] = captured
        removed_count += 1

    landed = promotion if promotion != NO_PIECE else piece
    added_squares[added_count] = to_square
    added_pieces[added_count] = landed
    added_count += 1

    if is_castle:
        if to_square == 6:
            rook_from, rook_to, rook = 7, 5, 3  # WR
        elif to_square == 2:
            rook_from, rook_to, rook = 0, 3, 3  # WR
        elif to_square == 62:
            rook_from, rook_to, rook = 63, 61, 9  # BR
        else:
            rook_from, rook_to, rook = 56, 59, 9  # BR
        removed_squares[removed_count] = rook_from
        removed_pieces[removed_count] = rook
        removed_count += 1
        added_squares[added_count] = rook_to
        added_pieces[added_count] = rook
        added_count += 1

    return removed_count, added_count


@njit(cache=True, nogil=True)
def tail(white_acc, black_acc, side_to_move, output_weights_stm_side, output_weights_other_side, output_bias):
    """Exact int16 SCReLU tail. ``side_to_move``: 0=white, 1=black. Returns
    centipawns, relative to the side to move -- matches Deep Blue's own
    ``evaluate()`` sign convention exactly (flipped once at the end)."""
    # The accumulators swap with the side to move; the output weights must NOT.
    # ``model.py``'s training forward pass concatenates (stm_acc, non_stm_acc)
    # in that fixed order, so the two output-weight halves are indexed by ROLE
    # (side-to-move vs other side), never by colour. Swapping both together
    # makes the two branches compute the same commutative sum, which silently
    # returns a white-relative score to a negamax that requires a
    # side-to-move-relative one -- the engine then steers Black into losing
    # positions on purpose. Measured: 0/96 games before this fix.
    if side_to_move == 0:
        stm_acc = white_acc
        ntm_acc = black_acc
    else:
        stm_acc = black_acc
        ntm_acc = white_acc
    w_stm = output_weights_stm_side
    w_ntm = output_weights_other_side

    qa = np.int64(QA)
    dot = np.int64(0)
    for i in range(stm_acc.shape[0]):
        x = np.int64(stm_acc[i])
        c = np.int64(0) if x < 0 else (qa if x > qa else x)
        dot += c * c * np.int64(w_stm[i])
    for i in range(ntm_acc.shape[0]):
        x = np.int64(ntm_acc[i])
        c = np.int64(0) if x < 0 else (qa if x > qa else x)
        dot += c * c * np.int64(w_ntm[i])

    out = dot // qa
    out += output_bias
    out *= np.int64(SCALE)
    denom = qa * np.int64(QB)
    q = out // denom
    if q < 0 and out % denom != 0:
        q += 1
    return q


def warm_up(feature_weights, feature_bias, w_stm, w_ntm, output_bias) -> None:
    """Force JIT compilation of every kernel above, outside any timed
    region -- called once at import, matching the rest of this project's
    convention for compiled search functions."""
    mail = np.full(64, NO_PIECE, dtype=np.int64)
    mail[4] = 5  # WK
    mail[60] = 11  # BK
    white_acc = refresh(feature_weights, feature_bias, mail, 4, 0)
    black_acc = refresh(feature_weights, feature_bias, mail, 60, 1)
    removed_sq = np.zeros(4, dtype=np.int64)
    removed_pc = np.zeros(4, dtype=np.int64)
    added_sq = np.zeros(4, dtype=np.int64)
    added_pc = np.zeros(4, dtype=np.int64)
    apply_delta(white_acc, feature_weights, removed_sq, removed_pc, 0, added_sq, added_pc, 0, 4, 0)
    tail(white_acc, black_acc, 0, w_stm, w_ntm, output_bias)
