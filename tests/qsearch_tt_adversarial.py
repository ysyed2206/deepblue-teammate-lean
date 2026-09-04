"""Adversarial regression suite for fastsearch18's qsearch transposition table.

fastsearch18 (parented on fastsearch4, the champion) is the first Deep Blue
variant to thread the zobrist hash into quiescence and probe/store the shared
TT from inside qsearch. fastsearch4's quiescence never touches the hash or the
table at all. This script is a standalone, hand-run adversarial check for that
one feature -- it does NOT touch tests/regression_fens.txt or any other file.

Two calling styles are used:

  * "Direct" tests call ``fastsearch18.quiescence`` (and, for comparison,
    ``fastsearch4.quiescence``) exactly the way ``negamax`` calls it, with a
    hand-built scratch harness (see ``QHarness``/``TT``). This lets a test
    fix the *ply* and the halfmove clock independently of whatever the real
    iterative-deepening search would have picked, which is required to
    reproduce the "same node, different ply" and "TT populated at clock X,
    read at clock Y" scenarios the mate-distance and fifty-move categories
    need.
  * "Integration" tests call ``FastEngine18.search`` / ``FastEngine4.search``
    the same way ``tools/regression_fast_variant.py`` does, to confirm the
    effect (or its absence) survives up through iterative deepening and to
    print fastsearch18 next to the TT-less fastsearch4 baseline so a
    divergence is visible immediately.

Every check prints a pass/FAIL line, the FEN, what was expected and what was
actually returned. Run directly with the project's interpreter:

    uv run python tests/qsearch_tt_adversarial.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from deepblue import fastsearch18 as fs18  # noqa: E402
from deepblue import fastsearch4 as fs4  # noqa: E402
from deepblue import zobrist as Z  # noqa: E402
from deepblue.constants import DRAW_SCORE, INFINITY, MATE_SCORE, MATE_THRESHOLD  # noqa: E402
from deepblue.fastcore import (  # noqa: E402
    MAX_MOVES,
    MAX_PLY,
    NO_PIECE,
    UNDO_STRIDE,
    decode,
    from_fen,
    generate_pseudo_tactical,
    move_to_uci,
)

# ---------------------------------------------------------------- reporting

_total = 0
_failures = 0


def report(tag: str, ok: bool, fen: str, expected: str, actual: str, note: str = "") -> bool:
    global _total, _failures
    _total += 1
    if not ok:
        _failures += 1
    mark = "pass" if ok else "FAIL"
    print(f"  {mark}  {tag:32s} expected: {expected}")
    print(f"        {'':32s} actual:   {actual}")
    print(f"        fen  {fen}")
    if note:
        print(f"        note {note}")
    return ok


def section(title: str) -> None:
    print(f"\n-- {title} " + "-" * max(0, 60 - len(title)))


# ---------------------------------------------------------------- harness


class TT:
    """A standalone qsearch-shaped transposition table, sized and typed
    exactly like FastEngine18's own arrays, so a test can control exactly
    what is (or is not) already cached before a probe."""

    def __init__(self) -> None:
        self.tt_key = np.zeros(fs18.TT_SIZE, dtype=np.uint64)
        self.tt_score = np.zeros(fs18.TT_SIZE, dtype=np.int32)
        self.tt_depth = np.full(fs18.TT_SIZE, -1, dtype=np.int16)
        self.tt_bound = np.zeros(fs18.TT_SIZE, dtype=np.int8)
        self.tt_move = np.zeros(fs18.TT_SIZE, dtype=np.uint32)


class QHarness:
    """Calls fastsearch18.quiescence directly, with the same scratch-buffer
    shapes negamax passes it, but full control over ply, alpha/beta and the
    TT arrays used."""

    def __init__(self) -> None:
        self.pseudo = np.zeros((MAX_PLY, MAX_MOVES), dtype=np.uint32)
        self.scores = np.zeros((MAX_PLY, MAX_MOVES), dtype=np.int32)
        self.undo = np.zeros(MAX_PLY * UNDO_STRIDE, dtype=np.int64)
        self.probe = np.zeros(MAX_MOVES, dtype=np.uint32)
        self.stop = np.zeros(1, dtype=np.uint8)
        self.counters = np.zeros(16, dtype=np.int64)

    def hash_of(self, bb, occ, st) -> "np.uint64":
        return np.uint64(
            Z.full_hash(bb, occ, st, Z.PIECE_KEYS, Z.SIDE_KEY, Z.CASTLE_KEYS, Z.EP_FILE_KEYS)
        )

    def run(self, fen: str, tt: TT, ply: int = 0, alpha=-INFINITY, beta=INFINITY, halfmove=None):
        bb, occ, mail, st = from_fen(fen)
        if halfmove is not None:
            st[3] = halfmove
        value = self.hash_of(bb, occ, st)
        self.stop[0] = 0
        self.counters[:] = 0
        score = fs18.quiescence(
            bb, occ, mail, st, alpha, beta, ply, value,
            self.pseudo, self.scores, self.undo, self.stop, self.counters, self.probe,
            tt.tt_key, tt.tt_score, tt.tt_depth, tt.tt_bound, tt.tt_move,
            Z.PIECE_KEYS, np.uint64(Z.SIDE_KEY), Z.CASTLE_KEYS, Z.EP_FILE_KEYS,
        )
        return int(score), value, tt

    def run4(self, fen: str, ply: int = 0, alpha=-INFINITY, beta=INFINITY, halfmove=None):
        """Same position through fastsearch4's TT-less quiescence, for a
        side-by-side baseline."""
        bb, occ, mail, st = from_fen(fen)
        if halfmove is not None:
            st[3] = halfmove
        self.stop[0] = 0
        self.counters[:] = 0
        score = fs4.quiescence(
            bb, occ, mail, st, alpha, beta, ply,
            self.pseudo, self.scores, self.undo, self.stop, self.counters, self.probe,
        )
        return int(score)


def stand_pat_of(fen: str, halfmove=None) -> int:
    bb, occ, mail, st = from_fen(fen)
    if halfmove is not None:
        st[3] = halfmove
    return int(fs18.evaluate(bb, st, fs18.MG_TABLE, fs18.EG_TABLE, fs18.PHASE_TABLE))


# A handful of unrelated positions used purely to "dirty" a TT before the
# real assertion, simulating a table that has already done a lot of
# unrelated work -- exactly the situation described in the brief ("does
# repeatedly probing/writing ... produce a consistent score").
WARM_UP_FENS = [
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "8/P6k/8/8/8/8/6K1/8 w - - 0 1",
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "3k4/3p4/8/K1P4r/8/8/8/8 b - - 0 1",
]


def dirty_engine() -> "fs18.FastEngine18":
    engine = fs18.FastEngine18()
    for fen in WARM_UP_FENS:
        engine.search(from_fen(fen), 150.0, 250.0, max_depth=3)
    return engine


# ============================================================= category 1


def category1_tt_consistency() -> None:
    section("1. qsearch TT consistency under repeated/dirty use (no reset)")
    fs18.warm_up()
    kiwipete = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"

    # 1a. Same engine, same position, searched twice back to back without
    # resetting the table. A fixed max_depth (rather than a time budget)
    # keeps iterative deepening deterministic so this is a real identity
    # check, not just "close enough".
    engine = fs18.FastEngine18()
    r1 = engine.search(from_fen(kiwipete), 5000.0, 8000.0, max_depth=4)
    r2 = engine.search(from_fen(kiwipete), 5000.0, 8000.0, max_depth=4)
    ok = r1[:3] == r2[:3]
    report(
        "repeat_same_engine", ok, kiwipete,
        f"identical (move, score, depth) both times: {r1[:3]}",
        f"first={r1[:3]} second={r2[:3]}",
    )

    # 1b. A "cold" engine (brand-new, empty TT) vs a "dirty" engine that has
    # already searched several unrelated positions into the SAME table
    # first. If a qsearch entry written for one of the warm-up positions
    # ever got misread as a hit for Kiwipete's own nodes (the practical
    # stand-in for an unreachable real 2^20-slot collision), the dirty run
    # would diverge from the cold one.
    cold = fs18.FastEngine18()
    cold_result = cold.search(from_fen(kiwipete), 5000.0, 8000.0, max_depth=4)
    dirty = dirty_engine()
    dirty_result = dirty.search(from_fen(kiwipete), 5000.0, 8000.0, max_depth=4)
    ok = cold_result[:3] == dirty_result[:3]
    report(
        "cold_vs_dirty_table", ok, kiwipete,
        f"identical (move, score, depth): {cold_result[:3]}",
        f"cold={cold_result[:3]} dirty(warmed on {len(WARM_UP_FENS)} unrelated fens)={dirty_result[:3]}",
    )


# ============================================================= category 2


def category2_mate_distance() -> None:
    section("2. Mate-distance normalization across ply (score_to_tt/score_from_tt)")

    # 2a. Pure round-trip of the exact functions the qsearch store/probe use.
    # "D" is the mate distance in plies FROM THE NODE ITSELF, which must be
    # invariant no matter which ply the node is stored at or later read
    # back at -- this is precisely the "cached at one ply, re-read at a
    # different ply" scenario the qsearch TT introduces for the first time.
    for D, store_ply, read_ply, sign, label in [
        (1, 3, 3, +1, "mate_for_stm_same_ply"),
        (2, 5, 1, +1, "mate_for_stm_shallower_read"),
        (4, 10, 0, +1, "mate_for_stm_root_read"),
        (1, 3, 3, -1, "getting_mated_same_ply"),
        (2, 5, 1, -1, "getting_mated_shallower_read"),
        (4, 10, 0, -1, "getting_mated_root_read"),
        (3, 0, 12, +1, "mate_for_stm_deeper_read"),
    ]:
        raw_at_store = sign * (MATE_SCORE - (store_ply + D))
        stored = int(fs18.score_to_tt(raw_at_store, store_ply))
        expected_stored = sign * (MATE_SCORE - D)
        back = int(fs18.score_from_tt(stored, read_ply))
        expected_back = sign * (MATE_SCORE - (read_ply + D))
        ok = stored == expected_stored and back == expected_back
        report(
            f"score_roundtrip_{label}", ok, "-",
            f"stored={expected_stored} read_back={expected_back} (mate-in-{D} from node)",
            f"stored={stored} read_back={back}",
            note=f"store_ply={store_ply} read_ply={read_ply}",
        )

    # 2b. Integration: real forced-mate positions, cold vs a table dirtied
    # by unrelated searches first (so any mate score that got cached at one
    # ply and misread at another would show up as a divergence), plus
    # fastsearch4 (no qsearch TT at all) run alongside as a sanity baseline.
    fs4.warm_up()
    cases = [
        ("mate_stm", "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "+", "back rank mate in one"),
        ("mated_stm", "k7/7R/1K6/8/8/8/8/8 b - - 0 1", "-", "forced mate against side to move"),
    ]
    for tag, fen, direction, note in cases:
        cold = fs18.FastEngine18()
        cold_r = cold.search(from_fen(fen), 800.0, 1200.0, max_depth=4)
        dirty = dirty_engine()
        dirty_r = dirty.search(from_fen(fen), 800.0, 1200.0, max_depth=4)
        baseline = fs4.FastEngine4()
        base_r = baseline.search(from_fen(fen), 800.0, 1200.0, max_depth=4)

        if direction == "+":
            sign_ok = cold_r[1] > MATE_THRESHOLD and dirty_r[1] > MATE_THRESHOLD
            base_ok = base_r[1] > MATE_THRESHOLD
        else:
            sign_ok = cold_r[1] < -MATE_THRESHOLD and dirty_r[1] < -MATE_THRESHOLD
            base_ok = base_r[1] < -MATE_THRESHOLD
        identical = cold_r[:3] == dirty_r[:3]
        ok = sign_ok and identical
        report(
            f"{tag}_cold_vs_dirty", ok, fen,
            f"cold==dirty mate score, correctly signed ({direction})",
            f"cold=(move={cold_r[0]},score={cold_r[1]},depth={cold_r[2]}) "
            f"dirty=(move={dirty_r[0]},score={dirty_r[1]},depth={dirty_r[2]}) "
            f"fastsearch4=(move={base_r[0]},score={base_r[1]},depth={base_r[2]}, mate_sign_ok={base_ok})",
            note=note,
        )


# ============================================================= category 3


def category3_stalemate() -> None:
    section("3. Stalemate correctness in qsearch's not-in-check branch")
    positions = [
        (
            "corpus_two_rooks",
            "8/8/8/5k2/8/1r6/K7/1r6 w - - 0 1",
            "EXP-016: Ka2's only pseudo-tactical moves are Kxb1/Kxb3, both illegal; no quiet move either.",
        ),
        (
            "corpus_rook_knight",
            "8/8/3r4/8/7k/7n/7K/5q2 w - - 0 1",
            "second stalemate shape from the corpus: the only pseudo-capture h2h3 is illegal.",
        ),
        (
            "variant_queen_king",
            "8/8/8/8/8/1qk5/8/K7 w - - 0 1",
            "new configuration (queen+king instead of two rooks): Ka1 has zero pseudo-tactical "
            "moves at all (no adjacent enemy piece) and zero legal quiet moves either -- exercises "
            "the count==0 path into any_legal_move rather than the pseudo-capture-but-illegal path.",
        ),
    ]
    harness = QHarness()
    for tag, fen, note in positions:
        beta_inf = stand_pat_of(fen)  # any finite beta forces the >=beta shortcut branch
        for branch, beta in (("normal_path", INFINITY), ("standpat_cutoff_path", beta_inf)):
            score18, _, tt = harness.run(fen, TT(), ply=0, alpha=-INFINITY, beta=beta)
            score4 = harness.run4(fen, ply=0, alpha=-INFINITY, beta=beta)
            ok = score18 == DRAW_SCORE and score4 == DRAW_SCORE
            report(
                f"{tag}_{branch}", ok, fen,
                f"DRAW_SCORE ({DRAW_SCORE}) from both fastsearch18 and fastsearch4",
                f"fastsearch18={score18} fastsearch4={score4} (beta={beta})",
                note=note,
            )


# ============================================================= category 4


def category4_promotion() -> None:
    section("4. Promotion inside qsearch: TT best_move round-trip")

    # This position is constructed (and independently confirmed with
    # python-chess) so that fxg8=Q and fxg8=R deliver immediate checkmate,
    # while fxg8=B and fxg8=N stalemate Black instead. order_qmoves tries
    # the queen promotion first, it wins outright, and the move loop stops
    # there -- so the TT's best_move field ends up holding a *promotion*
    # move, which is exactly what this category needs to check round-trips
    # without corruption.
    fen = "5Krk/R4P2/8/8/8/8/8/8 w - - 0 1"
    tt = TT()
    harness = QHarness()
    score, value, tt = harness.run(fen, tt, ply=0, alpha=-INFINITY, beta=INFINITY)
    index = int(value & fs18.TT_MASK)
    stored_key = int(tt.tt_key[index])
    stored_move = int(tt.tt_move[index])
    from_sq, to_sq, piece, captured, promotion, is_ep, is_castle, _ = decode(stored_move)
    uci = move_to_uci(stored_move) if stored_move else "<none>"

    key_ok = stored_key == int(value)
    mate_ok = score > MATE_THRESHOLD
    move_ok = (
        promotion != NO_PIECE
        and uci == "f7g8q"
        and to_sq == 62  # g8
        and from_sq == 53  # f7
        and not is_ep
    )
    ok = key_ok and mate_ok and move_ok
    report(
        "promotion_mate_tt_roundtrip", ok, fen,
        "score is mate-for-white, TT slot's key matches this position, best_move decodes to f7g8q",
        f"score={score} key_match={key_ok} decoded_move={uci} "
        f"(from={from_sq} to={to_sq} promotion_piece={promotion} ep={is_ep} castle={is_castle})",
    )

    # Same position, but confirm ALL FOUR promotion encodings the search
    # actually generates (not hand-built ones) decode cleanly -- this is
    # the direct "does the encoding itself corrupt" check, independent of
    # which one the move loop happens to keep as best_move.
    bb, occ, mail, st = from_fen(fen)
    buf = np.zeros(MAX_MOVES, dtype=np.uint32)
    count = generate_pseudo_tactical(bb, occ, mail, st, buf)
    promo_moves = []
    for i in range(count):
        move = int(buf[i])
        f, t, pc, cap, promo, ep, castle, dbl = decode(move)
        if f == 53 and t == 62 and promo != NO_PIECE:  # f7 -> g8
            promo_moves.append((promo, move_to_uci(move)))
    letters = sorted(u[-1] for _, u in promo_moves)
    ok = len(promo_moves) == 4 and letters == ["b", "n", "q", "r"] and len(set(promo_moves)) == 4
    report(
        "promotion_all_four_encodings_distinct", ok, fen,
        "generate_pseudo_tactical yields 4 distinct f7g8 promotions: q, r, b, n",
        f"found={sorted(u for _, u in promo_moves)}",
    )


# ============================================================= category 5


def category5_en_passant() -> None:
    section("5. En-passant inside qsearch: canonical (legal-only) hash treatment")

    legal_ep_fen = "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"
    # White Ka5, Pd5, Black pc5 (just double-pushed, ep square c6), Rh5.
    # dxc6 e.p. is pseudo-legal but would vacate BOTH c5 and d5 on rank 5,
    # opening the whole rank from Ka5 straight to Rh5 -- confirmed against
    # python-chess (d5c6 is NOT in board.legal_moves here, board.is_check()
    # is False before the move). This is the genuine pin case; an earlier
    # draft of this file mistakenly used the corpus's own EP fen
    # (`8/8/8/2k5/2pP4/8/B7/4K3 b - d3 0 3`) for this role, but python-chess
    # says that EP move IS legal there (it captures the pawn giving check,
    # rather than exposing anything) -- see the check-resolving-EP case
    # below instead.
    pinned_ep_fen = "8/6k1/8/K1pP3r/8/8/8/8 w - c6 0 3"
    # This one is NOT pinned: black's checking pawn on d4 is captured en
    # passant, which is itself the check-resolving move -- so this EP
    # capture must survive both `in_check` filtering AND the fact that the
    # side to move enters quiescence's *checked* branch (not the tactical
    # one canonical_ep_file's other candidate exercises).
    check_resolving_ep_fen = "8/8/8/2k5/2pP4/8/B7/4K3 b - d3 0 3"

    bb, occ, mail, st = from_fen(legal_ep_fen)
    file_legal = int(Z.canonical_ep_file(bb, occ, st))
    ok = file_legal == 5  # f-file
    report(
        "canonical_ep_file_legal", ok, legal_ep_fen,
        "canonical_ep_file == 5 (f-file): the EP capture is really available",
        f"canonical_ep_file={file_legal}",
    )

    bb2, occ2, mail2, st2 = from_fen(pinned_ep_fen)
    file_pinned = int(Z.canonical_ep_file(bb2, occ2, st2))
    ok = file_pinned == -1
    report(
        "canonical_ep_file_pinned_excluded", ok, pinned_ep_fen,
        "canonical_ep_file == -1: dxc6 e.p. is pseudo-legal but opens rank 5 from Ka5 to Rh5, so "
        "the nominal EP square must NOT affect the hash",
        f"canonical_ep_file={file_pinned}",
    )

    bb5, occ5, mail5, st5 = from_fen(check_resolving_ep_fen)
    file_check_resolving = int(Z.canonical_ep_file(bb5, occ5, st5))
    ok = file_check_resolving == 3  # d-file
    report(
        "canonical_ep_file_check_resolving", ok, check_resolving_ep_fen,
        "canonical_ep_file == 3 (d-file): cxd3 e.p. captures the very pawn giving check, so it "
        "IS available (confirmed against python-chess: black is in check from Pd4, and c4d3 is "
        "in board.legal_moves)",
        f"canonical_ep_file={file_check_resolving}",
    )

    # The hash invariant this actually protects: a pinned/non-executable EP
    # square must hash identically to the same position with no EP square
    # at all. Verified directly with full_hash, at a qsearch-relevant node.
    no_ep_fen = pinned_ep_fen.replace(" c6 ", " - ")
    bb3, occ3, mail3, st3 = from_fen(no_ep_fen)
    hash_pinned = Z.full_hash(bb2, occ2, st2, Z.PIECE_KEYS, Z.SIDE_KEY, Z.CASTLE_KEYS, Z.EP_FILE_KEYS)
    hash_no_ep = Z.full_hash(bb3, occ3, st3, Z.PIECE_KEYS, Z.SIDE_KEY, Z.CASTLE_KEYS, Z.EP_FILE_KEYS)
    ok = int(hash_pinned) == int(hash_no_ep)
    report(
        "hash_pinned_ep_equals_no_ep", ok, pinned_ep_fen,
        "full_hash(with unusable ep square) == full_hash(no ep square at all)",
        f"hash_with_ep={int(hash_pinned)} hash_without_ep={int(hash_no_ep)}",
    )

    # And the converse: a genuinely available EP right DOES change the hash
    # relative to the same position without it.
    no_ep_legal_fen = legal_ep_fen.replace(" f6 ", " - ")
    bb4, occ4, mail4, st4 = from_fen(no_ep_legal_fen)
    hash_legal = Z.full_hash(bb, occ, st, Z.PIECE_KEYS, Z.SIDE_KEY, Z.CASTLE_KEYS, Z.EP_FILE_KEYS)
    hash_no_ep_legal = Z.full_hash(bb4, occ4, st4, Z.PIECE_KEYS, Z.SIDE_KEY, Z.CASTLE_KEYS, Z.EP_FILE_KEYS)
    ok = int(hash_legal) != int(hash_no_ep_legal)
    report(
        "hash_legal_ep_differs_from_no_ep", ok, legal_ep_fen,
        "full_hash(with a real ep right) != full_hash(no ep square)",
        f"hash_with_ep={int(hash_legal)} hash_without_ep={int(hash_no_ep_legal)}",
    )

    # Now actually run both positions through qsearch (both engines) to
    # confirm the EP-adjacent node resolves without crashing and without
    # divergence between the TT and TT-less variants.
    harness = QHarness()
    for tag, fen, note in [
        ("legal_ep_in_qsearch", legal_ep_fen, "EP capture available inside qsearch's tactical move list"),
        ("pinned_ep_in_qsearch", pinned_ep_fen, "pseudo-legal-but-pinned EP must be filtered by in_check, not by the hash"),
        ("check_resolving_ep_in_qsearch", check_resolving_ep_fen,
         "EP capture is the check-resolving move -- exercises the *checked* branch's hash threading"),
    ]:
        score18, value, tt = harness.run(fen, TT(), ply=2, alpha=-INFINITY, beta=INFINITY)
        score4 = harness.run4(fen, ply=2, alpha=-INFINITY, beta=INFINITY)
        ok = abs(score18 - score4) <= 5  # both see the same tactics; TT must not change the eval
        report(
            tag, ok, fen,
            "fastsearch18 and fastsearch4 agree (within a few cp) on this qsearch node",
            f"fastsearch18={score18} fastsearch4={score4}",
            note=note,
        )


# ============================================================= category 6


def category6_fifty_move() -> None:
    section("6. Fifty-move boundary inside qsearch (TT_HALFMOVE_SAFE_LIMIT=80 vs the draw at 100)")

    # White Rd1 can simply win a hanging queen on d4: a clean, large,
    # non-mate tactical swing that quiescence must resolve fresh every time
    # unless the TT is legitimately usable AND correct.
    base = "4k3/8/8/8/3q4/8/8/3RK3 w - - {hm} 1"
    harness = QHarness()

    # 6a. halfmove=70: TT usable, gets populated.
    fen70 = base.format(hm=70)
    score70, value70, tt = harness.run(fen70, TT(), ply=0, halfmove=70)
    index = int(value70 & fs18.TT_MASK)
    wrote = int(tt.tt_key[index]) == int(value70) and int(tt.tt_depth[index]) == 0
    ok = score70 > 400 and wrote  # rook wins at least a queen for nothing
    report(
        "halfmove_70_tt_populated", ok, fen70,
        "large positive score (won the queen) and a qsearch entry (depth 0) actually written",
        f"score={score70} tt_entry_written={wrote}",
    )

    # 6b. halfmove=85: TT_HALFMOVE_SAFE_LIMIT (80) means this node must ALSO
    # be reachable at a *different* halfmove clock than what's cached at the
    # same hash. Since the hash does not encode the halfmove clock (only
    # piece placement / side / castling / ep-file do), this is exactly the
    # scenario the safe-limit exists to guard: prove a run that CAN read the
    # dirty table (same hash!) at halfmove=85 produces the same answer as a
    # run that never saw that table at all.
    fen85 = base.format(hm=85)
    score85_warm, _, _ = harness.run(fen85, tt, ply=0, halfmove=85)  # reuses the tt from 6a
    score85_cold, _, _ = harness.run(fen85, TT(), ply=0, halfmove=85)
    ok = score85_warm == score85_cold
    report(
        "halfmove_85_tt_disabled_not_stale", ok, fen85,
        "identical score whether or not the table already holds an entry for this exact hash",
        f"warm(reused clock-70 table)={score85_warm} cold(fresh table)={score85_cold}",
        note="TT_HALFMOVE_SAFE_LIMIT=80 <= 85, so this node must neither read nor write the table",
    )

    # 6c. halfmove=99: still one shy of the automatic draw -- must NOT be
    # forced to DRAW_SCORE, tactics must still stand.
    fen99 = base.format(hm=99)
    score99, _, _ = harness.run(fen99, TT(), ply=0, halfmove=99)
    ok = score99 > 400
    report(
        "halfmove_99_not_auto_draw", ok, fen99,
        "still a large positive tactical score (99 < 100, not yet a forced draw)",
        f"score={score99}",
    )

    # 6d. halfmove=100: must be DRAW_SCORE outright, regardless of the free
    # queen sitting right there for the taking.
    fen100 = base.format(hm=100)
    score100_cold, _, _ = harness.run(fen100, TT(), ply=0, halfmove=100)
    score100_warm, _, _ = harness.run(fen100, tt, ply=0, halfmove=100)
    ok = score100_cold == DRAW_SCORE and score100_warm == DRAW_SCORE
    report(
        "halfmove_100_forced_draw", ok, fen100,
        f"DRAW_SCORE ({DRAW_SCORE}) regardless of the hanging queen, cold or warm table",
        f"cold={score100_cold} warm={score100_warm}",
    )

    # 6e. Adjacent finding worth checking explicitly: quiescence's IN-CHECK
    # branch has no `st[3] >= 100` check of its own anywhere (unlike the
    # not-in-check branch, which checks it twice) -- on inspection this
    # looked like it could be a gap. It is NOT: king escapes a check by
    # moving to a child node that is (almost always) NOT itself in check,
    # and that child's own not-in-check branch carries the `st[3] >= 100`
    # guard, so the draw is still caught one ply later and propagates back
    # up correctly; the one case where the checked branch legitimately
    # produces its own terminal result (legal == 0, i.e. real checkmate)
    # is exactly the case that must override the fifty-move rule. Verified
    # here rather than asserted: a check-but-not-mate position at
    # halfmove==100, entered directly into the checked branch, on both
    # engines (the mechanism is identical in fastsearch4, which this was
    # built from).
    check_not_mate_fen = "4r2k/8/8/8/8/8/8/4K3 w - - 100 60"
    score18_check = harness.run(check_not_mate_fen, TT(), ply=0, halfmove=100)[0]
    score4_check = harness.run4(check_not_mate_fen, ply=0, halfmove=100)
    ok = score18_check == DRAW_SCORE and score4_check == DRAW_SCORE
    report(
        "halfmove_100_in_check_not_mate", ok, check_not_mate_fen,
        f"DRAW_SCORE ({DRAW_SCORE}): white is in check but not mated, so the 50-move rule still "
        "applies -- caught one ply later by the reply's own not-in-check branch, not by the "
        "checked branch itself",
        f"fastsearch18={score18_check} fastsearch4={score4_check}",
    )


# =================================================================== main


def main() -> None:
    print("qsearch TT adversarial suite -- fastsearch18 vs fastsearch4\n")
    category1_tt_consistency()
    category2_mate_distance()
    category3_stalemate()
    category4_promotion()
    category5_en_passant()
    category6_fifty_move()

    print(f"\n{_total - _failures}/{_total} adversarial checks passed")
    if _failures:
        print(f"{_failures} FAILURE(S) -- see FAIL lines above.")
    raise SystemExit(1 if _failures else 0)


if __name__ == "__main__":
    main()
