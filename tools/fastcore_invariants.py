"""Permanent invariant suite for the S1 core.

Every correctness claim made about fastcore lives here so it can be re-run on
demand, with a fixed seed, rather than existing only as a number someone once
saw in a terminal. Any change to fastcore must leave this at zero failures.

Checks:
  1. is_attacked      vs python-chess, every square, both colours
  2. in_check         vs python-chess
  3. make/unmake      exact restoration of bitboards, occupancy, mailbox, state
  4. occupancy        equals the union of the piece bitboards
  5. mailbox          agrees with the piece bitboards, both directions
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepblue import fastcore as F  # noqa: E402

STRESS_FENS = [
    chess.STARTING_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "8/8/8/2k5/2pP4/8/B7/4K3 b - d3 0 3",
    "3k4/3p4/8/K1P4r/8/8/8/8 b - - 0 1",
    "8/P1P5/8/8/8/8/5p1p/4K2k w - - 0 1",
    "4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1",
    "8/5k2/8/2Pp4/2B5/1K6/8/8 w - d6 0 1",
]


def sample_positions(rng: random.Random, count: int) -> list[str]:
    """Random walks from the stress positions, biased toward captures and
    promotions so the corpus reaches thin, promotion-heavy endings."""
    fens = list(STRESS_FENS)
    while len(fens) < count:
        board = chess.Board(rng.choice(STRESS_FENS))
        for _ in range(rng.randrange(0, 90)):
            moves = list(board.legal_moves)
            if not moves:
                break
            sharp = [m for m in moves if board.is_capture(m) or m.promotion]
            board.push(rng.choice(sharp if sharp and rng.random() < 0.45 else moves))
            fens.append(board.fen())
            if len(fens) >= count:
                break
    return fens[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260831)
    arguments = parser.parse_args()

    F.warm_up()
    rng = random.Random(arguments.seed)
    fens = sample_positions(rng, arguments.positions)

    attack_checks = attack_fails = 0
    check_checks = check_fails = 0
    restore_checks = restore_fails = 0
    occupancy_fails = mailbox_fails = 0
    first_failures: list[str] = []

    def report(message: str) -> None:
        if len(first_failures) < 5:
            first_failures.append(message)

    for fen in fens:
        board = chess.Board(fen)
        bb, occ, mail, st = F.from_fen(fen)

        # 4 + 5: representation invariants
        white = np.uint64(0)
        black = np.uint64(0)
        for index in range(6):
            white |= bb[index]
        for index in range(6, 12):
            black |= bb[index]
        if white != occ[0] or black != occ[1] or (white | black) != occ[2]:
            occupancy_fails += 1
            report(f"occupancy mismatch: {fen}")
        for square in range(64):
            piece = mail[square]
            bit = (occ[2] >> np.uint64(square)) & np.uint64(1)
            if piece == F.NO_PIECE:
                if bit:
                    mailbox_fails += 1
                    report(f"mailbox says empty, occupancy says filled at {square}: {fen}")
                    break
            else:
                if not (bb[piece] >> np.uint64(square)) & np.uint64(1):
                    mailbox_fails += 1
                    report(f"mailbox/bitboard disagree at {square}: {fen}")
                    break

        # 1: attack detection, every square, both colours
        for square in range(64):
            for side, colour in ((0, chess.WHITE), (1, chess.BLACK)):
                attack_checks += 1
                if bool(F.is_attacked(bb, occ, square, side)) != board.is_attacked_by(colour, square):
                    attack_fails += 1
                    report(f"is_attacked {chess.square_name(square)} side {side}: {fen}")

        # 2: in_check
        check_checks += 1
        if bool(F.in_check(bb, occ, st, st[0])) != board.is_check():
            check_fails += 1
            report(f"in_check: {fen}")

        # 3: make/unmake exact restoration
        stack, pseudo_stack, undo = F.new_search_buffers()
        before = (bb.copy(), occ.copy(), mail.copy(), st.copy())
        count = F.generate_legal(bb, occ, mail, st, stack, pseudo_stack, undo, 0)
        for index in range(count):
            move = stack[0, index]
            F.make_move(bb, occ, mail, st, move, undo, 0)
            F.unmake_move(bb, occ, mail, st, move, undo, 0)
            restore_checks += 1
            if not (
                np.array_equal(bb, before[0])
                and np.array_equal(occ, before[1])
                and np.array_equal(mail, before[2])
                and np.array_equal(st, before[3])
            ):
                restore_fails += 1
                report(f"make/unmake corruption on {F.move_to_uci(move)}: {fen}")
                break

    print(f"seed {arguments.seed}, {len(fens):,} positions\n")
    rows = [
        ("is_attacked vs python-chess", attack_checks, attack_fails),
        ("in_check vs python-chess", check_checks, check_fails),
        ("make/unmake exact restoration", restore_checks, restore_fails),
        ("occupancy = union of bitboards", len(fens), occupancy_fails),
        ("mailbox agrees with bitboards", len(fens), mailbox_fails),
    ]
    for name, checks, fails in rows:
        status = "pass" if fails == 0 else "FAIL"
        print(f"  {status}  {name:<34} {fails:>4} failures / {checks:>9,} checks")
    for message in first_failures:
        print(f"\n  {message}")
    total = sum(fails for _, _, fails in rows)
    print(f"\ntotal failures: {total}")
    raise SystemExit(1 if total else 0)


if __name__ == "__main__":
    main()
