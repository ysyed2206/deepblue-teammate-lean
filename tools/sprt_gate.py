"""Parallel, depth-honest promotion gate with sequential (SPRT) stopping.

WHY THIS EXISTS (2026-09-06). Every paired result this project has ever
recorded came from `paired_fast_variants*.py`, whose default is 60 ms/move.
At 60 ms this engine completes depth 4. Every search technique with a depth
threshold therefore never fired in any test that rejected it:

    null move pruning      depth >= 2-3   rejected at 40.6%
    LMR (5-6 attempts)     depth-gated    rejected
    LMP / razoring         depth-gated    rejected
    SEE pruning            depth >= 10    never tried at depth
    ProbCut                depth >= 5     rejected at 48.3%
    singular extensions    depth >= 8-10  rejected at 48.1%

Non-gated changes (check extension, mate distance pruning, eval terms) did at
least execute at depth 4, so their SIGN may survive, but their magnitudes were
measured at an operating point 25-50x faster than real play (competition games
average 1.7-2.9 s/move) and should not be trusted as calibrated.

THREE FIXES, all of which have to hold at once or the evidence is junk again:

1.  DEPTH HONESTY. Default mode is FIXED DEPTH, not fixed time. A depth-gated
    technique cannot be evaluated at a depth where it never triggers. Fixed
    depth also removes the clock entirely, which kills the "sign flip across
    replays" jitter (documented as AC-001) that made every previous result
    swing 20-30 points between identical runs. Pruning techniques whose whole
    purpose is to buy depth with saved time DO need a time-based confirmation
    -- `--mode time` exists for exactly that, and should be run only after a
    fixed-depth result says the change is not actively harmful.

2.  THROUGHPUT FROM CORES, NOT FROM A FAST CLOCK. The 60 ms default existed
    because one game at a time at realistic depth is unbearably slow. This
    runs games across a process pool, which is what makes honest depth
    affordable. Each worker pays the Numba warm-up once.

3.  SEQUENTIAL STOPPING. Fixed-N testing at n=240 cannot resolve a true
    +10-20 Elo effect -- that is why so many results clustered near 50% and
    were called "neutral". SPRT stops as soon as the evidence is decisive in
    either direction, spending games only where they buy information.

Openings include the real tournament seed positions (the competition starts
every game from a [SetUp] FEN 6-8 moves deep, and those seeds are drawn from
a small reused pool), so the gate measures strength in the kind of position
we are actually assigned rather than only from move 1. The opening book lives
in agent.py, not in the fastsearch modules, so it cannot confound this.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepblue.fastcore import from_fen  # noqa: E402
from deepblue import time_manager as _tm  # noqa: E402


def stockfish_budget(time_left_ms, increment_ms, moves_played):
    """The optimum/maximum budget shape from the DEEPBLUE1 handoff's timeman.py.

    Structurally different from our allocator, not merely differently tuned:

    * `effective` COUNTS FUTURE INCREMENT INCOME. Over the next `mtg` moves the
      engine will earn increment*(mtg-1), so that income is added to what it is
      allowed to plan around. Our allocator only ever counts the increment for
      the single move in front of it -- which in a long game badly understates
      the clock, since at 600 plies a side earns 150s of increment against a
      120s starting clock.
    * `mtg` is FLAT at 50 until move 40 and only then decays. Ours decays from
      move one and hits its floor by move ~32.
    * `maximum` reaches 1.3 + 0.11*mtg times optimum -- up to 6.8x -- where our
      hard limit is a fixed 1.4x.
    """
    overhead = _tm.PROTOCOL_OVERHEAD_MS
    mtg = 50 if moves_played < 40 else max(10, 60 - moves_played)
    effective = max(1.0, time_left_ms + increment_ms * (mtg - 1) - overhead * (2 + mtg))
    opt_scale = min(0.9 / mtg, 0.85 * time_left_ms / effective)
    max_scale = 1.3 + 0.11 * mtg
    optimum = max(1.0, opt_scale * effective)
    maximum = max(optimum, min(0.8 * time_left_ms - overhead, max_scale * optimum))
    return optimum, max(maximum, optimum)


def allocate_tuned(time_left_ms, increment_ms, moves_played, base_moves, min_moves,
                   increment_fraction=None, formula="current"):
    """time_manager.allocate() with the moves-remaining estimate overridable.

    time_manager.py is shared by every engine variant, so a time-management
    change cannot be A/B tested by building a new fastsearchNN -- both sides
    would use the same allocator. Instead the SAME engine plays both sides
    with DIFFERENT clock tunings, which isolates the tuning exactly.
    Mirrors time_manager.allocate() step for step; only base_moves,
    min_moves and increment_fraction are parameterised.

    increment_fraction is the late-game lever specifically: the reserve term
    (usable / moves_remaining) dominates the opening budget, but as the clock
    drains the increment term becomes 27-36% of the whole budget by move 55+.
    Raising it therefore adds time almost entirely where the engine is thin,
    unlike base_moves, which is zero-sum across the game and tested neutral
    twice.
    """
    if formula == "ethereal":
        # Ethereal's sudden-death allocation (src/timeman.c), which is the
        # branch matching this competition's 120s + 0.5s/move:
        #     ideal = 2.50 * ((time - overhead) + 25*inc) / 50
        #     max   = 10.00 * ((time - overhead) + 25*inc) / 50
        # Equivalent to (time + 25*inc) / 20 -- the Chess Programming Wiki's
        # "base/20 + inc/2" rule of thumb, which that page notes is "very
        # competitive with advanced time management schemes".
        #
        # Notably this is MORE generous early than ours, not less: 6.6s on
        # move 1 against our 4.66s. Every previous attempt here assumed the
        # engine was over-spending early and tried to shift time later; this
        # goes the other way.
        #
        # max_usage is 4x ideal (26.5s at move 1). Our hard limit is a timer
        # that aborts rather than a condition polled in-search, so it is kept
        # at our own HARD_MULTIPLIER, as with the stockfish variants.
        if time_left_ms < _tm.PANIC_THRESHOLD_MS:
            budget = min(_tm.PANIC_BUDGET_MS,
                         (float(time_left_ms) - _tm.PROTOCOL_OVERHEAD_MS) * 0.5)
            return max(0.0, budget), max(0.0, budget)
        usable = float(time_left_ms) - _tm.PROTOCOL_OVERHEAD_MS
        ideal = 2.50 * (usable + 25.0 * float(increment_ms)) / 50.0
        ceiling = usable * _tm.ABSOLUTE_CLOCK_FRACTION
        ideal = min(ideal, ceiling)
        hard = min(ideal * _tm.HARD_MULTIPLIER, usable * _tm.HARD_CLOCK_FRACTION)
        return ideal, min(max(hard, ideal), ceiling)

    if formula == "stockfish_sd":
        # Stockfish's SUDDEN-DEATH branch, which is the correct one for this
        # competition: 120s + 0.5s/move with no move requirement is sudden
        # death with increment, NOT a moves-in-time control. The DEEPBLUE1
        # handoff implemented the movestogo!=0 branch instead, which has no
        # ply term at all and is simply the wrong formula family for our clock.
        #
        #   optConstant = min(0.0029869 + 0.00033554*log10(sec), 0.004905)
        #   optScale    = min(0.012112 + (ply + 3.22713)**0.46866 * optConstant,
        #                     0.19404 * time / timeLeft)
        #
        # The (ply)**0.469 term is the whole point: the scale grows from ~1.73
        # at move 0 to ~6.9 by ply 60, so the budget rises through the game
        # rather than decaying as ours does.
        #
        # Stockfish's own maxScale reaches 6.873x optimum, but there maximumTime
        # is polled inside the search; here hard_ms is a timer that ABORTS, so
        # that ceiling is kept at our own HARD_MULTIPLIER.
        if time_left_ms < _tm.PANIC_THRESHOLD_MS:
            budget = min(_tm.PANIC_BUDGET_MS,
                         (float(time_left_ms) - _tm.PROTOCOL_OVERHEAD_MS) * 0.5)
            return max(0.0, budget), max(0.0, budget)
        overhead = _tm.PROTOCOL_OVERHEAD_MS
        mtg = 50
        ply = 2.0 * moves_played
        time_left = max(1.0, float(time_left_ms) + float(increment_ms) * (mtg - 1)
                        - overhead * (2 + mtg))
        log_time = math.log10(max(1.0, float(time_left_ms) / 1000.0))
        opt_constant = min(0.0029869 + 0.00033554 * log_time, 0.004905)
        opt_scale = min(0.012112 + pow(ply + 3.22713, 0.46866) * opt_constant,
                        0.19404 * float(time_left_ms) / time_left)
        optimum = max(1.0, opt_scale * time_left)
        usable = float(time_left_ms) - overhead
        ceiling = usable * _tm.ABSOLUTE_CLOCK_FRACTION
        optimum = min(optimum, ceiling)
        hard = min(optimum * _tm.HARD_MULTIPLIER, usable * _tm.HARD_CLOCK_FRACTION)
        return optimum, min(max(hard, optimum), ceiling)

    if formula == "stockfish_capped":
        # The Stockfish BUDGET CURVE with our own hard-limit discipline.
        #
        # The full formula produces maximum up to 6.8x optimum. In Stockfish
        # that maximum is polled inside the search as a stopping condition; in
        # this engine hard_ms is a threading.Timer that simply ABORTS, so a
        # 6.8x hard limit means an iteration starting near the soft limit can
        # run almost seven times over budget before anything halts it. That is
        # an integration mismatch, not necessarily a bad curve -- so this
        # variant keeps the mtg schedule and the future-increment accounting
        # (which together produce ~3.4s per move at move 50 against our 888ms)
        # while clamping the ceiling back to our own HARD_MULTIPLIER.
        if time_left_ms < _tm.PANIC_THRESHOLD_MS:
            budget = min(_tm.PANIC_BUDGET_MS,
                         (float(time_left_ms) - _tm.PROTOCOL_OVERHEAD_MS) * 0.5)
            return max(0.0, budget), max(0.0, budget)
        optimum, _wide = stockfish_budget(float(time_left_ms), float(increment_ms), moves_played)
        usable = float(time_left_ms) - _tm.PROTOCOL_OVERHEAD_MS
        ceiling = usable * _tm.ABSOLUTE_CLOCK_FRACTION
        optimum = min(optimum, ceiling)
        hard = min(optimum * _tm.HARD_MULTIPLIER, usable * _tm.HARD_CLOCK_FRACTION)
        return optimum, min(max(hard, optimum), ceiling)

    if formula == "stockfish":
        if time_left_ms < _tm.PANIC_THRESHOLD_MS:
            budget = min(_tm.PANIC_BUDGET_MS,
                         (float(time_left_ms) - _tm.PROTOCOL_OVERHEAD_MS) * 0.5)
            return max(0.0, budget), max(0.0, budget)
        return stockfish_budget(float(time_left_ms), float(increment_ms), moves_played)

    usable = float(time_left_ms) - _tm.PROTOCOL_OVERHEAD_MS
    if usable <= 0.0:
        return 0.0, 0.0
    if time_left_ms < _tm.PANIC_THRESHOLD_MS:
        budget = min(_tm.PANIC_BUDGET_MS, usable * 0.5)
        return budget, budget
    moves_remaining = max(min_moves, base_moves - moves_played // 2)
    if increment_fraction is None:
        increment_fraction = _tm.INCREMENT_FRACTION
    soft = usable / moves_remaining + increment_ms * increment_fraction
    hard = min(soft * _tm.HARD_MULTIPLIER, usable * _tm.HARD_CLOCK_FRACTION)
    ceiling = usable * _tm.ABSOLUTE_CLOCK_FRACTION
    soft = min(soft, ceiling)
    hard = min(max(hard, soft), ceiling)
    return soft, hard

# Resign adjudication thresholds. 600cp sustained over 6 consecutive plies
# (3 moves by each side) -- the cutechess default. Deliberately conservative:
# a position both engines score at 6 pawns for three moves running is decided,
# and playing it out only burns wall time.
RESIGN_SCORE = 600
RESIGN_PLIES = 6
MATE_THRESHOLD_ADJ = 29_000

PLY_CAP = 200
HUGE_MS = 1.0e9

CLASSIC_OPENINGS = [
    ("start", chess.STARTING_FEN),
    ("open ruy", "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"),
    ("sicilian", "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"),
    ("french", "rnbqkbnr/pppp1ppp/4p3/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2"),
    ("caro-kann", "rnbqkbnr/pp1ppppp/2p5/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2"),
    ("queens gambit", "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2"),
    ("kings indian", "rnbqkb1r/pppppp1p/5np1/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3"),
    ("english", "rnbqkbnr/pppp1ppp/8/4p3/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 0 2"),
    ("closed centre", "r1bqkb1r/pp1n1ppp/2p1pn2/3p4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQkq - 0 6"),
    ("symmetrical", "rnbqkbnr/pp2pppp/8/2pp4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3"),
]

BOOK_PATH = Path(__file__).resolve().parent.parent / "deepblue" / "opening_book.json"


def tournament_openings(limit: int) -> list[tuple[str, str]]:
    """Real competition seed positions, which is what we actually get dealt."""
    try:
        book = json.loads(BOOK_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - optional enrichment, never fatal
        return []
    out = []
    for key, entry in book.items():
        if entry.get("predicted"):
            continue  # only positions actually seen in real games
        rounds = entry.get("rounds_seen") or ["?"]
        out.append((f"seed r{rounds[0]}", f"{key} 0 1"))
    return out[:limit]


_WORKER: dict = {}


def _init_worker(baseline: str, candidate: str) -> None:
    """One Numba warm-up per process, then reuse for every game it plays."""
    for name in (baseline, candidate):
        mod = importlib.import_module(f"deepblue.{name}")
        cls = getattr(mod, "FastEngine" + name.removeprefix("fastsearch"))
        mod.warm_up()
        _WORKER[name] = cls


def _play(white: str, black: str, fen: str, mode: str, depth: int,
          move_ms: float, hard_mult: float, base_ms: float = 120_000.0,
          increment_ms: float = 500.0, tunings=None) -> tuple[str, str]:
    engines = {chess.WHITE: _WORKER[white](), chess.BLACK: _WORKER[black]()}
    board = chess.Board(fen)
    clocks = {chess.WHITE: base_ms, chess.BLACK: base_ms}
    played = {chess.WHITE: 0, chess.BLACK: 0}
    # Resign adjudication. Standard practice in engine testing (cutechess's
    # `-resign movecount=3 score=600`): once BOTH engines agree the position is
    # lost by a wide margin for several consecutive moves, the remaining moves
    # decide nothing and only cost wall time.
    #
    # Both sides must agree, which is what makes it safe: a single engine's
    # misevaluation cannot adjudicate a game it is not actually losing. Our own
    # games average 110 plies with the last 30-40 routinely played out in
    # completely decided positions.
    resign_streak = 0
    resign_winner = None
    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < PLY_CAP:
        engine = engines[board.turn]
        position = from_fen(board.fen())
        engine.record_game_position(position)
        if mode == "clock":
            # Real protocol: the engine is told only its remaining clock and
            # decides its own budget, exactly as in competition. This is the
            # only mode that exercises time_manager.allocate().
            mover = board.turn
            base_moves, min_moves, inc_fraction, formula = tunings[mover]
            soft, hard = allocate_tuned(int(clocks[mover]), int(increment_ms),
                                        played[mover], base_moves, min_moves,
                                        inc_fraction, formula)
            started = time.monotonic()
            result = engine.search(position, soft, hard)
            spent = (time.monotonic() - started) * 1000.0
            clocks[mover] -= spent
            if clocks[mover] <= 0.0:
                loser = "white" if mover == chess.WHITE else "black"
                return ("black" if mover == chess.WHITE else "white"), f"{loser} flagged", len(board.move_stack)
            clocks[mover] += increment_ms
            played[mover] += 1
        elif mode == "depth":
            # No clock at all: the same position always yields the same move,
            # so a rerun of this gate reproduces exactly, unlike every
            # time-based result this project has recorded.
            result = engine.search(position, HUGE_MS, HUGE_MS, max_depth=depth)
        else:
            result = engine.search(position, move_ms, move_ms * hard_mult)
        uci = result[0]
        if uci is None:
            return "problem", "no move", len(board.move_stack)
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            return "problem", f"illegal {uci} at {board.fen()}", len(board.move_stack)
        # result is (move, score, depth, nodes, elapsed); score is from the
        # side to move's point of view.
        score = result[1]
        if score is not None and abs(score) < MATE_THRESHOLD_ADJ:
            side_losing = (board.turn if score <= -RESIGN_SCORE else
                           (not board.turn) if score >= RESIGN_SCORE else None)
            if side_losing is None:
                resign_streak, resign_winner = 0, None
            else:
                winner = "black" if side_losing == chess.WHITE else "white"
                resign_streak = resign_streak + 1 if winner == resign_winner else 1
                resign_winner = winner
                if resign_streak >= RESIGN_PLIES:
                    return resign_winner, "adjudicated", len(board.move_stack)
        else:
            resign_streak, resign_winner = 0, None
        board.push(move)
    plies = len(board.move_stack)
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return "draw", "ply cap", plies
    if outcome.winner is None:
        return "draw", outcome.termination.name.lower(), plies
    return ("white" if outcome.winner == chess.WHITE else "black"), outcome.termination.name.lower(), plies


def _run_pair(job):
    """One opening, both colours -- the paired unit, kept on one worker."""
    (label, fen, baseline, candidate, mode, depth, move_ms, hard_mult,
     base_tuning, cand_tuning) = job
    results = []
    for cand_is_white in (True, False):
        white = candidate if cand_is_white else baseline
        black = baseline if cand_is_white else candidate
        tunings = {chess.WHITE: cand_tuning if cand_is_white else base_tuning,
                   chess.BLACK: base_tuning if cand_is_white else cand_tuning}
        outcome, detail, plies = _play(white, black, fen, mode, depth, move_ms, hard_mult,
                                       tunings=tunings)
        if outcome == "problem":
            results.append(("problem", f"{label}: {detail}", plies))
        elif outcome == "draw":
            results.append(("draw", f"{label} {detail}", plies))
        else:
            cand_won = (outcome == "white") == cand_is_white
            results.append(("win" if cand_won else "loss", f"{label} {detail}", plies))
    return results


def sprt_llr(wins: int, draws: int, losses: int, elo0: float, elo1: float) -> float:
    """Log-likelihood ratio under a fixed-draw-ratio trinomial model.

    Draws cancel (their probability is held at the observed rate under both
    hypotheses), so only decisive games carry evidence -- which is also why
    this resolves far faster than staring at a pooled percentage.
    """
    n = wins + draws + losses
    if n == 0 or wins + losses == 0:
        return 0.0
    draw_ratio = draws / n

    def wl(elo: float) -> tuple[float, float]:
        expected = 1.0 / (1.0 + 10.0 ** (-elo / 400.0))
        return expected - draw_ratio / 2.0, 1.0 - expected - draw_ratio / 2.0

    w0, l0 = wl(elo0)
    w1, l1 = wl(elo1)
    if min(w0, l0, w1, l1) <= 1e-9:
        return 0.0
    return wins * math.log(w1 / w0) + losses * math.log(l1 / l0)


def elo_estimate(wins: int, draws: int, losses: int) -> tuple[float, float]:
    """Point estimate and a rough 95% interval, in Elo."""
    n = wins + draws + losses
    if n == 0:
        return 0.0, 0.0
    score = (wins + draws / 2.0) / n
    if score <= 0.0 or score >= 1.0:
        return (math.copysign(9999.0, score - 0.5), 0.0)
    elo = -400.0 * math.log10(1.0 / score - 1.0)
    # standard error of the score, propagated through the logistic
    var = (wins * (1 - score) ** 2 + draws * (0.5 - score) ** 2
           + losses * (0 - score) ** 2) / max(n - 1, 1)
    se = math.sqrt(var / n)
    slope = 400.0 / (math.log(10) * max(score * (1 - score), 1e-9))
    return elo, 1.96 * se * slope


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--mode", choices=("depth", "time", "clock"), default="depth",
                    help="clock = real 120s+0.5s/move protocol via time_manager.allocate(); "
                         "the only mode that can test time-management changes")
    ap.add_argument("--depth", type=int, default=8,
                    help="fixed search depth per move in depth mode")
    ap.add_argument("--move-ms", type=float, default=1000.0,
                    help="soft per-move budget in time mode (competition averages 1700-2900)")
    ap.add_argument("--hard-mult", type=float, default=1.4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-pairs", type=int, default=400,
                    help="hard stop; SPRT normally ends far sooner")
    ap.add_argument("--elo0", type=float, default=0.0)
    ap.add_argument("--elo1", type=float, default=15.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--base-moves-baseline", type=int, default=_tm.BASE_MOVES_REMAINING)
    ap.add_argument("--increment-fraction-baseline", type=float, default=_tm.INCREMENT_FRACTION)
    ap.add_argument("--increment-fraction-candidate", type=float, default=_tm.INCREMENT_FRACTION)
    ap.add_argument("--budget-formula-baseline", choices=("current", "stockfish", "stockfish_capped", "stockfish_sd", "ethereal"), default="current")
    ap.add_argument("--budget-formula-candidate", choices=("current", "stockfish", "stockfish_capped", "stockfish_sd", "ethereal"), default="current")
    ap.add_argument("--base-moves-candidate", type=int, default=_tm.BASE_MOVES_REMAINING)
    ap.add_argument("--min-moves", type=int, default=_tm.MIN_MOVES_REMAINING)
    ap.add_argument("--fixed-games", action="store_true",
                    help="run the whole sample without SPRT early stopping, "
                         "for an unbiased Elo estimate")
    ap.add_argument("--competition-openings", action="store_true",
                    help="use the 51 real starting FENs from rounds 1-60 "
                         "instead of classic openings (in-distribution)")
    ap.add_argument("--seed-openings", type=int, default=14,
                    help="how many real tournament seed positions to include")
    args = ap.parse_args()

    if args.competition_openings:
        # IN-DISTRIBUTION TESTING. Every game in this competition starts from a
        # supplied FEN at move 5-9 -- pieces developed, often already castled.
        # CLASSIC_OPENINGS start at move 2-6, so every result this project has
        # ever produced was measured on positions the engine never actually
        # plays from, with ten extra opening moves per game and an opening book
        # that applies in testing but not in competition.
        #
        # These 51 positions are the real starting FENs from rounds 1-60.
        import json
        data = json.loads(
            (Path(__file__).resolve().parent.parent /
             "tests" / "competition_openings.json").read_text(encoding="utf-8"))
        openings = [(label, fen) for label, fen in data]
    else:
        openings = CLASSIC_OPENINGS + tournament_openings(args.seed_openings)
    lower = math.log(args.beta / (1.0 - args.alpha))
    upper = math.log((1.0 - args.beta) / args.alpha)

    setting = ("real game clock 120s + 0.5s/move" if args.mode == "clock"
               else f"fixed depth {args.depth}" if args.mode == "depth"
               else f"{args.move_ms:.0f} ms/move (hard x{args.hard_mult})")
    print(f"{args.candidate} vs {args.baseline}", flush=True)
    print(f"mode: {setting}   openings: {len(openings)}   workers: {args.workers}", flush=True)
    print(f"SPRT H0 ={args.elo0:+.0f} Elo  H1 ={args.elo1:+.0f} Elo   "
          f"bounds [{lower:+.2f}, {upper:+.2f}]", flush=True)

    jobs = []
    for index in range(args.max_pairs):
        label, fen = openings[index % len(openings)]
        jobs.append((label, fen, args.baseline, args.candidate,
                     args.mode, args.depth, args.move_ms, args.hard_mult,
                     (args.base_moves_baseline, args.min_moves,
                      args.increment_fraction_baseline, args.budget_formula_baseline),
                     (args.base_moves_candidate, args.min_moves,
                      args.increment_fraction_candidate, args.budget_formula_candidate)))

    wins = draws = losses = problems = 0
    ply_counts: list[int] = []
    started = time.monotonic()
    verdict = "inconclusive (hit max pairs)"

    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init_worker,
                  initargs=(args.baseline, args.candidate)) as pool:
        for pair_results in pool.imap_unordered(_run_pair, jobs):
            for outcome, detail, plies in pair_results:
                ply_counts.append(plies)
                if outcome == "win":
                    wins += 1
                elif outcome == "loss":
                    losses += 1
                elif outcome == "draw":
                    draws += 1
                else:
                    problems += 1
                    print(f"  PROBLEM: {detail}", flush=True)

            games = wins + draws + losses
            llr = sprt_llr(wins, draws, losses, args.elo0, args.elo1)
            if games and games % 20 == 0:
                score = (wins + draws / 2.0) / games
                elo, margin = elo_estimate(wins, draws, losses)
                avg_plies = sum(ply_counts) / len(ply_counts)
                over_60 = sum(1 for p in ply_counts if p >= 60)
                print(f"  {games:4d} games  +{wins} ={draws} -{losses}  "
                      f"{score*100:5.1f}%  {elo:+6.1f} +/-{margin:.0f} Elo  "
                      f"LLR {llr:+.2f}  ({time.monotonic()-started:.0f}s)  "
                      f"avg {avg_plies:.0f} plies, {over_60}/{len(ply_counts)} reached ply 60", flush=True)

            # --fixed-games runs the full sample and never stops early.
            #
            # WHY THIS MATTERS. An SPRT stops the instant the LLR crosses a
            # bound -- that is, at the moment the evidence looks most extreme --
            # so the point estimate it reports is inflated by construction.
            # Measured here on three separate candidates, each of which
            # accepted on a run that stopped early and then evaporated on a run
            # that did not:
            #
            #     129   +44  ->  -57
            #     139   +59  ->  -30
            #     150   +50  ->   0.0
            #
            # Use the SPRT when the question is "is this better than X?" and a
            # fixed sample when the question is "how much is this worth?".
            if args.fixed_games:
                continue
            if llr >= upper:
                verdict = f"ACCEPT H1: candidate is better than {args.elo0:+.0f} Elo"
                pool.terminate()
                break
            if llr <= lower:
                verdict = f"REJECT: candidate is not better than {args.elo0:+.0f} Elo"
                pool.terminate()
                break

    games = wins + draws + losses
    score = (wins + draws / 2.0) / games if games else 0.0
    elo, margin = elo_estimate(wins, draws, losses)
    print(f"\n=== {args.candidate} vs {args.baseline} ({setting}) ===")
    print(f"{games} games  +{wins} ={draws} -{losses}  score {score*100:.1f}%")
    print(f"Elo {elo:+.1f} +/- {margin:.0f}   LLR {sprt_llr(wins, draws, losses, args.elo0, args.elo1):+.2f}")
    if ply_counts:
        over_60 = sum(1 for p in ply_counts if p >= 60)
        print(f"game length: avg {sum(ply_counts)/len(ply_counts):.0f} plies, "
              f"min {min(ply_counts)}, max {max(ply_counts)}, "
              f"{over_60}/{len(ply_counts)} reached ply 60 "
              f"(the range where the two tunings under test actually diverge)")
    if problems:
        print(f"PROBLEMS: {problems} (illegal move or no move -- investigate before trusting)")
    print(f"VERDICT: {verdict}")
    print(f"elapsed {time.monotonic()-started:.0f}s")


if __name__ == "__main__":
    main()
