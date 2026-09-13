# Deep Blue

Deep Blue is a Python chess engine built for the [Optiver AI Chessathon](https://aichessathon.com).
It is designed for a constrained tournament environment: a single CPU core, a strict per-move
clock, no network access during games, and a source-readable submission.

The current competition entry is [`agent.py`](agent.py). It exposes the required
`get_move(fen, time_left_ms) -> str` interface and uses the `FastEngine185` search engine.

## Engine design

The engine combines a classical, handcrafted evaluation with an aggressively pruned alpha-beta
search. The emphasis is on making reliable decisions within the clock rather than on a particular
headline search depth.

- Iterative-deepening negamax with alpha-beta and principal-variation search.
- A fixed-size transposition table using Zobrist hashing.
- Quiescence search, including check evasion and static-exchange pruning of losing captures.
- Null-move pruning, reverse futility pruning, late-move reductions, multicut and singular
  extensions, each guarded by re-search or verification where appropriate.
- Move ordering from hash moves, captures, killers and gravity-based history heuristics.
- Tapered middlegame/endgame evaluation with material, piece-square tables, pawn structure,
  passed pawns, rook files, mobility, king shelter and king-attack terms.
- Time allocation based on the remaining clock, observed increment and root-move stability.
- Opening preparation for known tournament seed positions, with legality checks before use.
- Defensive handling of repetition, terminal positions, exceptions and low-clock fallback moves.

The agent warms its Numba-compiled search functions during import so compilation does not take
time from its first move.

## Repository layout

```
agent.py                 Competition entry point and game-level safety logic
deepblue/                Board representation, search, evaluation and time manager
tests/                   Unit and regression tests
harness/                 Local game runner and packaging tools
baselines/               Simple reference opponents
tools/                   Profiling, regression, packaging and experiment utilities
docs/                    Notes on engine development and experiment methodology
nnue_lab/                Offline neural-evaluation research; not used by agent.py
```

`nnue_lab/` is retained as research code only. The current submission does **not** load an NNUE
model or require model weights at runtime.

## Run locally

This project uses [uv](https://docs.astral.sh/uv/) for its development environment.

```bash
uv sync
make play     # one full-clock game against the greedy baseline
make arena    # a small fast-game arena
make gate     # lint, type check and two smoke games
make zip      # build submission.zip with agent.py at the archive root
```

You can also start a local game from a specific FEN:

```bash
make play FEN="r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
```

## Testing approach

Engine changes are developed as isolated candidates, checked against a regression position set,
then tested in paired local games. A promising score over a small number of games is treated as a
lead, not proof: chess results are noisy, so candidates are only promoted once the measurement is
large enough to distinguish a real gain from variance.

Historical experiments and design decisions are documented in
[`EXPERIMENTS.md`](EXPERIMENTS.md), [`ARCHITECTURE.md`](ARCHITECTURE.md), and
[`ENGINE_GUIDE.md`](ENGINE_GUIDE.md). [`HANDOFF.md`](HANDOFF.md) is a detailed working log and may
contain machine-specific notes; it is useful for development context rather than as end-user
documentation.

## Submission contract

The competition platform imports `agent.py` and repeatedly calls:

```python
def get_move(fen: str, time_left_ms: int) -> str:
    ...  # returns a UCI move such as "e2e4" or "e7e8q"
```

The authoritative requirements are the [agent contract](https://aichessathon.com/docs/agent-contract.md)
and [competition rules](https://aichessathon.com/docs/rules.md). Always check those documents before
uploading: local packaging and harness tests are safeguards, not platform acceptance decisions.

## License

This repository is available under the [MIT License](LICENSE).
