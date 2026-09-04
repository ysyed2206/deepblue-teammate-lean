# NNUE v0 research references

Research snapshot: 2026-09-01. These projects were studied for concepts only. No third-party
implementation file, source fragment, trained network, or binary is copied into this lab. The
implementation here is an independent Python/Numba design with a deliberately smaller scope.

The named reference checkouts were not present under this workspace, and the sandbox did not
permit a reliable search of the whole user profile. Consequently the public repositories below,
on their current default branches, are the sources actually inspected.

## Production engine references

- **Coda** — [repository](https://github.com/adamtwiss/coda),
  [`src/nnue.rs`](https://github.com/adamtwiss/coda/blob/main/src/nnue.rs),
  [`src/threat_accum.rs`](https://github.com/adamtwiss/coda/blob/main/src/threat_accum.rs),
  [README](https://github.com/adamtwiss/coda/blob/main/README.md), and
  [CLAUDE.md](https://github.com/adamtwiss/coda/blob/main/CLAUDE.md). The useful concepts are
  separate piece-square and threat state, lazy materialisation from an accurate ancestor, Finny
  caching, pairwise activations, material-conditioned heads, and architecture-specific vector
  kernels. Its wide, threat-aware network and enormous training corpus make it a systems reference,
  not a laptop-feasible v0 target.
- **PlentyChess** — [repository](https://github.com/Yoshie2000/PlentyChess),
  [`src/nnue.h`](https://github.com/Yoshie2000/PlentyChess/blob/main/src/nnue.h),
  [`src/nnue.cpp`](https://github.com/Yoshie2000/PlentyChess/blob/main/src/nnue.cpp),
  [`src/threat-inputs.h`](https://github.com/Yoshie2000/PlentyChess/blob/main/src/threat-inputs.h),
  and [`src/threat-inputs.cpp`](https://github.com/Yoshie2000/PlentyChess/blob/main/src/threat-inputs.cpp).
  It combines king-bucketed piece-square inputs with a much larger attack/interaction vocabulary,
  pawn-pair inputs, a wide accumulator, multiple output heads, dirty-feature tracking, and SIMD.
  This reinforces that threat inputs require both much more data and a tuned low-level kernel.
- **Reckless** — [repository](https://github.com/codedeliveryservice/Reckless) and
  [`src/nnue.rs`](https://github.com/codedeliveryservice/Reckless/blob/main/src/nnue.rs). Relevant
  ideas are separate accumulator stacks, replay from a known-correct ancestor, refresh on a change
  in king conditioning, pairwise reduction, output buckets, and per-platform scalar/vector paths.
- **Obsidian** — [canonical repository](https://github.com/gab8192/Obsidian),
  [`src/nnue.h`](https://github.com/gab8192/Obsidian/blob/main/src/nnue.h), and
  [`src/nnue.cpp`](https://github.com/gab8192/Obsidian/blob/main/src/nnue.cpp). Its very wide feature
  transformer, pairwise front end, material heads, dirty-piece updates, and Finny entries are useful
  mature-engine patterns. Its transformer alone is far larger than the entire v0 budget here.
- **Pawnstar** — [NNUE documentation](https://github.com/jonny-reckless/pawnstar/blob/main/nnue/README.md)
  and [`src/nnue.h`](https://github.com/jonny-reckless/pawnstar/blob/main/src/nnue.h). This is the
  closest architectural precedent: eight king buckets over a 768-entry relative piece-square
  vocabulary, one shared transformer, two oriented accumulators, squared clipped activations,
  side-to-move-first concatenation, and a scalar output. It also demonstrates exact int16 inference
  before later output approximations. We retain the broad architecture but independently define the
  encoder, file format, rounding, incremental API, and tests.

Repository licences differ (GPL-family for Coda, PlentyChess, Pawnstar, Obsidian and the Stockfish
trainer; AGPL-3.0 for Reckless). This lab avoids licence ambiguity by using the publications as
research references and writing its own implementation.

## Training references

- **Stockfish nnue-pytorch** — [repository](https://github.com/official-stockfish/nnue-pytorch)
  and [technical NNUE notes](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md).
  Principles adopted: sparse binary features with low feature churn; most capacity in the sparse
  first layer; two perspectives; shallow, integer-friendly inference; probability-space fitting of
  centipawn labels; and explicit float-versus-quantised validation. We do not attempt to reproduce
  Stockfish's full architecture, data blend, trainer, or network.
- **Tan and Watkinson Medina, “Study of the Proper NNUE Dataset”** —
  [arXiv paper](https://arxiv.org/abs/2412.17948) and
  [HTML](https://arxiv.org/html/2412.17948). The paper's stability selection is stronger than this
  lab's inexpensive proxy: it compares static, quiescence, and deeper search values. Its empirical
  work is on Xiangqi, so it motivates a controlled Western-chess quiet/general comparison but does
  not establish that quiet-only chess training is superior. Our filter is explicitly labelled
  approximate and its result is measured rather than presumed.

## Data and competition sources

- **Lichess chess-position-evaluations** —
  [dataset card](https://huggingface.co/datasets/Lichess/chess-position-evaluations),
  [README](https://huggingface.co/datasets/Lichess/chess-position-evaluations/blob/main/README.md),
  and [files](https://huggingface.co/datasets/Lichess/chess-position-evaluations/tree/main/data).
  It is CC0-1.0, contains Stockfish browser analyses, and is updated over time. The lab range-read
  bounded prefixes from five evenly spaced Parquet shards (0000, 0004, 0008, 0012, and 0016), pinned
  to revision `abb8f0b1251f89295a35b5ac801cb08a873812de`, rather than downloading the roughly 42 GB
  corpus.
- **AI Chessathon canonical contract** —
  [agent contract](https://aichessathon.com/docs/agent-contract.md) and
  [rules](https://aichessathon.com/docs/rules.md), fetched on 2026-09-01. The design respects the
  participant-trained-model rule, Python-source submission rule, fixed offline package stack, one
  core, 2 GB RAM, 50 MB uncompressed submission, and no-network runtime. Stockfish is an offline
  teacher/orientation check only and is never part of a candidate artifact.

## Independent v0 design

The v0 feature index is fully specified by this repository rather than imported from a reference:

1. Maintain fixed White and Black perspectives.
2. Vertically flip squares for Black, and express piece colour as own/opponent relative to the
   perspective.
3. Bucket the perspective king by file pair (`ab`, `cd`, `ef`, `gh`) and oriented board half.
4. Form `bucket * 768 + relative_colour * 384 + piece_type * 64 + oriented_square`.
5. Sum one shared transformer row per board piece plus one bias for each perspective.
6. Apply squared clamp-to-[0,1], concatenate side-to-move perspective first, and use one scalar tail.

No threats, pawn pairs, cache, deeper tail, material head, SIMD-specific path, or search integration
is in v0. Widths are tested in the requested order and chosen on held-out evidence, not size alone.

The lab's exact quantisation is derived from its float equation. With feature scale `QA`, output
scale `QB`, and centipawn scale `S`:

- `Wq = round(QA * W)` and `bq = round(QA * b)`;
- `A = clamp(bq + sum(Wq), 0, QA)`;
- `vq = round(QB * v)` and `cq = round(QB * c)`;
- integer output is the symmetrically rounded value of
  `S * (cq * QA^2 + sum(vq * A^2)) / (QB * QA^2)`.

The numerator is accumulated in int64. This preserves the exact squared activation at v0 widths;
an approximate int8 or early-shift tail is intentionally deferred.
