# Pawnstar production NNUE notes

Focused archaeology performed on 2026-09-02. These are research notes, not
copied implementation. Deep Blue's code remains unchanged.

## Early lineage worth reproducing

Pawnstar's first useful production recipe was deliberately small: shared
perspective `Chess768`, one accumulator for each colour, SCReLU, STM accumulator
first, opponent accumulator second, and one scalar output. There were no king
buckets or hidden tail.

| Generation | Architecture/data | Reported donor result |
|---|---|---|
| v1 | H=256; about 3.6M Pawnstar-HCE self-play positions | weaker than its HCE (about -67 Elo) |
| v2 | H=256; about 60.4M public PlentyChess bullet-format records | about +330 Elo at fixed depth versus its HCE |
| v3 | H=256; roughly 750M records | no clear improvement over v2; capacity-limited |
| v4 | H=512; the same roughly 60M corpus | clear improvement over v2 (about +55 fixed-depth, +71 equal-time in the donor report) |

Those Elo figures are Pawnstar's results only. They are motivation, not claims
about Deep Blue.

Relevant history includes the initial NNUE work
([6c68a89](https://github.com/jonny-reckless/pawnstar/commit/6c68a89)),
the v2 data switch
([08ef383](https://github.com/jonny-reckless/pawnstar/commit/08ef383)),
incremental accumulators
([4cde8fd](https://github.com/jonny-reckless/pawnstar/commit/4cde8fd)),
the H=512 change
([964c941](https://github.com/jonny-reckless/pawnstar/commit/964c941)),
and the v4 ship
([441a9c2](https://github.com/jonny-reckless/pawnstar/commit/441a9c2)).

## Exact early recipe

- Features: `2 colours x 6 pieces x 64 squares = 768` shared rows. The black
  perspective swaps relative colour and flips ranks with `square xor 56`.
- Network: `768 -> H` shared affine for both perspectives; SCReLU; concatenate
  `[STM, opponent]`; affine `2H -> 1`.
- Output: side-to-move relative, with evaluation scale 400 cp.
- Quantisation: feature weights and bias are int16 at `QA=255`; output weights
  are int16 at `QB=64`; output bias is int16 at `QA*QB`.
- Integer forward: clamp each accumulator to `[0, QA]`, square it, dot with
  output weights, divide the dot by one `QA`, add the output bias, then scale by
  400 and divide by `QA*QB`. Lab Q1 uses those same two signed divisions,
  truncating toward zero, and cross-checks exporter and Numba exactly.
- Trainer: AdamW, batch 16,384, starting learning rate 0.001, step decay by 0.3
  at thirds of the run, normally 40 superbatches/approximately epochs for the
  shipped early nets.
- Loss: squared error after sigmoid. Bullet's target was a 50:50 blend of game
  result and `sigmoid(score/400)`.
- Binary records: Bullet `ChessBoard` records are 32 bytes and normalize the
  position and score to the side to move. Piece nibbles encode relative colour
  plus piece type; no separate STM bit is needed after normalization.

Primary implementation references:

- [Pawnstar repository](https://github.com/jonny-reckless/pawnstar)
- [NNUE documentation](https://github.com/jonny-reckless/pawnstar/blob/main/nnue/README.md)
- [Current NNUE header](https://github.com/jonny-reckless/pawnstar/blob/main/src/nnue.h)
- [Bullet data documentation](https://github.com/jw1912/bullet/blob/main/docs/3-data.md)

## Data decision and deliberate training deviations

The historical v2/v4 files can be identified exactly in the public
[`Yoshie2000/plentychess_data_bulletformat`](https://huggingface.co/datasets/Yoshie2000/plentychess_data_bulletformat)
repository: 60,431,369 records across 1,933,803,808 bytes. However, its Hugging
Face metadata and file tree expose no licence declaration. Public download is
not enough to establish reuse rights, so these weights are **not** trained on
that corpus.

We instead use the pinned, CC0
[`Lichess/chess-position-evaluations`](https://huggingface.co/datasets/Lichess/chess-position-evaluations)
source already validated by the baseline pipeline (revision
`abb8f0b1251f89295a35b5ac801cb08a873812de`). It supplies Stockfish scores but
no game result. Therefore our objective is pure teacher probability MSE,
`MSE(sigmoid(pred/400), sigmoid(clipped_stm_cp/400))`; inventing a result blend
from the same score would add no independent information. We use CPU PyTorch,
smaller batches, and far fewer passes than Pawnstar's GPU/Bullet run. These are
resource/data-source deviations, while the H=256 versus H=512 architecture
comparison remains controlled.

## Runtime lessons retained for later integration

Pawnstar's largest practical runtime gains came from incremental accumulator
maintenance, vectorized/integer tail work, then lazy/deferred updates and
copy-make accumulator stacks. Its no-bucket generations update king moves like
ordinary piece moves; they do not refresh a king-bucket accumulator. Later
features included an evaluation cache and runtime AVX-VNNI dispatch.

The donor's int8 **output** path was accepted after testing (reported +31.8 Elo),
while an int8 feature transformer was rejected (reported -8 Elo even after a
lossless retrain). Hidden-head and output-bucket experiments were also rejected.
Accordingly this lab first validates the exact int16 Q1 path. An int8-output Q2
is optional and cannot replace Q1 evidence.

Useful later-history references include int8 output
([8b61a64](https://github.com/jonny-reckless/pawnstar/commit/8b61a64)),
the rejected int8 transformer
([172216e](https://github.com/jonny-reckless/pawnstar/commit/172216e)), and
lazy updates
([7502314](https://github.com/jonny-reckless/pawnstar/commit/7502314)).
