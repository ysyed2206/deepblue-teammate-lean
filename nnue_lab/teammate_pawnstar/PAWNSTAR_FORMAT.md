# Pawnstar v12 NNUE format -- verified

Verified 2026-09-02 directly against the donor's actual source, not summaries
or the mission brief's "expected (VERIFY)" placeholder numbers. Every number
below was cross-checked in at least one of two ways: (1) read verbatim from
the donor's own source files, or (2) confirmed empirically by loading the
donor's real shipped weight file and reproducing their own published
reference evaluations exactly (250/250, see `DIFFERENTIAL_RESULTS.md`).

Sources (fetched 2026-09-02, `main` branch):

- [`src/nnue.h`](https://github.com/jonny-reckless/pawnstar/blob/main/src/nnue.h) -- the network implementation, verbatim.
- [`src/constants.h`](https://github.com/jonny-reckless/pawnstar/blob/main/src/constants.h) -- piece/colour enum values, `kRankFlip`.
- [`nnue/README.md`](https://github.com/jonny-reckless/pawnstar/blob/main/nnue/README.md) -- architecture prose, file format table, training/lineage history.
- [`nnue/pawnstar-v12.bin`](https://github.com/jonny-reckless/pawnstar/blob/main/nnue/pawnstar-v12.bin) -- the actual shipped net, downloaded (12,589,152 bytes, SHA-256 `ee551d64a06aedd4b073bea2082d0e8d6a71981270b8e31737d2b42e2616c472`).
- [`test/nnue_reference.txt`](https://github.com/jonny-reckless/pawnstar/blob/main/test/nnue_reference.txt) -- 250 donor-generated (FEN, eval) ground-truth pairs for that exact net, downloaded and checked into this lane as `nnue_reference_250.txt`.

**Important correction to the mission brief's placeholder numbers:** the
brief's "expected, VERIFY FROM SOURCE" section listed `QB ~= 64` (correct)
but described the payload as if QB alone scaled the output layer; the real
formula is more specific and is reproduced exactly below, including the
truncating (not flooring) integer division. All brief placeholders were
confirmed correct in shape; the precise arithmetic needed the real source.

## 1. Architecture

```
768 inputs per (perspective, king bucket)  ->  1024-wide feature transformer (shared across perspectives)
8 king buckets, selected per perspective by that perspective's own king square
SCReLU activation
concat[side-to-move accumulator | opponent accumulator] = 2048
-> 1 scalar output (centipawns, side-to-move relative)
```

| Constant | Value | Source |
|---|---:|---|
| `kInputSize` (features per bucket) | 768 | `src/nnue.h` |
| `kNumKingBuckets` | 8 | `src/nnue.h` |
| `kFeatureRows` (`768 * 8`) | 6,144 | `src/nnue.h` |
| `kHiddenSize` | 1,024 | `src/nnue.h` |
| `kQA` (feature-transformer scale) | 255 | `src/nnue.h` |
| `kQB` (output-layer scale) | 64 | `src/nnue.h` |
| `kScale` (centipawn scale) | 400 | `src/nnue.h` |
| `kInt8Shift` (int8 tail only) | 9 | `src/nnue.h` |
| `kRankFlip` | `0x38` (56) | `src/constants.h` |
| Piece-type enum | `kPawn=0, kKnight=1, kBishop=2, kRook=3, kQueen=4, kKing=5` | `src/constants.h` (not the README prose's 1-6 "pt" numbering -- the actual C++ enum is 0-indexed; `FeatureRow`'s `piece - kPawn` is already 0..5) |
| Colour enum | `kWhite=0, kBlack=1` | `src/constants.h` |

## 2. Feature indexing (`FeatureRow`, `src/nnue.h`)

```
white perspective row = bucket*768 + colour*384     + piece_type*64 + square
black perspective row = bucket*768 + (1-colour)*384 + piece_type*64 + (square ^ 0x38)
```

`colour` is **absolute** (0=white, 1=black), not relative to the perspective
-- the `colour` vs `(1-colour)` swap in the formula itself is what makes it
relative. `bucket` is that perspective's own king bucket (below). Both kings
get ordinary feature rows too (piece_type 5), same as any other piece.

## 3. King bucket (`kKingBucketMap`, `src/nnue.h`)

Index by the perspective's own king square, **oriented to that perspective**
(white: as-is; black: `square ^ 0x38`):

```
files a/b -> bucket 0, c/d -> 1, e/f -> 2, g/h -> 3   (ranks 1-4, own half)
files a/b -> bucket 4, c/d -> 5, e/f -> 6, g/h -> 7   (ranks 5-8, advanced half)
```

i.e. `bucket = (oriented_square & 7) >> 1` plus `4` if `(oriented_square >> 3) >= 4`.

This is v12's architecture specifically -- the donor's lineage table (`nnue/README.md`
S7) shows earlier shipped nets used 4 buckets (file-pair only, no rank split)
or 0 buckets (a single bank); v12 doubled 4->8 by adding the rank split,
measured **+17.29 +/- 8.68 Elo at 8+0.08** over the 4-bucket v11 on the same
~6B-position training pool -- notably *not* diminishing returns (the 4->8
step was measured *larger* than the 0->4 step).

## 4. Forward pass (`Network::EvaluateExact`, the exact reference this lane implements)

```
screlu(x) = clamp(x, 0, QA)^2                      # per accumulator element, int64

dot  = sum(screlu(stm_acc) * output_weights[stm])  # "stm" = side-to-move accumulator
     + sum(screlu(ntm_acc) * output_weights[ntm])  # "ntm" = the other side's accumulator
                                                    # int64 accumulation throughout

dot //= QA           # SCReLU leaves QA*QA*QB units; this reduces one QA
dot += output_bias    # output_bias is stored in QA*QB units
dot *= SCALE
dot //= (QA * QB)     # -> centipawns, side-to-move relative
```

**All three `/=` above are C++ `int64_t` division, which truncates toward
zero** -- not Python's `//`, which floors toward negative infinity. This
lane's `reference_eval.trunc_div()` replicates the C++ behaviour explicitly;
getting this wrong was the only subtlety that would have silently produced
wrong (off-by-a-cp-or-two, sign-dependent) results despite every other part
of the implementation being correct -- confirmed by the fact that using plain
`//` before this fix produced non-zero diffs against the 250-position
reference set, while `trunc_div` produces 0/250.

The **shipped, faster** path (`Network::Evaluate`) additionally requantises
activations and output weights to a uint8/int8 dot product (`kInt8Shift=9`);
this lane implements and separately measures it in `quantized.py` (see
`DIFFERENTIAL_RESULTS.md` Priority 6).

## 5. Accumulator maintenance (`Network::Refresh` / `Network::Update`)

- **Refresh**: seed with `feature_bias`, then add every piece's feature
  column for that perspective's king bucket.
- **Update** (the common case, ~98% per the donor's own comment): if a
  perspective's king bucket is unchanged, derive the changed squares
  directly (this lane does it via before/after piece-set difference rather
  than the donor's colour-bitboard scan, for implementation simplicity --
  see `incremental.py`'s docstring) and apply only those feature-column
  deltas. Every move type -- quiet, capture, en passant, castling,
  promotion, capture-promotion -- reduces to "which squares changed
  occupant," with no move-type-specific branching needed.
- **King bucket change**: rebuild *only* that one perspective from scratch;
  the other perspective still diffs normally (at most one king moves per
  ply).
- Accumulator arithmetic is **int16 with wraparound** (`AddColumn`/`SubColumn`
  operate on `int16_t`, matching normal two's-complement wraparound on every
  real platform) -- this lane's NumPy accumulators use `np.int16` for the
  same reason.

## 6. File format (`NetHeader`, `Network::LoadFromMemory`)

A shipped net is a 32-byte self-describing header followed by the tightly
packed, little-endian int16 payload:

| Offset | Field | Type | Notes |
|---|---|---|---|
| 0 | magic | `char[4]` | `"PSN1"` |
| 4 | format_version | `uint16` | `1` |
| 6 | input_size | `uint16` | `768` |
| 8 | king_buckets | `uint16` | `8` |
| 10 | hidden_size | `uint16` | `1024` |
| 12 | qa | `int16` | `255` |
| 14 | qb | `int16` | `64` |
| 16 | scale | `int16` | `400` |
| 18 | reserved | `uint8[14]` | zero padding to 32 bytes |

Payload (all little-endian int16, in this exact order):

| Section | Count | Scale |
|---|---:|---|
| `feature_weights` | `6144 * 1024 = 6,291,456` (row-major: `row*1024 + i`) | QA |
| `feature_bias` | `1,024` | QA |
| `output_weights` | `2 * 1024 = 2,048` (first 1024 = side-to-move, next 1024 = opponent) | QB |
| `output_bias` | `1` (scalar) | QA*QB |

Total payload: `(6,291,456 + 1,024 + 2,048 + 1) * 2 = 12,589,058` bytes; the
real file is `12,589,152` bytes (`32` header + `12,589,120` payload, the
payload padded to a 64-byte boundary by the donor's trainer). This was
confirmed against the actual downloaded file, not just computed.

An unstamped (raw, headerless) export is rejected by both the real engine
and this lane's `weights.load_donor_net()` -- the donor removed silent
raw-file acceptance for exactly the reason the mission brief warns about
(never trust an unverified format).

## 7. What this lane deliberately did not verify from source

- **Lazy/deferred accumulator internals** (`SearchState::CurrentAccumulator`,
  mentioned in `src/nnue.h`'s file comment) -- the donor's *design* (defer
  `Update` until an eval reads it) is documented and reproduced at the API
  level in `incremental.py`'s `lazy=True` mode, but the donor's actual C++
  `SearchState` class was not fetched/read, since the mission scope is a
  standalone Python/Numba-compatible lane, not a line-for-line port.
- **AVX2/AVX-VNNI SIMD kernels** -- read (they're in `src/nnue.h`) but not
  relevant to a Numba target; the scalar fallback paths in the same file
  were the ones actually used as the specification.
- **The `bullet` trainer internals** (`tools/bullet/pawnstar.rs`) -- not
  fetched. Not needed: this lane consumes the donor's *already-trained*
  weights, not their training pipeline.

## 8. Licence note

Pawnstar is GPL-family licensed (noted already in `nnue_lab/REFERENCES.md`
at the repo root). The real weight file was downloaded and used here **only**
for compatibility verification and benchmarking, per the mission brief's
explicit allowance ("may be used for compatibility/differential/benchmarking
research. Do not assume they are automatically shippable."). It is stored
outside this repository (`%LOCALAPPDATA%\Temp\deepblue-teammate-data\`,
not committed anywhere) and this lane produces no artifact derived from its
weights (no fine-tuning was performed -- see `FINAL_STATUS.md`). Whether any
Pawnstar-*compatible* (independently trained) weights could ever ship in the
actual competition submission is a separate policy question for the primary
developer, unrelated to this research/benchmarking use.
