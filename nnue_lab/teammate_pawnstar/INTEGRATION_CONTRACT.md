# Integration contract

What Deep Blue's live search (`deepblue/fastcore.py`, S1) would actually need
to call this lane, if a future integration decision is made. This is **not**
a change to `fastcore.py` -- nothing here has been applied to it, per the
mission's write boundary. It exists so a future integration session does not
have to re-derive this from scratch.

## What S1 already looks like (read, not modified)

`fastcore.py`'s position is four flat NumPy arrays -- `bb` (bitboards),
`occ` (occupancy), `mail` (a 64-entry mailbox, piece index per square or
`NO_PIECE=15`), `st` (side to move, castling, en passant, clocks). Piece
codes: `WP,WN,WB,WR,WQ,WK = 0..5`, `BP,BN,BB_,BR,BQ,BK = 6..11`. This is
**exactly** the convention `nnue_lab/features.py`'s existing `encode_mailbox()`
already assumes (`mail == 5` for the white king, `mail == 11` for black) --
this lane's own `features.py` uses the same colour/piece-type split
(`colour = piece // 6`, `piece_type = piece % 6`) so translating between the
two mailbox conventions is a one-line reindex, not a redesign.

`make_move(bb, occ, mail, st, move, undo, ply)` and the matching
`unmake_move(...)` are free Numba functions that mutate `mail` in place; the
packed `move` int decodes (via `decode()`) to `(from_square, to_square,
piece, captured, promotion, is_ep, is_castle, is_double)`. Every field this
lane's `incremental.py` needs to classify a transition (mover, captured
piece, promotion, en passant, castling) is already sitting in that decoded
tuple -- no additional bookkeeping would be required on Deep Blue's side.

## The gap between the prototype here and a real integration

This lane's `incremental.py` (`NNUEState`) takes two full `chess.Board`
snapshots (before/after) and diffs them by comparing piece sets. That is the
right shape for a *standalone, testable* prototype (it let the 100k-transition
gate in `DIFFERENTIAL_RESULTS.md` be written and checked without touching
`fastcore.py` at all), but a real integration would **not** want to
construct two `chess.Board` objects per node -- it already has the exact
delta for free from `decode(move)`.

## The recommended integration shape (not implemented here)

```python
# Called immediately after fastcore.make_move(bb, occ, mail, st, move, undo, ply):
nnue_update_from_move(nnue_state, mail_before_move_info, move, side_before)

# Called immediately after fastcore.unmake_move(...):
nnue_update_from_move(nnue_state, mail_after_move_info, inverse_of(move), side_before)
```

Concretely, a `nnue_update_from_move` adapter would:

1. Decode `move` (already done by the caller in most call sites).
2. Translate Deep Blue's piece codes to this lane's `(colour, piece_type)`
   pair (`colour = piece // 6`, `piece_type = piece % 6` -- both conventions
   already agree on this split, so it is a no-op past the reindex).
3. Build the `removed`/`added` `(square, colour, piece_type)` lists directly
   from the decoded move fields (mover's from/to squares, captured piece and
   its square -- accounting for en passant's offset square exactly as
   `fastcore.make_move` itself already does -- and, on castling, the rook's
   from/to squares) instead of diffing two full boards. This is the same
   information `incremental.py`'s board-diff approach derives the hard way;
   a real integration has it for free.
4. Check both perspectives' king buckets before/after (via this lane's
   `features.king_bucket()`, unchanged) and call `numba_kernels.update_numba`
   (delta) or `numba_kernels.refresh_numba` (bucket change) accordingly --
   the same branch `incremental.py._apply()` already implements, just fed
   from decoded-move data instead of two boards.
5. For the static-evaluation call site specifically: replace it with
   `numba_kernels.tail_numba(...)` on the current accumulators. `BENCHMARKS.md`
   shows this is not the search's bottleneck at Deep Blue's current node
   rate, so no further optimisation would be needed before a first paired-game
   trial.

## Minimal new state Deep Blue's search would carry

One `NNUEState`-equivalent (or just the two raw `int16[1024]` accumulator
arrays plus the two current bucket ints) per active search line, refreshed
once at the root (`numba_kernels.refresh_numba`, both perspectives) and
updated incrementally thereafter. This mirrors exactly how Deep Blue already
carries `undo` arrays alongside `bb`/`occ`/`mail`/`st` per ply -- an NNUE
accumulator pair is the same shape of per-line mutable state.

## What this lane does NOT decide

Whether to actually make this change, and whether to use the exact int16
path (`reference_eval`-equivalent) or the faster, ~6-28 cp-approximate int8
path (`quantized.py`) in the shipped submission, is a strength/evidence
decision for a future session with real paired-game data -- not something
this lane's scope covers. Per `ARCHITECTURE.md`'s own promotion gate (used
for every `fastsearch*` candidate), any real integration should go through
the same fixed-depth bench -> paired-game -> promotion process before
becoming the champion evaluator, exactly like every other change in this
repository.
