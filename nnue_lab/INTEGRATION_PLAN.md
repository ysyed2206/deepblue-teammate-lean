# Deferred NNUE v0 integration plan

Status: **plan only**. No step below has been applied to Deep Blue mainline.

The lab result supports integrating the H=128 quantised candidate behind an experimental switch,
then deciding from one-core search and arena evidence. The minimum later change set is:

1. Add the frozen `candidate_int16.npz` as a submission-owned model asset and add one small NNUE
   runtime module. At import, validate format version, dimensions, `QA=255`, `QB=64`, scale 400,
   dtypes, and the frozen SHA-256 before exposing the arrays.
2. Allocate one fixed `int32[2,128]` accumulator state per active search. Fully refresh it once from
   the root mailbox. Keep the two rows in fixed White/Black order; side to move only chooses output
   concatenation order.
3. Around each existing make/unmake pair, apply the already-tested participant-owned delta logic in
   place. Ordinary moves remove source/add destination; captures also remove the captured square;
   en passant uses its actual capture square; promotions replace pawn with promoted piece; castling
   moves the rook. Refresh only the moving king's perspective when its bucket changes. Unmake uses
   the exact inverse and refreshes that perspective when required.
4. Null moves do not change features or accumulators; only side-to-move ordering changes at the
   tail. Repetition, fifty-move, terminal, and mate-score logic remain outside NNUE and unchanged.
5. Replace only the static-evaluation call site with the ready-accumulator integer tail. Keep the
   accepted HCE callable behind the experiment switch for differential tests and immediate fallback.
6. Redirect Numba caches to permitted runtime cache space and warm the loader, refresh, delta, and
   tail signatures during the existing import-time budget. Force one inference thread. The model
   requires no network, external executable, or training data at runtime.

Before an arena run, port these lab gates without weakening them:

- perspective symmetry and feature-index tests;
- at least 50,000 exact incremental/full/unmake transitions on the integrated path, including every
  special move and king-bucket boundary;
- float-versus-integer reference vectors and model-header/hash rejection tests;
- root refresh and search-return restoration assertions;
- null-move, aspiration/re-search, quiescence, cutoff, and exception-path accumulator restoration.

Then benchmark the complete search, not just the tail:

1. Compare nodes/s, evaluations/s, completed depth, and timeout margin on a fixed FEN suite under the
   competition's one-core timing model.
2. Attribute the observed node-rate change separately to update, tail, and bucket-refresh traffic.
3. Run paired games from identical openings against the current champion with colours reversed.
4. Ship only if the game result and time-safety evidence compensate for any NPS loss. Otherwise keep
   HCE and retain this lab candidate for v1 work.

Likely later files affected are one new runtime module, one model asset, the active search's
make/unmake/evaluation plumbing, and import-time warm-up. Exact filenames should be chosen from the
then-current mainline because search development is concurrent. No move-generation semantics,
training code, Stockfish component, or network access is needed.
