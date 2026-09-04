Deep Blue Patch 08 — donor-derived isolated candidates

All candidates are based directly on accepted fastsearch4.

fastsearch14 = P1a only
  Preserve an existing same-key TT move when a new TT write has best_move == 0.

fastsearch15 = P1b only
  Keep TT move ordering at PV nodes, but permit TT score/bound cutoffs only at non-PV nodes.

fastsearch16 = P1a + P1b combined
  Test only if 14 and/or 15 show positive evidence.

fastsearch17 = independent P4 experiment
  At a depth-0 node that is in check, floor depth to 1 instead of entering qsearch;
  also adds a MAX_PLY guard before indexing ply-sized search buffers.

No accepted mainline file is overwritten.
