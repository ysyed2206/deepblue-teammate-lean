Deep Blue Patch 06c — SEE regression fixture correction

Corrects only the colour-flipped promotion-inside-exchange regression FEN in tools/see_diff.py.
The mirrored piece on c2 is a WHITE knight (N), not a black knight (n).
No SEE/search algorithm code is changed relative to Patch 06b.
