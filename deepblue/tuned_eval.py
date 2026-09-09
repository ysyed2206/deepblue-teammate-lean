"""Texel-tuned material+piece-square tables.

Fitted by tools/texel_tune.py against 400,000 Stockfish-annotated Lichess
positions (CC0), minimising sigmoid-space error against the teacher score
through the same phase blend the engine uses at runtime.

Validation loss 0.021265 -> 0.014968 (+29.61% relative), with train and
validation within 0.4% of each other -- 768 fitted values against 400k
positions is a healthy ratio, so this is a fit, not a memorisation.

Only the WHITE half is fitted. Black entries are derived as the negated,
vertically mirrored white entries, so the evaluation stays exactly colour
symmetric by construction rather than by the optimiser happening to find it.
"""

from __future__ import annotations

import numpy as np

TUNED_TEMPO_BONUS = 35

WHITE_MG = {
    1: [94, 98, 104, 110, 110, 104, 98, 94, -32, 6, -9, -16, -2, 24, 42, 36, -21, 2, 12, 9, 34, 21, 48, 47, -6, 10, 18, 29, 32, 34, 41, 42, 21, 25, 42, 46, 62, 57, 66, 83, -27, 89, 31, 88, 95, 125, 123, 96, 160, 195, 192, 211, 286, 217, 183, 261, 94, 98, 104, 110, 110, 104, 98, 94],
    2: [41, 69, 7, 63, 85, 95, 64, 75, 51, 74, 96, 93, 97, 116, 50, 72, 70, 92, 108, 99, 142, 104, 117, 103, 102, 124, 135, 103, 130, 148, 170, 110, 123, 142, 134, 170, 135, 178, 141, 200, 146, 136, 125, 230, 222, 212, 202, 208, 197, 203, 230, 197, 292, 201, 195, 261, -10, 155, 106, 16, 281, 45, 359, 192],
    3: [118, 131, 123, 96, 110, 104, 162, 161, 161, 159, 173, 135, 133, 145, 155, 107, 170, 135, 149, 138, 154, 142, 149, 154, 125, 169, 133, 170, 165, 150, 144, 153, 166, 145, 192, 173, 202, 167, 153, 121, 173, 246, 103, 220, 174, 123, 156, 211, 251, 184, 185, 56, 161, 191, 177, 211, 18, 132, -36, -53, 22, 57, 138, 159],
    4: [131, 149, 159, 157, 179, 149, 211, 160, 106, 105, 162, 135, 167, 165, 212, 174, 142, 185, 174, 151, 164, 197, 235, 232, 145, 124, 177, 144, 189, 213, 216, 241, 159, 206, 186, 240, 246, 200, 285, 277, 210, 301, 238, 277, 312, 303, 413, 440, 274, 282, 273, 384, 347, 396, 385, 290, 291, 363, 227, 232, 361, 278, 360, 342],
    5: [687, 685, 687, 695, 699, 699, 736, 769, 651, 700, 708, 706, 712, 728, 710, 777, 703, 711, 700, 690, 703, 700, 723, 762, 709, 687, 689, 693, 713, 684, 742, 733, 701, 688, 715, 721, 729, 667, 674, 770, 694, 656, 748, 717, 862, 739, 844, 834, 685, 709, 662, 589, 610, 785, 759, 774, 597, 700, 840, 79, 812, 821, 834, 747],
    6: [-57, 80, 46, -58, 7, -60, 61, 49, -69, 28, -21, -118, -78, -43, 18, 12, -126, -136, -151, -144, -134, -149, -96, -210, -117, -99, -159, -174, -79, -24, -222, -204, -75, -105, 75, -339, 118, 52, -112, -259, -80, 191, 208, -81, -4, 332, 69, -28, 124, 77, 35, -56, -51, 231, 45, 56, -35, 322, -101, -218, -79, -89, 76, -21],
}

WHITE_EG = {
    1: [120, 120, 120, 120, 120, 120, 120, 120, 199, 182, 147, 82, 150, 169, 175, 128, 178, 165, 108, 121, 118, 126, 136, 100, 160, 179, 111, 135, 120, 136, 140, 127, 186, 175, 127, 144, 126, 158, 211, 120, 255, 291, 175, 194, 159, 54, 276, 165, 377, 307, 339, 234, 184, 199, 372, 138, 120, 120, 120, 120, 120, 120, 120, 120],
    2: [164, 173, 220, 214, 204, 105, 184, 55, 237, 100, 379, 275, 248, 252, 221, 346, 209, 225, 311, 349, 311, 312, 263, 254, 284, 315, 361, 376, 324, 321, 287, 257, 281, 319, 312, 400, 395, 419, 333, 302, 351, 324, 379, 299, 303, 399, 318, 338, 172, 211, 274, 333, 336, 364, 178, 180, 276, 291, 298, 308, 258, 288, 336, 159],
    3: [208, 320, 218, 291, 237, 241, 337, 169, 233, 277, 325, 269, 304, 295, 338, 268, 350, 353, 327, 336, 325, 336, 269, 282, 302, 276, 369, 319, 322, 347, 247, 272, 203, 445, 310, 416, 326, 354, 320, 333, 258, 169, 382, 273, 328, 425, 350, 326, 177, 311, 218, 510, 401, 304, 227, 342, 363, 239, 447, 256, 519, 318, 380, 269],
    4: [443, 461, 456, 481, 466, 474, 417, 430, 420, 434, 453, 445, 427, 434, 419, 406, 441, 450, 410, 454, 436, 415, 439, 377, 491, 497, 462, 484, 461, 439, 413, 426, 499, 501, 488, 474, 441, 500, 476, 420, 496, 470, 511, 497, 445, 462, 481, 389, 461, 454, 479, 475, 448, 449, 420, 477, 446, 478, 548, 514, 462, 488, 421, 422],
    5: [500, 602, 578, 502, 575, 480, 494, 581, 566, 529, 510, 554, 554, 487, 487, 576, 587, 576, 618, 599, 597, 665, 552, 595, 617, 674, 694, 719, 639, 712, 636, 597, 664, 615, 725, 741, 800, 881, 807, 640, 665, 749, 688, 754, 646, 903, 658, 653, 710, 723, 779, 856, 831, 872, 869, 834, 821, 714, 655, 1104, 690, 613, 629, 573],
    6: [-22, -85, -54, -50, -84, -4, -57, -68, -15, -54, -73, -20, -39, -53, -41, -51, -71, -24, -25, -16, -34, -38, -35, 19, 14, 27, 11, 45, -6, -29, -2, -3, -27, 58, 32, 68, 33, 21, 42, -25, 4, 0, -11, 41, 12, 6, 58, 3, 36, 147, 55, 89, 1, 102, 95, 60, 9, 139, 114, 27, 75, 143, 53, -39],
}


def build_tables():
    """Signed, pre-mirrored [12,64] tables, matching fastsearch._build_tables."""
    middlegame = np.zeros((12, 64), dtype=np.int32)
    endgame = np.zeros((12, 64), dtype=np.int32)
    for index, piece_type in enumerate((1, 2, 3, 4, 5, 6)):
        white_mg = WHITE_MG[piece_type]
        white_eg = WHITE_EG[piece_type]
        for square in range(64):
            middlegame[index, square] = white_mg[square]
            endgame[index, square] = white_eg[square]
            middlegame[index + 6, square] = -white_mg[square ^ 56]
            endgame[index + 6, square] = -white_eg[square ^ 56]
    return middlegame, endgame


TUNED_MG_TABLE, TUNED_EG_TABLE = build_tables()

# Uniform rescale back to the engine's native centipawn scale.
# Texel tuning fits values only up to the sigmoid constant K, so the fitted
# tables came out roughly 2x compressed against a K of 400. The rest of the
# engine carries FIXED centipawn constants that assume a pawn is about 100 --
# RFP_MARGIN_PER_DEPTH, the pawn-shield penalty, the passed-pawn bonus --
# and those do not move with the tables. A uniform rescale preserves every
# relative weight the tuner learned while restoring that compatibility.
RESCALE = 1.392
SCALED_MG_TABLE = np.rint(TUNED_MG_TABLE * RESCALE).astype(np.int32)
SCALED_EG_TABLE = np.rint(TUNED_EG_TABLE * RESCALE).astype(np.int32)

