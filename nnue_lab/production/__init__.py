"""Isolated no-king-bucket Perspective Chess768 production prototype."""

from nnue_lab.production.features import NUM_FEATURES, PADDING_FEATURE
from nnue_lab.production.model import PerspectiveChess768NNUE

__all__ = ("NUM_FEATURES", "PADDING_FEATURE", "PerspectiveChess768NNUE")
