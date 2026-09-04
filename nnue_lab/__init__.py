"""Isolated participant-trained NNUE experiment for Deep Blue.

Nothing in this package is imported by the live engine.  The lab deliberately
depends on the engine in only one direction: selected tests and benchmarks read
``deepblue.fastcore`` without changing it.
"""

