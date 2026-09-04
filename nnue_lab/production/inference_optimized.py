"""Isolated, exact Q1 tail experiments; canonical runtime remains unchanged.

Call ``select_tail_kernel(model)`` once for immutable, validated parameters.
Both candidates hoist perspective views and compute clipped weighted squares
in int32. For QA <= 255 and int16 output weights, an individual term is bounded
by 32768 * 255**2 = 2,130,739,200 < INT32_MAX. The full-int32 candidate additionally
requires QA**2 * sum(abs(output_weights)) <= INT32_MAX, which bounds every partial
sum for any accumulator state and either perspective. Otherwise use int64 sums.
QA > 255 falls back to the canonical int64 kernel. No floating point is used.
"""

# mypy: disable-error-code="no-untyped-def"

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit

from nnue_lab.production.inference_numba import evaluate_ready, truncating_divide

if TYPE_CHECKING:
    from nnue_lab.production.model import QuantizedParameters


def tail_bounds(model: QuantizedParameters) -> dict[str, int | bool]:
    """Validate an artifact and derive bounds with unbounded Python integers."""
    model.validate()
    weights = [int(value) for value in model.output_weights]
    product_bound = model.qa * model.qa * max(abs(value) for value in weights)
    dot_bound = model.qa * model.qa * sum(abs(value) for value in weights)
    return {
        "product_absolute_bound": product_bound,
        "dot_absolute_bound": dot_bound,
        "int32_products_safe": model.qa <= 255,
        "int32_reduction_safe": model.qa <= 255 and dot_bound <= 2_147_483_647,
    }


def select_tail_kernel(model: QuantizedParameters):
    """Prepare once; parameters must not change after validation/selection.

    This selection is a lab candidate only, not an installed engine evaluator.
    """
    bounds = tail_bounds(model)
    if bounds["int32_reduction_safe"]:
        return _evaluate_ready_int32_sum
    if bounds["int32_products_safe"]:
        return _evaluate_ready_int32_products
    return evaluate_ready


@njit(cache=True, nogil=True, inline="always")
def _finish(dot, output_bias, qa, qb, evaluation_scale):
    # Preserve both truncations and the placement of the QA*QB-scaled bias.
    dequantized = truncating_divide(np.int64(dot), np.int64(qa)) + np.int64(
        output_bias
    )
    return truncating_divide(
        dequantized * np.int64(evaluation_scale), np.int64(qa) * np.int64(qb)
    )


@njit(cache=True, nogil=True)
def _evaluate_ready_int32_products(
    accumulators, side_to_move, output_weights, output_bias, qa, qb, evaluation_scale
):
    """Private: select_tail_kernel must establish int16 weights and QA <= 255."""
    width = accumulators.shape[1]
    stm = accumulators[side_to_move]
    opponent = accumulators[1 - side_to_move]
    qa32 = np.int32(qa)
    dot = np.int64(0)
    for neuron in range(width):
        us = np.int32(min(max(stm[neuron], np.int32(0)), qa32))
        them = np.int32(min(max(opponent[neuron], np.int32(0)), qa32))
        us_term = np.int32(np.int32(us * us) * np.int32(output_weights[neuron]))
        them_term = np.int32(
            np.int32(them * them) * np.int32(output_weights[width + neuron])
        )
        dot += np.int64(us_term) + np.int64(them_term)
    return _finish(dot, output_bias, qa, qb, evaluation_scale)


@njit(cache=True, nogil=True)
def _evaluate_ready_int32_sum(
    accumulators, side_to_move, output_weights, output_bias, qa, qb, evaluation_scale
):
    """Private: additionally requires the full absolute dot bound <= INT32_MAX."""
    width = accumulators.shape[1]
    stm = accumulators[side_to_move]
    opponent = accumulators[1 - side_to_move]
    qa32 = np.int32(qa)
    dot = np.int32(0)
    for neuron in range(width):
        us = np.int32(min(max(stm[neuron], np.int32(0)), qa32))
        them = np.int32(min(max(opponent[neuron], np.int32(0)), qa32))
        us_term = np.int32(np.int32(us * us) * np.int32(output_weights[neuron]))
        them_term = np.int32(
            np.int32(them * them) * np.int32(output_weights[width + neuron])
        )
        dot = np.int32(dot + us_term)
        dot = np.int32(dot + them_term)
    return _finish(np.int64(dot), output_bias, qa, qb, evaluation_scale)
