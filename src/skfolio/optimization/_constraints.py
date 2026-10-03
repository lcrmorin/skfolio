"""Shared convex expressions for portfolio constraints.

Inputs are already aligned to the investable asset universe by the caller.
"""

import cvxpy as cp
import cvxpy.constraints as cpc
import numpy as np

import skfolio.typing as skt
from skfolio.typing import FloatArray


def weight_constraints(
    w: cp.Variable,
    factor: skt.Factor,
    min_weights: float | FloatArray | None,
    max_weights: float | FloatArray | None,
    allow_negative_weights: bool,
    max_long: float | None = None,
    max_short: float | None = None,
    scale: float = 1.0,
    owner: str = "Optimizer",
) -> list[cpc.Constraint]:
    """Constrain aligned weights and aggregate long and short exposure."""
    constraints = []
    if min_weights is not None:
        if not allow_negative_weights and np.any(min_weights < 0):
            raise ValueError(
                f"{owner} must have non negative `min_weights` "
                f"constraint otherwise the problem becomes non-convex."
            )
        constraints.append(w * scale >= min_weights * factor * scale)

    if max_weights is not None:
        constraints.append(w * scale <= max_weights * factor * scale)

    if max_long is not None:
        max_long = float(max_long)
        if max_long <= 0:
            raise ValueError("`max_long` must be strictly positive")
        constraints.append(cp.sum(cp.pos(w)) * scale <= max_long * factor * scale)

    if max_short is not None:
        max_short = float(max_short)
        if max_short <= 0:
            raise ValueError("`max_short` must be strictly positive")
        constraints.append(cp.sum(cp.neg(w)) * scale <= max_short * factor * scale)

    return constraints


def budget_constraints(
    w: cp.Variable,
    factor: skt.Factor,
    budget: float | None,
    min_budget: float | None = None,
    max_budget: float | None = None,
    scale: float = 1.0,
) -> list[cpc.Constraint]:
    """Constrain the equality budget or lower and upper budget limits."""
    constraints = []
    if min_budget is not None:
        constraints.append(cp.sum(w) * scale >= float(min_budget) * factor * scale)

    if max_budget is not None:
        constraints.append(cp.sum(w) * scale <= float(max_budget) * factor * scale)

    if budget is not None:
        if max_budget is not None:
            raise ValueError(
                "`max_budget`and `budget` cannot be provided at the same time"
            )
        if min_budget is not None:
            raise ValueError(
                "`min_budget`and `budget` cannot be provided at the same time"
            )
        constraints.append(cp.sum(w) * scale == float(budget) * factor * scale)

    return constraints
