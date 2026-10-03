"""Base Online Portfolio Selection estimator."""

# Copyright (c) 2025
# Author: Carlo Nicolini <nicolini.carlo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.base import check_is_fitted
from sklearn.utils.validation import _check_sample_weight, validate_data

import skfolio.typing as skt
from skfolio.optimization._base import BaseOptimization
from skfolio.optimization.online._mixins import (
    OnlineParameterConstraintsMixin,
)
from skfolio.optimization.online._projection import (
    AutoProjector,
    ProjectionConfig,
)
from skfolio.optimization.online._utils import net_to_relatives
from skfolio.portfolio import MultiPeriodPortfolio, Portfolio
from skfolio.utils.tools import input_to_array


class BaseOnlineOptimization(BaseOptimization, OnlineParameterConstraintsMixin):
    """Online Portfolio Selection (OPS) base class.

    This class serves as a foundation for implementing various online portfolio
    selection algorithms. It handles the common logic for fitting data sequentially,
    managing weights, and applying projections. Subclasses should implement the
    core update logic.
    """

    def __init__(
        self,
        *,
        warm_start: bool = True,
        initial_weights: npt.ArrayLike | None = None,
        initial_wealth: float | npt.ArrayLike | None = None,
        previous_weights: skt.MultiInput | None = None,
        transaction_costs: skt.MultiInput = 0.0,
        management_fees: skt.MultiInput = 0.0,
        groups: skt.Groups | None = None,
        linear_constraints: skt.LinearConstraints | None = None,
        left_inequality: skt.Inequality | None = None,
        right_inequality: skt.Inequality | None = None,
        max_turnover: float | None = None,
        min_weights: skt.MultiInput | None = 0.0,
        max_weights: skt.MultiInput | None = 1.0,
        budget: float | None = 1.0,
        X_tracking: npt.ArrayLike | None = None,
        tracking_error_benchmark: npt.ArrayLike | None = None,
        max_tracking_error: float | None = None,
        covariance: npt.ArrayLike | None = None,
        variance_bound: float | None = None,
        portfolio_params: dict | None = None,
        fallback: skt.Fallback = None,
        raise_on_failure: bool = True,
    ):
        super().__init__(
            portfolio_params=portfolio_params,
            fallback=fallback,
            raise_on_failure=raise_on_failure,
        )
        self.warm_start = warm_start
        self.initial_weights = initial_weights
        self.initial_wealth = initial_wealth
        self.previous_weights = previous_weights
        self.transaction_costs = transaction_costs
        self.management_fees = management_fees
        self.groups = groups
        self.linear_constraints = linear_constraints
        self.left_inequality = left_inequality
        self.right_inequality = right_inequality
        self.max_turnover = max_turnover
        self.min_weights = min_weights
        self.max_weights = max_weights
        self.budget = budget
        self.X_tracking = X_tracking
        self.tracking_error_benchmark = tracking_error_benchmark
        self.max_tracking_error = max_tracking_error
        self.covariance = covariance
        self.variance_bound = variance_bound

        self._is_initialized: bool = False
        self._weights_initialized: bool = False
        self._projector: AutoProjector | None = None
        self._t: int = 0
        self._last_trade_weights_: np.ndarray | None = None

    def _initialize_projector(self):
        projection_config = ProjectionConfig(
            lower=self.min_weights,
            upper=self.max_weights,
            budget=self.budget,
            groups=self.groups,
            linear_constraints=self.linear_constraints,
            left_inequality=self.left_inequality,
            right_inequality=self.right_inequality,
            X_tracking=self.X_tracking,
            tracking_error_benchmark=self.tracking_error_benchmark,
            max_tracking_error=self.max_tracking_error,
            covariance=self.covariance,
            variance_bound=self.variance_bound,
            previous_weights=self._execution_anchor,
            max_turnover=self._decision_turnover,
        )
        return AutoProjector(projection_config)

    def _initialize_weights(self, num_assets: int) -> None:
        """Initialize an algorithm reference, without constructing an account."""
        initial = self.initial_weights
        if self.initial_wealth is not None:
            wealth = np.asarray(self.initial_wealth, dtype=float)
            if not np.isfinite(wealth).all() or np.any(wealth < 0) or wealth.sum() <= 0:
                raise ValueError("initial_wealth must be finite and positive")
            if wealth.ndim:
                if initial is not None:
                    raise ValueError(
                        "initial_weights and initial_wealth cannot both be arrays"
                    )
                if wealth.shape != (num_assets,):
                    raise ValueError(f"initial_wealth must have shape ({num_assets},)")
                initial = wealth / wealth.sum()
        if initial is None:
            initial = np.full(num_assets, 1 / num_assets)
        initial = np.asarray(initial, dtype=float)
        if initial.shape != (num_assets,) or not np.isfinite(initial).all():
            raise ValueError(
                "initial_weights has incompatible shape or nonfinite values"
            )
        self.weights_ = initial.copy()
        self._weights_initialized = True

    def _apply_initial_projection(self) -> None:
        """Project the initial traded portfolio once before the first round."""
        if self._projector is None or not self._weights_initialized:
            return
        if self._t != 0 or self._last_trade_weights_ is not None:
            return
        # Initial algorithm reference is not an executed allocation.
        static = AutoProjector(
            replace(self._projector.config, previous_weights=None, max_turnover=None)
        )
        self.weights_ = static.project(self.weights_)

    def _compute_effective_relatives(self, gross_relatives: np.ndarray) -> np.ndarray:
        """Use additive fees in fee-aware learning, matching Portfolio returns.

        This adjusts the decision objective only. Actual fees are booked exclusively
        by Portfolio. No holdings or wealth are computed here.
        """
        return gross_relatives - getattr(self, "_management_fees_arr", 0.0)

    _requires_single_period_evaluation = True

    def partial_fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike | None = None,
        sample_weight: npt.ArrayLike | None = None,
        **fit_params: Any,
    ) -> BaseOnlineOptimization:
        """Continue learning from one observation or a causal block of net returns.

        The target after observing row t is intended for period t+1. Validation
        covers the complete block before any numerical update. Targets recorded in all_weights_ are diagnostic learning history, not trades.
        """
        if getattr(self, "_needs_reset", False):
            raise RuntimeError(
                "A previous update failed; call fit to reset before reuse"
            )
        self._validate_params()
        if fit_params:
            raise TypeError(f"Unsupported fit parameters: {sorted(fit_params)}")
        if not hasattr(X, "columns"):
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(1, -1)
        returns = validate_data(
            self,
            X=X,
            reset=not self._is_initialized,
            dtype=float,
            ensure_2d=True,
            accept_sparse=False,
        )
        if sample_weight is not None:
            _check_sample_weight(sample_weight, returns)
            warnings.warn(
                "sample_weight is ignored for online estimators.",
                UserWarning,
                stacklevel=2,
            )
        warmup = getattr(self, "_in_warmup", False)
        if len(returns) > 1 and self.max_turnover is not None and not warmup:
            raise ValueError(
                "Execution-constrained blocks require row-aligned holdings; "
                "use fit for warmup or one-row partial_fit for each rebalance"
            )
        trade_list = []
        try:
            for i, row in enumerate(net_to_relatives(returns)):
                decision = not warmup or i == len(returns) - 1
                self._decision_turnover = self.max_turnover if decision else None
                self._execution_anchor = (
                    None
                    if not decision or self.previous_weights is None
                    else self._clean_previous_weights(returns.shape[1]).copy()
                )
                if self._execution_anchor is not None:
                    if not np.isfinite(self._execution_anchor).all():
                        raise ValueError(
                            "previous_weights must contain finite holdings"
                        )
                    self._execution_anchor.setflags(write=False)
                if (
                    self._decision_turnover is not None
                    and self._execution_anchor is None
                ):
                    raise ValueError(
                        "max_turnover requires explicit previous_weights holdings"
                    )
                self._update_one(row)
                if not hasattr(self, "initial_weights_"):
                    self.initial_weights_ = self._last_trade_weights_.copy()
                self.n_observations_ = self._t
                trade_list.append(self._last_trade_weights_.copy())
        except Exception:
            # Engines/predictors may already have consumed the failing observation.
            # Do not pretend that keeping the previous public target rolls them back.
            self._needs_reset = True
            raise
        finally:
            if trade_list:
                history = np.asarray(trade_list)
                self.all_weights_ = (
                    np.vstack((self.all_weights_, history))
                    if hasattr(self, "all_weights_")
                    else history
                )
        return self

    @contextmanager
    def _warmup_context(self):
        """Mark historical rows as learning-only until the final entry decision."""
        old = getattr(self, "_in_warmup", False)
        self._in_warmup = True
        try:
            yield
        finally:
            self._in_warmup = old

    def _update_one(self, gross_relatives: np.ndarray) -> None:
        """Observe one return vector using the existing strategy machinery."""
        raise NotImplementedError

    def _reset_state_for_fit(self) -> None:
        """Clear learned state; subclasses also reset their numerical components."""
        for name in list(vars(self)):
            if name.endswith("_") and not name.startswith("__"):
                delattr(self, name)
        self._needs_reset = False
        self._weights_initialized = False
        self._is_initialized = False
        self._projector = None
        self._t = 0
        self._last_trade_weights_ = None

    def fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike | None = None,
        **fit_params: Any,
    ) -> BaseOnlineOptimization:
        """Reset learned state and process observations causally.

        ``warm_start`` is temporarily accepted for source compatibility but no
        longer controls this method. Use ``partial_fit`` to continue a stream.
        """
        self._reset_state_for_fit()
        with self._warmup_context():
            return self.partial_fit(X, y, **fit_params)

    def _clean_input(
        self,
        value: float | dict | npt.ArrayLike | None,
        n_assets: int,
        fill_value: Any,
        name: str,
    ) -> float | np.ndarray:
        """Convert input to cleaned float or ndarray."""
        if value is None:
            return fill_value
        if np.isscalar(value):
            v = float(value)
            if name == "management_fees" and v >= 1.0:
                raise ValueError("management_fees must be < 1.0 per period")
            return v
        arr = input_to_array(
            items=value,
            n_assets=n_assets,
            fill_value=fill_value,
            dim=1,
            assets_names=(
                self.feature_names_in_ if hasattr(self, "feature_names_in_") else None
            ),
            name=name,
        )
        if name == "management_fees" and np.any(np.asarray(arr, dtype=float) >= 1.0):
            raise ValueError("All management_fees values must be < 1.0 per period")
        return arr

    def _initialize_for_evaluation(self, X: npt.ArrayLike) -> None:
        """Initialize before the first return without consuming that observation."""
        self._validate_params()
        data = validate_data(self, X, reset=True, dtype=float)
        self._decision_turnover = self.max_turnover
        self._execution_anchor = (
            None
            if self.previous_weights is None
            else self._clean_previous_weights(data.shape[1]).copy()
        )
        if self.max_turnover is not None and self._execution_anchor is None:
            raise ValueError("max_turnover requires explicit previous_weights holdings")
        self._initialize_algorithm(data.shape[1])
        if self.max_turnover is not None:
            self.weights_ = self._projector.project(self.weights_)
        self._synchronize_target()
        self.initial_weights_ = self.weights_.copy()
        self.n_observations_ = 0

    def _initialize_algorithm(self, n_assets: int) -> None:
        """Initialize numerical components without observing returns."""
        raise NotImplementedError

    def _synchronize_target(self) -> None:
        """Use the final decision as the next local update center."""
        for name in ("_foco_engine", "_engine"):
            engine = getattr(self, name, None)
            if engine is not None and hasattr(engine, "_x_t"):
                engine._x_t = self.weights_.copy()
        strategy = getattr(self, "_strategy_impl", None)
        if strategy is not None and hasattr(strategy, "_mu"):
            strategy._mu = self.weights_.copy()

    def predict(self, X: npt.ArrayLike) -> Portfolio:
        """Evaluate a target using skfolio's common accounting implementation."""
        if getattr(self, "_needs_reset", False):
            raise RuntimeError(
                "A previous update failed; call fit to reset before reuse"
            )
        return super().predict(X)

    def fit_predict(
        self, X: npt.ArrayLike, y: npt.ArrayLike | None = None, **fit_params: Any
    ) -> MultiPeriodPortfolio:
        """Evaluate from the initial allocation, with no warmup, using Portfolio.

        For a historical learning window followed by evaluation, use online_predict.
        Initial wealth, if supplied, configures the algorithm reference only; the
        returned portfolio uses skfolio's normalised return/wealth convention.
        """
        if fit_params:
            raise TypeError(f"Unsupported fit parameters: {sorted(fit_params)}")
        from skfolio.model_selection._online._validation import _continue_online_predict

        self._reset_state_for_fit()
        self._initialize_for_evaluation(X)
        previous = self.previous_weights
        try:
            return _continue_online_predict(self, X, y, refit_last=True)
        finally:
            self.previous_weights = previous

    def predict_online(
        self, X: npt.ArrayLike, y: npt.ArrayLike | None = None
    ) -> MultiPeriodPortfolio:
        """Continue evaluation on a fitted copy through common model selection."""
        from skfolio.model_selection._online._validation import _continue_online_predict

        check_is_fitted(self, "weights_")
        if getattr(self, "_needs_reset", False):
            raise RuntimeError(
                "A previous update failed; call fit to reset before reuse"
            )
        return _continue_online_predict(deepcopy(self), X, y, refit_last=False)


# Temporary compatibility alias for Carlo's imports and downstream experiments.
OnlinePortfolioSelection = BaseOnlineOptimization
