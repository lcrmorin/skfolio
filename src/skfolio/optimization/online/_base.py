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
        self._wealth_initialized: bool = False
        self._projector: AutoProjector | None = None
        self._t: int = 0
        self._last_trade_weights_: np.ndarray | None = None
        self._configured_previous_weights_: np.ndarray | None = None
        self._current_previous_weights_: np.ndarray | None = None
        self._previous_weights_state_initialized: bool = False

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

    def _ensure_previous_weights_state(self, num_assets: int) -> None:
        """Initialize immutable config and mutable online holdings state."""
        if self._previous_weights_state_initialized:
            return

        if self.previous_weights is None:
            configured = None
        else:
            configured = np.asarray(
                self._clean_previous_weights(num_assets), dtype=float
            ).copy()

        self._configured_previous_weights_ = configured
        self._current_previous_weights_ = (
            None if configured is None else configured.copy()
        )
        self._previous_weights_state_initialized = True

    def _initialize_weights(self, num_assets: int):
        if self.initial_weights is not None:
            initial = np.asarray(self.initial_weights, dtype=float)
            if initial.shape != (num_assets,):
                raise ValueError("initial_weights has incompatible shape")
            self.weights_ = initial
        else:
            self.weights_ = np.ones(num_assets, dtype=float) / float(num_assets)
        self._weights_initialized = True

    def _initialize_wealth(self, num_assets: int) -> None:
        """Initialize wealth tracking based on initial_wealth parameter.

        Handles three cases:
        1. Array initial_wealth → compute initial_weights if not provided
        2. Scalar initial_wealth → use as starting wealth
        3. No initial_wealth → default to 1.0

        Raises
        ------
        ValueError
            If both initial_weights and initial_wealth are arrays (incompatible).
        """
        # Case: Both arrays → error
        if self.initial_weights is not None and self.initial_wealth is not None:
            iw_arr = np.asarray(self.initial_weights, dtype=float)
            ih_arr = np.asarray(self.initial_wealth, dtype=float)
            if iw_arr.ndim > 0 and ih_arr.ndim > 0:
                raise ValueError(
                    "initial_weights and initial_wealth cannot both be arrays. "
                    "Provide either initial_weights (for allocation) or "
                    "initial_wealth (for dollar amounts), not both."
                )

        # Case: Array initial_wealth
        if self.initial_wealth is not None:
            ih_arr = np.asarray(self.initial_wealth, dtype=float)
            if ih_arr.ndim > 0:  # Array
                if ih_arr.shape != (num_assets,):
                    raise ValueError(
                        f"initial_wealth array must have shape ({num_assets},), "
                        f"got {ih_arr.shape}"
                    )
                total = float(np.sum(ih_arr))
                if total <= 0:
                    raise ValueError("Sum of initial_wealth must be positive")

                # Set wealth
                self.wealth_ = total

                # Compute weights if not provided
                if not self._weights_initialized:
                    budget = self.budget if self.budget is not None else 1.0
                    self.weights_ = (ih_arr / total) * budget
                    self._weights_initialized = True
            else:  # Scalar
                self.wealth_ = float(self.initial_wealth)
                if self.wealth_ <= 0:
                    raise ValueError("initial_wealth must be positive")
        else:
            # Default: unit wealth
            self.wealth_ = 1.0

        self._wealth_initialized = True

    def _update_wealth(
        self,
        trade_weights: np.ndarray,
        effective_relatives: np.ndarray,
        previous_weights: np.ndarray | None,
    ) -> None:
        """Update wealth after observing period returns.

        Computes net portfolio return accounting for:
        - Gross return: trade_weights^T * effective_relatives
        - Transaction costs: sum(c_i * |trade_weights_i - previous_weights_i|)

        Parameters
        ----------
        trade_weights : np.ndarray
            Weights used for trading (before update).
        effective_relatives : np.ndarray
            Price relatives after management fees.
        previous_weights : np.ndarray | None
            Holdings carried into the current rebalance.
        """
        # Gross portfolio return (after management fees)
        gross_return = float(np.dot(trade_weights, effective_relatives))

        # Transaction costs (proportional to turnover)
        txn_cost = 0.0
        if previous_weights is not None and hasattr(self, "_transaction_costs_arr"):
            prev_arr = np.asarray(previous_weights, dtype=float)
            if prev_arr.shape == trade_weights.shape:
                turnover = np.abs(trade_weights - prev_arr)
                # Total cost as fraction of portfolio
                if np.isscalar(self._transaction_costs_arr):
                    txn_cost = float(self._transaction_costs_arr * np.sum(turnover))
                else:
                    txn_cost = float(np.dot(self._transaction_costs_arr, turnover))

        # Net return after costs
        net_return = gross_return - txn_cost

        # Update wealth: W_{t+1} = W_t * net_return
        self.wealth_ *= max(net_return, 1e-16)  # Prevent negative/zero wealth

    def _drift_weights(
        self,
        trade_weights: np.ndarray,
        effective_relatives: np.ndarray,
    ) -> np.ndarray:
        """Drift trade weights through realized returns."""
        denom = float(np.dot(trade_weights, effective_relatives))
        if denom <= 0.0:
            return trade_weights.copy()
        return (trade_weights * effective_relatives) / denom

    def _apply_initial_projection(self) -> None:
        """Project the initial traded portfolio once before the first round."""
        if self._projector is None or not self._weights_initialized:
            return
        if self._t != 0 or self._last_trade_weights_ is not None:
            return
        # Initial algorithm reference is not an executed allocation.
        static = AutoProjector(replace(
            self._projector.config, previous_weights=None, max_turnover=None
        ))
        self.weights_ = static.project(self.weights_)

    def _compute_effective_relatives(self, gross_relatives: np.ndarray) -> np.ndarray:
        """Apply management fees to gross relatives.

        Parameters
        ----------
        gross_relatives : np.ndarray
            Gross price relatives for one period (shape: (n_assets,)).

        Returns
        -------
        np.ndarray
            Effective relatives after applying management fees multiplicatively.
            If no fees configured, returns gross_relatives unchanged.

        Notes
        -----
        Management fees are applied as multiplicative discount:
        effective_relative_i = gross_relative_i * (1 - fee_i)
        """
        if not hasattr(self, "_management_fees_arr"):
            return gross_relatives

        # If fees are scalar zero, no adjustment needed
        if np.isscalar(self._management_fees_arr):
            if self._management_fees_arr <= 0:
                return gross_relatives
            if self._management_fees_arr >= 1.0:
                warnings.warn(
                    "management_fees >= 1.0 encountered; effective relatives may collapse.",
                    UserWarning,
                    stacklevel=2,
                )
            return gross_relatives * (1.0 - self._management_fees_arr)

        # Array of fees: element-wise multiplication
        fees = np.asarray(self._management_fees_arr, dtype=float)
        if np.any(fees >= 1.0):
            warnings.warn(
                "Some management_fees >= 1.0 encountered; effective relatives may collapse.",
                UserWarning,
                stacklevel=2,
            )
        return gross_relatives * (1.0 - fees)

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
        covers the complete block before any numerical update. Legacy trajectory
        and wealth attributes are retained during the execution-layer migration.
        """
        if getattr(self, "_needs_reset", False):
            raise RuntimeError("A previous update failed; call fit to reset before reuse")
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
                    None if not decision or self.previous_weights is None
                    else self._clean_previous_weights(returns.shape[1]).copy()
                )
                if self._execution_anchor is not None:
                    if not np.isfinite(self._execution_anchor).all():
                        raise ValueError("previous_weights must contain finite holdings")
                    self._execution_anchor.setflags(write=False)
                if self._decision_turnover is not None and self._execution_anchor is None:
                    raise ValueError("max_turnover requires explicit previous_weights holdings")
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
                    if hasattr(self, "all_weights_") else history
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
        self._wealth_initialized = False
        self._is_initialized = False
        self._projector = None
        self._t = 0
        self._last_trade_weights_ = None
        self._configured_previous_weights_ = None
        self._current_previous_weights_ = None
        self._previous_weights_state_initialized = False
        # Note: wealth_ is a learned attribute, only created after fit/partial_fit

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

    def fit_predict(
        self, X: npt.ArrayLike, y: npt.ArrayLike | None = None, **fit_params: Any
    ) -> MultiPeriodPortfolio:
        """Fit the model and return the sequential portfolio trajectory.

        Unlike offline optimization where `fit_predict` evaluates the final learned
        weights uniformly across all periods (look-ahead bias), online optimization
        processes data sequentially. This method returns a `MultiPeriodPortfolio`
        representing the true chronological trajectory of the strategy, evaluating
        each period using the weights computed *prior* to observing that period's returns.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.
        y : Ignored
            Present for API consistency.
        **fit_params : Any
            Additional parameters passed to `fit`.

        Returns
        -------
        MultiPeriodPortfolio
            The sequential trajectory of portfolios over time.
        """
        self.fit(X, y, **fit_params)

        return self._portfolio_from_history(X, self.all_weights_)

    def _portfolio_from_history(
        self, X: npt.ArrayLike, weights: np.ndarray
    ) -> MultiPeriodPortfolio:
        """Build the legacy trajectory until execution moves to model selection."""
        if self.portfolio_params is None:
            ptf_kwargs = {}
        else:
            ptf_kwargs = self.portfolio_params.copy()

        for param in [
            "transaction_costs",
            "management_fees",
            "previous_weights",
            "risk_free_rate",
        ]:
            if param not in ptf_kwargs and hasattr(self, param):
                ptf_kwargs[param] = getattr(self, param)

        name = ptf_kwargs.pop("name", type(self).__name__)

        portfolios = []
        X_arr = np.asarray(X)

        # We need to construct the previous_weights correctly for the transaction costs
        # skfolio's Portfolio charges total_cost = (tc * |prev - w|).sum()
        self._ensure_previous_weights_state(self.n_features_in_)
        prev_w = (
            None
            if self._configured_previous_weights_ is None
            else self._configured_previous_weights_.copy()
        )

        for i in range(len(X_arr)):
            X_i = X_arr[i : i + 1]

            # Create kwargs for this specific portfolio
            ptf_kwargs_i = ptf_kwargs.copy()
            ptf_kwargs_i["previous_weights"] = (
                weights[i].copy() if prev_w is None else prev_w
            )

            ptf = Portfolio(X=X_i, weights=weights[i], name=name, **ptf_kwargs_i)
            portfolios.append(ptf)

            # Drift weights for the NEXT period's previous_weights (if drift_aware logic applies)
            returns_i = 1.0 + X_arr[i]
            prev_w = self._drift_weights(weights[i], returns_i)

        return MultiPeriodPortfolio(portfolios=portfolios, name=name)

    def predict_online(
        self, X: npt.ArrayLike, y: npt.ArrayLike | None = None
    ) -> MultiPeriodPortfolio:
        """
        Predict the online learning trajectory on X without mutating state.

        Deep-copies the current fitted estimator and continues the sequential
        learning process over X.  The original estimator's state is unchanged,
        making this method safe to call repeatedly or alongside live streaming
        via ``partial_fit``.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Net returns for the evaluation period.
        y : Ignored
            Present for API consistency.

        Returns
        -------
        MultiPeriodPortfolio
            Sequential trajectory over X, starting from the current fitted state.
        """
        check_is_fitted(self, "weights_")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim < 2 or X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_arr.shape[1] if X_arr.ndim >= 2 else X_arr.shape[0]} features,"
                f" but {type(self).__name__} was fitted with {self.n_features_in_} features."
            )
        other = deepcopy(self)
        # Preserve the execution anchor at the beginning of this continuation.
        previous = deepcopy(other._current_previous_weights_)
        other.partial_fit(X_arr, y)
        other._configured_previous_weights_ = previous
        return other._portfolio_from_history(X_arr, other.all_weights_[-len(X_arr) :])


# Temporary compatibility alias for Carlo's imports and downstream experiments.
OnlinePortfolioSelection = BaseOnlineOptimization
