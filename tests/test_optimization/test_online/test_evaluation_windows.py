"""Window evaluation learns causally and anchors trades to executed holdings."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.base import clone

from skfolio.measures import RatioMeasure
from skfolio.model_selection import OnlineGridSearch, online_predict, online_score
from skfolio.optimization.online import FollowTheLoser, FollowTheWinner
from skfolio.portfolio import Portfolio


@pytest.mark.parametrize("family", [FollowTheWinner, FollowTheLoser])
@pytest.mark.parametrize("drift", [True, False])
def test_windows_match_manual_execution(family, drift):
    model = family(**({"strategy": "pamr"} if family is FollowTheLoser else {}))
    model.set_params(
        previous_weights=[0.6, 0.4],
        max_turnover=0.2,
        transaction_costs=0.002,
        management_fees=0.001,
        portfolio_params={"weight_drift": drift},
    )
    X = np.random.default_rng(23).normal(0.002, 0.04, (15, 2))
    result = online_predict(model, X, warmup_size=4, test_size=3, reduce_test=True)
    manual = clone(model).fit(X[:4])
    expected = []
    for start in range(4, len(X), 3):
        block = X[start : start + 3]
        anchor = np.asarray(manual.previous_weights)
        assert np.abs(manual.weights_ - anchor).sum() <= 0.2 + 1e-6
        portfolio = Portfolio(
            block,
            weights=manual.weights_,
            previous_weights=anchor,
            transaction_costs=0.002,
            management_fees=0.001,
            weight_drift=drift,
        )
        expected.extend(portfolio.returns)
        manual.set_params(previous_weights=portfolio.ending_weights)
        # Independent row-wise reference: only the last row receives execution limits.
        for index, row in enumerate(block):
            manual._in_warmup = index != len(block) - 1
            if manual._in_warmup:
                cap = manual.max_turnover
                manual.set_params(max_turnover=None, previous_weights=None)
                manual.partial_fit(row)
                manual.set_params(
                    max_turnover=cap, previous_weights=portfolio.ending_weights
                )
            else:
                manual.partial_fit(row)
        manual._in_warmup = False
    assert_allclose(result.returns, expected, atol=1e-9)
    assert_allclose(
        online_score(
            model,
            X,
            warmup_size=4,
            test_size=3,
            reduce_test=True,
            scoring=RatioMeasure.SHARPE_RATIO,
        ),
        result.sharpe_ratio,
    )
    changed = X.copy()
    changed[7:] *= -1
    second = online_predict(
        model, changed, warmup_size=4, test_size=3, reduce_test=True
    )
    assert_allclose(result.returns[:3], second.returns[:3])


def test_grid_search_refits_completed_windows():
    X = np.random.default_rng(24).normal(0.002, 0.04, (16, 2))
    model = FollowTheWinner(
        previous_weights=[0.5, 0.5],
        max_turnover=0.3,
        portfolio_params={"weight_drift": True},
    )
    search = OnlineGridSearch(
        model, param_grid={"learning_rate": [0.05, 0.1]}, warmup_size=4, test_size=3
    ).fit(X)
    assert search.best_estimator_.n_observations_ == len(X)
    assert np.isfinite(search.best_estimator_.weights_).all()
