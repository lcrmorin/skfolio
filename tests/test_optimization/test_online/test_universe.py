"""Named universe changes restart kernels without hiding liquidation turnover."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from sklearn.base import clone

from skfolio.optimization.online import FollowTheLoser, FollowTheWinner


@pytest.mark.parametrize("family", [FollowTheWinner, FollowTheLoser])
def test_reorder_preserves_state_and_membership_change_restarts(family):
    model = family(
        universe_policy="restart",
        **({"strategy": "pamr"} if family is FollowTheLoser else {}),
    )
    first = pd.DataFrame([[0.01, 0.02], [0.02, -0.01]], columns=["a", "b"])
    reference = clone(model).fit(first)
    model.fit(first)
    model.partial_fit(first[["b", "a"]])
    reference.partial_fit(first)
    assert_allclose(model.weights_, reference.weights_)
    assert model.n_observations_ == 4
    changed = pd.DataFrame([[0.03, -0.02, 0.01]], columns=["a", "c", "d"])
    model.partial_fit(changed)
    assert_allclose(model.weights_, clone(model).fit(changed).weights_)
    assert model.n_observations_ == 1
    assert model.n_universe_restarts_ == 1
    assert list(model.previous_universe_) == ["a", "b"]
    assert model.all_weights_.shape == (1, 3)


def test_invalid_new_universe_does_not_reset():
    model = FollowTheWinner(universe_policy="restart").fit(
        pd.DataFrame([[0.01, 0.02]], columns=["a", "b"])
    )
    with pytest.raises(ValueError, match=r"NaN|finite"):
        model.partial_fit(pd.DataFrame([[np.nan, 0.02]], columns=["a", "c"]))
    assert list(model.feature_names_in_) == ["a", "b"]
    assert model.n_observations_ == 1
    with pytest.raises(ValueError, match="y must"):
        model.partial_fit(pd.DataFrame([[0.01, 0.02]], columns=["a", "c"]), y=[1, 2])
    assert list(model.feature_names_in_) == ["a", "b"]
    model.set_params(min_weights=[0, 0])
    with pytest.raises(ValueError, match="scalar or named"):
        model.partial_fit(pd.DataFrame([[0.01, 0.02]], columns=["a", "c"]))
    assert model.n_observations_ == 1


@pytest.mark.parametrize("recovery", ["rollback", "previous_weights"])
def test_liquidation_counts_towards_turnover(recovery):
    X = pd.DataFrame([[0.01, 0.02]], columns=["a", "b"])
    model = FollowTheWinner(universe_policy="restart", recovery=recovery).fit(X)
    model.set_params(previous_weights={"a": 0.4, "b": 0.6}, max_turnover=1.1)
    changed = pd.DataFrame([[0.02, 0.01]], columns=["a", "c"])
    with pytest.raises(ValueError):
        model.partial_fit(changed)
    assert list(model.feature_names_in_) == ["a", "b"]
    model.set_params(max_turnover=1.3)
    model.partial_fit(changed)
    turnover = 0.6 + np.abs(model.weights_ - [0.4, 0]).sum()
    assert turnover <= 1.3 + 1e-6
    portfolio = model.predict(changed)
    assert_allclose(portfolio.turnover, turnover)


def test_universe_restart_clones_a_fresh_prior():
    from skfolio.moments import EWCovariance, EWMu
    from skfolio.prior import EmpiricalPrior

    model = FollowTheWinner(
        universe_policy="restart",
        prior_estimator=EmpiricalPrior(
            mu_estimator=EWMu(min_observations=1),
            covariance_estimator=EWCovariance(min_observations=1),
        ),
    ).fit(pd.DataFrame([[0.01, 0.02], [0.02, 0.03]], columns=["a", "b"]))
    old_prior = model.prior_estimator_
    changed = pd.DataFrame([[0.03, 0.01]], columns=["a", "c"])
    model.partial_fit(changed)
    reference = clone(model).fit(changed)
    assert model.prior_estimator_ is not old_prior
    assert_allclose(
        model.prior_estimator_.return_distribution_.mu,
        reference.prior_estimator_.return_distribution_.mu,
    )
