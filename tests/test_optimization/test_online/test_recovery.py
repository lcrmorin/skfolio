"""Recovery restores the complete learner, not just its published weights."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from skfolio.moments import EWCovariance, EWMu
from skfolio.optimization.online import FollowTheLoser, FollowTheWinner
from skfolio.prior import EmpiricalPrior


@pytest.mark.parametrize("family", [FollowTheWinner, FollowTheLoser])
def test_rollback_restores_kernel_prior_and_whole_block(monkeypatch, family):
    kwargs = {} if family is FollowTheWinner else {"strategy": "pamr"}
    prior = EmpiricalPrior(
        mu_estimator=EWMu(min_observations=1),
        covariance_estimator=EWCovariance(min_observations=1),
        max_history=10,
    )
    model = family(**kwargs, recovery="rollback", prior_estimator=prior)
    reference = family(**kwargs, recovery="rollback", prior_estimator=prior)
    initial = [[0.01, 0.02], [0.02, -0.01]]
    model.fit(initial)
    reference.fit(initial)
    before = model.weights_.copy()
    original = family._update_one

    def fail_after_mutation(self, row):
        original(self, row)
        if row[0] > 1.5:
            raise RuntimeError("Injected post-update failure")

    monkeypatch.setattr(family, "_update_one", fail_after_mutation)
    with pytest.raises(RuntimeError, match="Injected"):
        model.partial_fit([[0.03, 0.01], [0.6, 0.02]])
    assert_allclose(model.weights_, before)
    assert model.n_observations_ == 2
    assert model.prior_estimator is prior
    assert model.last_update_consumed_ is False
    assert model.n_recoveries_ == 1
    assert_allclose(model.all_weights_, reference.all_weights_)
    assert_allclose(
        model.prior_estimator_.return_distribution_.covariance,
        reference.prior_estimator_.return_distribution_.covariance,
    )
    model.predict([[0.01, 0.01]])
    corrected = [[0.03, 0.01], [0.04, 0.02]]
    model.partial_fit(corrected)
    reference.partial_fit(corrected)
    assert_allclose(model.weights_, reference.weights_)
    assert model.last_update_consumed_ is True
    assert model.recovery_error_ is None


def test_holdings_recovery_continues_after_allocation_failure(monkeypatch):
    model = FollowTheWinner(recovery="previous_weights", previous_weights=[0.8, 0.2])
    model.fit([[0.01, 0.02]])
    original = FollowTheWinner._update_one

    def fail(self, row):
        original(self, row)
        if row[0] > 1.5:
            raise ValueError("Injected allocation failure")

    monkeypatch.setattr(FollowTheWinner, "_update_one", fail)
    with pytest.warns(UserWarning, match="not learned"):
        model.partial_fit([[0.6, 0.02]])
    assert_allclose(model.weights_, [0.8, 0.2])
    assert_allclose(model._foco_engine._x_t, model.weights_)
    assert model.fallback_ == "previous_weights"
    assert model.n_observations_ == 1
    model.predict([[0.02, 0.01]])
    model.partial_fit([[0.01, 0.02]])
    assert model.n_observations_ == 2
    assert model.fallback_ is None


def test_invalid_input_never_becomes_a_holdings_fallback():
    model = FollowTheWinner(recovery="previous_weights", previous_weights=[0.5, 0.5])
    model.fit([[0.01, 0.02]])
    with pytest.raises(ValueError):
        model.partial_fit([[np.nan, 0.02]])
    assert model.n_observations_ == 1
    model.partial_fit([[0.02, 0.01]])
