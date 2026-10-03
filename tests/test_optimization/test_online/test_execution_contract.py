"""Execution boundaries for the port of Carlo's online framework."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from skfolio.optimization.online import FollowTheLoser, FollowTheWinner


@pytest.mark.parametrize("kind", ["winner", "loser"])
def test_loss_uses_pre_observation_target(kind):
    model = (FollowTheWinner(learning_rate=2.0, update_mode="omd")
             if kind == "winner" else FollowTheLoser(strategy="pamr", epsilon=0.5))
    X = np.array([[0.2, -0.1], [-0.1, 0.3]])
    before = np.array([0.5, 0.5])
    cumulative = 0.0
    for row in X:
        model.partial_fit(row)
        expected = -np.log(before @ (1 + row))
        assert_allclose(model.loss_, expected)
        cumulative += expected
        before = model.weights_.copy()
    if kind == "winner":
        assert_allclose(model.cumulative_loss_, cumulative)


@pytest.mark.parametrize("strategy", ["eg", "pamr", "olmar", "rmr", "cwmr"])
def test_turnover_uses_fresh_external_holdings_including_cold_start(strategy):
    model = (FollowTheWinner(strategy=strategy, learning_rate=.2)
             if strategy == "eg" else FollowTheLoser(strategy=strategy))
    model.set_params(max_turnover=0.0, previous_weights=[.8, .2])
    model.partial_fit([.2, -.1])
    assert_allclose(model.weights_, [.8, .2], atol=1e-7)
    model.set_params(previous_weights={"a": .2, "b": .8})
    # Use positional holdings here; named alignment has its own test.
    model.set_params(previous_weights=[.2, .8]).partial_fit([-.1, .2])
    assert_allclose(model.weights_, [.2, .8], atol=1e-7)


def test_zero_turnover_does_not_project_actual_holdings_into_feasibility():
    from skfolio.optimization.online._projection import project_with_turnover
    anchor = np.array([.9, .1])
    with pytest.raises(ValueError, match="Infeasible"):
        project_with_turnover([.5, .5], previous_weights=anchor,
                              max_turnover=0, lower=0, upper=.6, budget=1)
    assert_allclose(anchor, [.9, .1])


def test_positive_turnover_cap_rejects_infeasible_funding():
    from skfolio.optimization.online._projection import project_with_turnover
    with pytest.raises(ValueError, match="Infeasible"):
        project_with_turnover([.5, .5], previous_weights=[0, 0],
                              max_turnover=.1, lower=0, upper=1, budget=1)


def test_mixed_eg_target_is_next_engine_reference():
    model = FollowTheWinner(update_mode="omd", learning_rate=2,
                            eg_tilde=True, eg_tilde_alpha=.3)
    model.partial_fit([.2, -.1])
    assert_allclose(model.weights_, model._foco_engine._x_t)


@pytest.mark.parametrize("kind", ["winner", "loser"])
def test_warmup_only_final_target_uses_entry_holdings(kind):
    model = (FollowTheWinner(learning_rate=.5, update_mode="omd")
             if kind == "winner" else FollowTheLoser(strategy="pamr", epsilon=.5))
    model.set_params(max_turnover=0, previous_weights=[.8, .2])
    model.fit(np.array([[.2, -.1], [-.1, .3], [.1, -.1]]))
    assert_allclose(model.initial_weights_, [.5, .5])
    assert not np.allclose(model.all_weights_[1], [.8, .2])
    assert_allclose(model.weights_, [.8, .2], atol=1e-7)
    assert model.n_observations_ == 3
    with pytest.raises(ValueError, match="row-aligned"):
        model.partial_fit([[.1, -.1], [-.1, .1]])


def test_failure_requires_reset_before_continuing():
    model = FollowTheWinner(max_turnover=0, previous_weights=[.9, .1], max_weights=.6)
    with pytest.raises(ValueError, match="Infeasible"):
        model.partial_fit([.1, -.1])
    with pytest.raises(RuntimeError, match="reset"):
        model.partial_fit([0., 0.])
    model.set_params(previous_weights=[.5, .5]).fit([[0., 0.]])
    assert_allclose(model.weights_, [.5, .5])
