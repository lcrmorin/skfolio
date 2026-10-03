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
