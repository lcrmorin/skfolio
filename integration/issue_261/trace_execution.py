"""Reproduce the current execution contract; the before snapshot is archived."""

import json

import numpy as np

from skfolio.model_selection import online_predict
from skfolio.optimization.online import FollowTheLoser, FollowTheWinner

report = {}
for name, model in [
    ("winner", FollowTheWinner(learning_rate=0.2, update_mode="omd")),
    ("loser", FollowTheLoser(strategy="pamr", epsilon=0.5)),
]:
    model.set_params(max_turnover=0, previous_weights=[0.8, 0.2])
    model.partial_fit([0.2, -0.1])
    first = model.weights_.copy()
    model.set_params(previous_weights=[0.2, 0.8]).partial_fit([-0.1, 0.2])
    np.testing.assert_allclose(first, [0.8, 0.2], atol=1e-7)
    np.testing.assert_allclose(model.weights_, [0.2, 0.8], atol=1e-7)
    report[name] = {
        "first_external_anchor_target": first.tolist(),
        "refreshed_external_anchor_target": model.weights_.tolist(),
        "optimizer_books_wealth": hasattr(model, "wealth_"),
    }
    assert not hasattr(model, "wealth_")

model = FollowTheWinner(learning_rate=2, update_mode="omd")
model.partial_fit([0.2, -0.1])
expected = -np.log(1.05)
np.testing.assert_allclose(model.loss_, expected)
report["pre_observation_loss"] = float(model.loss_)

model = FollowTheWinner(management_fees=0.01, transaction_costs=0.002)
path = model.fit_predict(np.array([[0.2, -0.1], [-0.1, 0.2]]))
np.testing.assert_allclose(path.returns[0], 0.05 - 0.01 - 0.002)
report["first_period_net_return"] = float(path.returns[0])

X = np.array([[0.2, -0.1], [-0.1, 0.2], [0.1, 0.2], [0.2, -0.1], [0.1, -0.1]])
model = FollowTheWinner(
    learning_rate=0.5,
    max_turnover=0,
    previous_weights=[0.0, 0.0],
    portfolio_params={"weight_drift": True},
)
path = online_predict(
    model,
    X,
    warmup_size=2,
    test_size=1,
    entry_rebalancing_params={"max_turnover": None},
)
for previous, current in zip(list(path)[:-1], list(path)[1:], strict=True):
    np.testing.assert_allclose(current.weights, previous.ending_weights, atol=1e-7)
report["entry_override"] = {
    "warmup_rows": 2,
    "evaluated_periods": len(path.returns),
    "later_turnover": [
        float(np.abs(p.weights - p.previous_weights).sum()) for p in list(path)[1:]
    ],
}
print(json.dumps(report, indent=2))
