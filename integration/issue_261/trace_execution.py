"""Read-only probes of execution state at integration commit 7b9076b."""

import json
from unittest.mock import patch

import numpy as np

from skfolio.model_selection import online_predict
from skfolio.optimization.online import FollowTheLoser, FollowTheWinner
from skfolio.optimization.online._projection import AutoProjector
from skfolio.portfolio import Portfolio


def array(x):
    return None if x is None else np.asarray(x).tolist()


report = {}
x = np.array([[0.2, -0.1], [-0.1, 0.2], [0.15, -0.05], [0.01, 0.03]])
m = FollowTheWinner(learning_rate=0.2, update_mode="omd", max_turnover=0.1)
m.partial_fit(x[:1])
old_target = m.weights_.copy()
old_internal = m._current_previous_weights_.copy()
external = np.array([0.9, 0.1])
m.set_params(previous_weights=external)
anchors = []
original_project = AutoProjector.project


def record_project(self, y):
    anchors.append(array(self.config.previous_weights))
    return original_project(self, y)


with patch.object(AutoProjector, "project", record_project):
    m.partial_fit(x[1:2])
report["winner_external_anchor"] = dict(
    supplied=array(external),
    cached_before=array(old_internal),
    old_target=array(old_target),
    projection_anchors=list(anchors),
    cached_after=array(m._current_previous_weights_),
)
assert np.allclose(anchors[-1], old_target)
assert not np.allclose(anchors[-1], external)

m = FollowTheLoser(strategy="pamr", epsilon=0.5, max_turnover=0.1)
m.partial_fit(x[:1])
old_target = m.weights_.copy()
m.set_params(previous_weights=external)
anchors.clear()
with patch.object(AutoProjector, "project", record_project):
    m.partial_fit(x[1:2])
report["loser_external_anchor"] = dict(
    supplied=array(external),
    old_target=array(old_target),
    projection_anchors=list(anchors),
)
assert np.allclose(anchors[-1], old_target)

m = FollowTheWinner(learning_rate=2.0, update_mode="omd").partial_fit(x[:1])
realized = float(-np.log(m.initial_weights_ @ (1 + x[0])))
after_update = float(-np.log(m.weights_ @ (1 + x[0])))
report["winner_loss_timing"] = dict(
    reported=float(m.loss_),
    loss_of_pre_observation_target=realized,
    loss_of_next_target=after_update,
)
assert np.isclose(m.loss_, after_update) and not np.isclose(m.loss_, realized)

m = FollowTheWinner(learning_rate=0.2, management_fees=0.01).partial_fit(x[:1])
p = Portfolio(
    x[:1], weights=m.initial_weights_, management_fees=0.01, weight_drift=True
)
report["management_fee_conventions"] = dict(
    internal_return=float(m.wealth_ - 1), portfolio_return=float(p.returns[0])
)
assert not np.isclose(m.wealth_ - 1, p.returns[0])

events = []
original_update = FollowTheWinner._update_one


def record_update(self, row):
    event = dict(
        observation=self._t,
        public_previous=array(self.previous_weights),
        internal_previous=array(self._current_previous_weights_),
    )
    original_update(self, row)
    event.update(
        target_used=array(self._last_trade_weights_),
        internal_wealth=float(self.wealth_),
    )
    events.append(event)


with patch.object(FollowTheWinner, "_update_one", record_update):
    result = online_predict(
        FollowTheWinner(learning_rate=0.2, portfolio_params={"weight_drift": True}),
        x,
        warmup_size=2,
        test_size=1,
    )
report["online_predict_warmup_and_first_update"] = dict(
    updates=events,
    evaluated_periods=len(result.returns),
    first_portfolio_ending=array(result[0].ending_weights),
)
assert len(events) == 3  # Two warmup rows, then first evaluated observation.
assert np.allclose(events[2]["public_previous"], result[0].ending_weights)
assert not np.allclose(events[2]["public_previous"], events[2]["internal_previous"])

p_drift = Portfolio(x[:1], weights=[0.5, 0.5], weight_drift=True)
p_fixed = Portfolio(x[:1], weights=[0.5, 0.5], weight_drift=False)
report["portfolio_drift_switch"] = dict(
    enabled=array(p_drift.ending_weights), disabled=array(p_fixed.ending_weights)
)
print(json.dumps(report, indent=2))
