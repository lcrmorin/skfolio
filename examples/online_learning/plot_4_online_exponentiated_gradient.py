r"""
=============================================
Causal Exponentiated Gradient with Constraints
=============================================

This discussion example reuses the shared online optimization kernels. The public
estimator naming is still under review. EG learns a target without a covariance
estimate; an incremental prior is useful only when a statistical constraint needs
one. Portfolio execution, drift and transaction costs remain in the common
model-selection evaluation path.
"""

# %%
# Deterministic synthetic returns
# ==============================
import numpy as np
import pandas as pd

from skfolio.model_selection import online_predict
from skfolio.moments import EWCovariance, EWMu
from skfolio.optimization.online import FollowTheWinner
from skfolio.prior import EmpiricalPrior

X = pd.DataFrame(
    np.random.default_rng(261).normal(0.0005, 0.015, (60, 4)),
    columns=["asset_a", "asset_b", "asset_c", "asset_d"],
)

# %%
# Learning and execution
# ======================
# At each step the return is observed before the next target is computed.
# Entropy projection preserves EG geometry under the weight bounds. ``fit`` resets
# state, while ``partial_fit`` continues it. FTRL remains the family default, so
# we select OMD explicitly. Holdings at entry are equal weighted here.
model = FollowTheWinner(
    strategy="eg",
    update_mode="omd",
    learning_rate=0.05,
    max_weights=0.4,
    max_turnover=0.1,
    previous_weights=np.full(4, 0.25),
    transaction_costs=0.001,
    portfolio_params={"weight_drift": True},
    store_weights=False,
)
portfolio = online_predict(model, X, warmup_size=10, test_size=1)

# %%
# Optional causal statistics
# ==========================
# This prior receives one row at a time. Its first three observations are a
# statistical warmup. A finite covariance is required at the first execution
# decision; the optimizer never fits the prior on a future block.
prior = EmpiricalPrior(
    mu_estimator=EWMu(min_observations=3),
    covariance_estimator=EWCovariance(min_observations=3),
    max_history=10,
)
constrained_model = FollowTheWinner(
    strategy="eg",
    update_mode="omd",
    learning_rate=0.05,
    max_weights=0.4,
    prior_estimator=prior,
    variance_bound=0.0004,
    portfolio_params={"weight_drift": True},
    store_weights=False,
)
constrained_portfolio = online_predict(
    constrained_model, X, warmup_size=10, test_size=1
)

# %%
# Scope
# =====
# Returns must be finite and greater than -1. Named universe changes can opt into
# ``universe_policy="restart"``; failures can opt into ``recovery="rollback"``.
# Larger ``test_size`` values execute one target per window and learn the completed
# window before the next rebalance. Direct execution-constrained blocks outside
# warmup still require explicit row-aligned holdings.
print("EG evaluation periods:", len(portfolio.returns))
print("EG with causal variance constraint:", len(constrained_portfolio.returns))

# Five-period rebalancing, with holdings drifting between decisions.
window_portfolio = online_predict(model, X, warmup_size=10, test_size=5)
print("EG five-period-window evaluation periods:", len(window_portfolio.returns))
