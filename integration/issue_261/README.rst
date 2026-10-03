Carlo's online framework: local integration
=========================================

This is a working integration branch, not a merge-ready replacement for PR #382.
The algorithms are imported from Carlo Nicolini's implementation, not recreated
from the standalone EG draft.

Provenance
----------

* Integration base: skfolio/skfolio main at 5b657bd.
* Source: CarloNicolini/skfolio, universal_portfolio,
  c1bd1f168cf7d660c81f74710aadd9941e64a2c7.
* Local source branch: codex/261-carlo-source (unchanged original history).
* Working branch: codex/261-carlo-integration.
* Baseline import commit: 3c4cce7. Original source headers and tests are retained.
* The source's datasets and obsolete model-selection implementation are excluded.
  The log-wealth measure and portfolio attribute are included as dependencies.

Discussion informing this port:

* https://github.com/skfolio/skfolio/issues/261#issuecomment-5441142865
* https://github.com/skfolio/skfolio/issues/261#issuecomment-5686429395
* https://github.com/skfolio/skfolio/pull/382#issuecomment-5887036683

First adaptation
----------------

* Rename the shared base to BaseOnlineOptimization, with a temporary old-name
  alias. Keep FollowTheWinner and FollowTheLoser and their engine/strategy layers.
* Centralize validation and causal block iteration in partial_fit. Both families
  accept a vector, one-row matrix, or block. Preserve DataFrame feature names and
  reject feature changes during a stream; a fresh fit can use a new universe.
* fit always resets. warm_start is retained only for source compatibility and
  no longer changes fit semantics. Continue a stream through partial_fit.
* Add initial_weights_ and n_observations_. Targets after row t are used at t+1.
* Clear cached objectives and adaptive components on reset.
* Adapt the legacy prediction helper to continue through partial_fit on a copy.
* Use current online_predict with single-observation evaluation windows; reject
  larger actual test splits before warmup. This is a provisional capability flag.
* Make autograd optional for analytical operation. Its verification mode remains
  available when installed, without adding a project runtime dependency.

The shared engine, mirror maps, strategy updates and projection mathematics are
unchanged. In particular, FollowTheWinner still defaults to FTRL, not the OMD
setting from #382. Set update_mode="omd" explicitly when comparing with that draft.

Local smoke example
-------------------

From this worktree, with the existing project environment::

    PYTHONPATH=src /Users/lucasmorin/skfolio/.venv/bin/python - <<'PY'
    import numpy as np
    from skfolio.model_selection import online_predict
    from skfolio.optimization.online import FollowTheWinner

    X = np.random.default_rng(261).normal(0.001, 0.02, (40, 4))
    model = FollowTheWinner(strategy="eg", update_mode="omd", learning_rate=0.05)
    model.partial_fit(X[:10]).partial_fit(X[10:20])
    print(model.n_observations_, model.weights_)
    prediction = online_predict(model, X, warmup_size=10, test_size=1)
    print(prediction.returns)
    PY

Remaining integration boundaries
--------------------------------

These are unresolved, not features completed by the lifecycle adaptation:

1. Execution and turnover: Carlo's internal wealth, drift, fees and trajectory
   bookkeeping remain. Their warmup trajectories are simulated, not actual
   execution. Neither turnover anchors nor fee semantics have been aligned with
   the current model-selection/Portfolio layer. Do not treat constrained or
   cost-aware online evaluation as validated. Define observation, prior, update
   and rebalance ordering before extracting this code.
2. Shared constraints: AutoProjector still uses its original projection rules.
   Extract/reuse current ConvexOptimization constraint expressions and choose the
   objective using the algorithm's geometry. In particular, bounded EG is not yet
   the KL-constrained update from #382. Do not silently equate the two.
3. Optional priors: raw covariance/tracking inputs are still legacy parameters.
   Replace them with optional causal priors and metadata routing where appropriate.
   Gradient-based adaptive matrices remain algorithm state, never prior estimates.
4. Public API: discuss thin, explicitly parameterized sklearn estimators versus
   aliases, and how to expose generic compositions. No top-level public exports
   or final concrete-estimator interface are added in this step.
5. Robustness: decide missing/bankrupt returns (legacy clipping remains), dynamic
   universes, numerical failure recovery, fallback and partial-block semantics.
   Numerical failures are not transactional; reset with fit before reuse after
   such a failure. Whole-block finite-input validation does precede learning.
6. Memory: legacy trajectories are retained and appended during partial_fit for
   parity checks. This is not a bounded-memory production stream. Move execution
   history to the evaluation layer and decide on optional diagnostic storage.
7. Validation and packaging: migrate Carlo's documentation/examples, add current
   estimator checks and metadata tests, remove optional verification code from
   production if agreed, and resolve imported typing/lint/docstring diagnostics.
   No datasets or autograd runtime dependency are required for the port.

Proposed PR boundaries after integration review
----------------------------------------------

* Shared constraint extraction can be reviewed independently where feasible.
* Port shared numerical machinery, lifecycle and concrete API together as a
  coherent first integration; retain coverage of both winner and loser families.
* Execution/turnover and causal-prior integration depend on the agreed contracts
  and shared constraints. Keep these dependency relationships explicit.
* Additional public strategies/compositions and larger evaluation windows follow
  after parity and execution semantics are established.

Validation commands
-------------------

The original tests use relative imports, so use pytest's importlib mode.
The analytical lifecycle tests run without autograd::

    PYTHONPATH=src /Users/lucasmorin/skfolio/.venv/bin/python -m pytest \
      -q --import-mode=importlib -o addopts='' \
      tests/test_optimization/test_online/test_v1_lifecycle.py

For the full original gradient-verification suite, install autograd separately.
For this session it was installed only in /private/tmp/skfolio-carlo-test-deps::

    PYTHONPATH=src:/private/tmp/skfolio-carlo-test-deps \
      /Users/lucasmorin/skfolio/.venv/bin/python -m pytest \
      -q --import-mode=importlib -o addopts='' \
      tests/test_optimization/test_online tests/test_model_selection/test_online \
      tests/test_measures tests/test_portfolio

Baseline: 647 passed, 1 expected failure before lifecycle edits. The 50 new
lifecycle cases pass without autograd. Final integration results are recorded
in VALIDATION.rst. This branch has not been pushed.
