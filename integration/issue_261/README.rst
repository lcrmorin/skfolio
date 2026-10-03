Carlo's online framework: local integration
=========================================

Branch: codex/261-carlo-integration. This remains local, with nothing pushed.
It ports Carlo's shared machinery instead of recreating standalone algorithms.

Provenance
----------

* Current-main integration base: skfolio/skfolio at 5b657bd.
* Carlo source: universal_portfolio at c1bd1f168cf7d660c81f74710aadd9941e64a2c7.
* Original history is preserved on codex/261-carlo-source.
* Attributed baseline import: 3c4cce7. Original headers and numerical tests remain.
* Legacy datasets and model-selection code were excluded. The log-wealth measure
  dependency was imported and adapted to current missing-period handling.
* Earlier lifecycle integration: 646f9cb; log-wealth compatibility: 5677491.

Execution changes by commit
---------------------------

* 89e6f1a: report loss using the target chosen before the observation.
* cd4dcd7: use fresh external holdings for turnover; reject infeasible projections.
* 5a8e2b2: distinguish warmup from streaming decisions and reject ambiguous blocks.
* 9aa062f: delegate realised accounting to Portfolio; align management fees and
  preserve decision penalties separately from booked costs.
* 706b033: migrate legacy execution assertions to portfolio-owned accounting.
* 96dae23: entry/gap/failure guards, named holdings, final-refit coverage and optional
  target storage. Rename the previous algorithm target to avoid implying a trade.

See EXECUTION_MAPPING.rst for the current contract. The pre-change analysis and
trace remain in EXECUTION_MAPPING_BEFORE.rst and execution_trace_before.json.
These are historical evidence, not descriptions of current behaviour.

Usage
-----

From this worktree::

    PYTHONPATH=src /Users/lucasmorin/skfolio/.venv/bin/python - <<'PY'
    import numpy as np
    from skfolio.model_selection import online_predict
    from skfolio.optimization.online import FollowTheWinner

    X = np.random.default_rng(261).normal(0.001, 0.02, (40, 4))
    model = FollowTheWinner(
        strategy="eg", update_mode="omd", learning_rate=0.05,
        previous_weights=np.full(4, 0.25), max_turnover=0.1,
        portfolio_params={"weight_drift": True}, store_weights=False,
    )
    result = online_predict(model, X, warmup_size=10, test_size=1)
    print(result.returns)
    PY

The generic winner default remains Carlo's FTRL configuration. Set OMD explicitly
when comparing with the original #382 draft. Entropy updates now use KL projection
for bounds and richer convex intersections. Native constrained geometry for all
adaptive/composite maps remains a separate extension.

For a unit-budget entry from cash under a restrictive cap, explicitly relax entry
with entry_rebalancing_params={"max_turnover": None}, or supply feasible holdings.
A one-period evaluation still needs weight_drift=True for held-weight semantics.

Learning produces targets, not trades. fit resets; partial_fit continues. Direct
streaming callers supply holdings at each rebalance. Cost/turnover-aware blocks
are rejected outside explicit warmup; there is no implicit execution simulator.

Validation
----------

See VALIDATION.rst for counts and remaining checks. The execution probes run without
autograd::

    PYTHONPATH=src /Users/lucasmorin/skfolio/.venv/bin/python \
      integration/issue_261/trace_execution.py

The original gradient-verification tests need autograd, installed for this session
only in /private/tmp/skfolio-carlo-test-deps. The relevant test command is::

    PYTHONPATH=src:/private/tmp/skfolio-carlo-test-deps \
      /Users/lucasmorin/skfolio/.venv/bin/python -m pytest \
      -q --import-mode=importlib -o addopts='' \
      tests/test_optimization/test_online tests/test_model_selection/test_online \
      tests/test_measures tests/test_portfolio

Architecture discussion sources
-------------------------------

* https://github.com/skfolio/skfolio/issues/261#issuecomment-5441142865
* https://github.com/skfolio/skfolio/issues/261#issuecomment-5686429395
* https://github.com/skfolio/skfolio/pull/382#issuecomment-5887036683

Current integration additions
-----------------------------

* d40ae7f: entropy geometry, shared budget/exposure constraints and projection fixes.
* bc1a3a3: shared named/group expressions and projection dispatch cleanup.
* ef3cc34: cloned causal priors, metadata routing, covariance warmup and named inputs.
* 15b9afb: unsupported return/universe rejection with regression coverage.

The final validation/compatibility and documentation commits follow these changes.
See REVIEW_PLAN.rst for proposed PR boundaries and a discussion message. The example
at examples/online_learning/plot_4_online_exponentiated_gradient.py runs without
network access or autograd. Both the plain EG and covariance-constrained evaluation
produce 50 out-of-sample periods.

Supported statistical integration is optional incremental priors, with covariance
consumed by a variance bound. NaN covariance is allowed only during learning-only
warmup. Metadata, y and requested sample weights are sliced by observation. All
execution decisions requiring covariance must see a finite matrix. A missing prior
remains absent; the algorithm's adaptive matrices are never treated as covariance.

Returns must be finite and strictly greater than -1. Asset names/order are fixed
until fit resets the stream. Numerical or prior failures require fit before reuse;
there is no automatic rollback or fallback. Bigger execution windows and constrained
streaming blocks remain outside the supported contract. The public family API has
not been redesigned.
