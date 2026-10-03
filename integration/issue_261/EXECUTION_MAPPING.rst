Execution mapping after implementation
======================================

The execution changes are implemented locally. EXECUTION_MAPPING_BEFORE.rst and
execution_trace_before.json preserve the original findings at 7b9076b. Current
reproducible assertions are in trace_execution.py and the execution-contract tests.
The historical probe source is available in commit 855e1d6; it is not expected to
pass against the updated implementation.

State ownership
---------------

* The estimator owns weights_, initial_weights_, n_observations_, the previous
  algorithm target (_previous_target_), predictors and the shared OCO engine.
* Adaptive gradient matrices and CWMR state remain algorithm state. They are not
  return covariance estimates and were not moved into priors.
* previous_weights is the external holdings input, resolved afresh at each actual
  decision. The internal _execution_anchor is a validated read-only copy.
* The old cached/simulated holdings fields and local drift/wealth update helpers
  are removed from the online winner/loser lifecycle.
* Portfolio owns evaluated returns, fees, turnover accounting and ending holdings.
  Model selection owns chronological propagation between evaluated periods.
* There are no wealth_ or all_wealth_ attributes on the learning estimators.
  Benchmark estimators retain their separate comparator histories.
* all_weights_ is diagnostic algorithm-target history, not an execution record.
  Set store_weights=False for bounded-memory streaming; True remains the default
  for compatibility with Carlo's original regret/trajectory diagnostics.

Observation and rebalance ordering
---------------------------------

Let b_t be the algorithm target chosen before observation t and h_t the actual
holdings after observation t, before the next rebalance.

1. Portfolio evaluates the deployed target for period t using the chosen execution
   model and supplies its ending holdings to model selection.
2. Model selection sets previous_weights=h_t before the next learning call.
3. partial_fit consumes r_t once. Its update center/gradient uses the intended
   algorithm target b_t, not h_t.
4. Turnover constraints and enabled trading penalties use h_t unchanged. There is
   no second drift of supplied holdings and no strategy-level anchor overwrite.
5. The resulting b_(t+1) is used for the next return period. Portfolio books that
   rebalance once. Online learning does not create trades or realised wealth.

The final turnover projection also runs on loser cold-start/early-return paths.
EG mixing and loser post-processing synchronize their final targets with local
engine state. SWORD's separate expert iterates remain separate.

The full L1 constraint is sum(abs(b_(t+1)-h_t)) <= max_turnover. Zero turnover
returns actual holdings unchanged if feasible; otherwise it raises. Positive-cap
problems that need a solver are solved over the constraint intersection, with
post-solve feasibility checks. Large passive-aggressive candidates are scaled
without changing the objective minimizer to improve numerical conditioning.

Warmup, entry and direct calls
-----------------------------

* fit resets and processes its block as warmup. Intermediate targets are learning
  states with static constraints, without execution-relative caps or penalties.
* The last warmup row computes the first deployable target against entry holdings.
  It is consumed once, not replayed for a separate entry update.
* online_predict identifies its historical warmup block explicitly. Entry parameter
  overrides affect the entry decision and first Portfolio; ordinary settings are
  restored before later decisions. Static/algorithm parameter overrides still apply
  during the warmup call; this is not a general per-row parameter-routing API.
* A configured max_turnover requires explicit previous_weights. Funding a unit
  allocation from zeros under a cap below one is infeasible. Supply compatible
  holdings or explicitly relax entry via entry_rebalancing_params.
* Outside warmup, blocks with turnover constraints or transaction costs are
  rejected: one vector cannot represent every intermediate execution anchor.
  Supply actual holdings and call partial_fit for each rebalance.
* Cost-free unconstrained training blocks remain causal and equivalent to rows.
* Evaluation requires contiguous single-observation windows. Purged and other
  noncontiguous execution schedules are rejected before learning.
* A direct caller owns execution. Repeated partial_fit calls do not imply trades
  and do not automatically drift previous_weights.

Accounting and legacy helpers
-----------------------------

Portfolio's weight_drift setting is authoritative. For held/drifted turnover use
portfolio_params={"weight_drift": True}. Its normal default remains False, which
propagates constant target weights as ending holdings. The port does not silently
change this shared skfolio default.

Management fees are additive per-period fees in both Portfolio accounting and
fee-aware algorithm observations. Reversion predictors observe gross relatives;
when apply_fees_to_phi=True the future fee is subtracted once from the forecast.
A decision penalty is distinct from charging a cost: the former may affect the
chosen target, while Portfolio books the realised cost once. The old switches
that disabled penalties because execution costs were nonzero are removed.

loss_ is pre-update algorithm log loss, including fee-aware returns, and excludes
booked transaction costs. It is not an account-performance measure. Winner's
cumulative_loss_ sums these correctly timed observations.

fit_predict starts evaluation at the explicit initial allocation, without a
historical warmup. predict_online continues on a fitted copy. Both use the same
private model-selection path as online_predict to construct Portfolio objects
and propagate holdings. The public online_predict warmup API is unchanged.

initial_wealth is retained only for source compatibility. An array initializes
proportional algorithm weights; a scalar does not create an account or change
normalized returns. Use Portfolio evaluation and an external capital amount to
scale realised wealth.

Failure handling
----------------

Whole-block input validation occurs before numerical updates. A numerical error,
infeasible decision or invalid candidate marks the stream unusable; partial_fit
and predict require a fresh fit before reuse. This avoids claiming that restoring
a public target would undo already-mutated predictors or adaptive matrices.
The original exception is raised. Transactional rollback and online fallback
recovery are not implemented by this change.

Implementation locations
------------------------

* optimization/online/_base.py: validation, warmup, fresh anchors, diagnostic
  history, failure guard and delegation of legacy evaluation helpers.
* optimization/online/_winner.py, _loser.py and _strategies.py: target state,
  learning updates, correctly timed loss and explicit decision penalties.
* optimization/online/_projection.py: turnover feasibility and stable solver path.
* model_selection/_online/_validation.py: entry/warmup/continuation context,
  chronology checks and the existing common Portfolio accounting loop.

All paths above are below src/skfolio/.

Current constraints and statistical integration
----------------------------------------------

Weight/exposure, budget and named linear constraint expressions are shared with
ConvexOptimization. The OCO engine requests entropy geometry for entropy maps;
bounded-simplex KL projection has a fast path, and richer constraints use CVXPY.
Euclidean maps keep Euclidean projection. Statistical covariance and tracking
constraints can both be active; the short-exposure cap is an upper bound.

A configured prior is cloned and updated one row at a time before producing the
next target. It supplies covariance only when variance_bound is requested. During
learning-only warmup, unavailable covariance does not constrain historical targets.
At an execution decision the covariance must be finite on the complete fixed
universe. y and requested metadata, including sample weights, follow the same row.
No prior is created by default. fit resets both algorithm and fitted prior state.

Input blocks containing missing/non-finite or <= -100% returns are rejected before
processing their valid prefix. Changed feature names/order require a fresh fit.
These are rejection policies, not support for bankrupt or changing-universe streams.
A failing prior update has the same reset-required semantics as a kernel failure.

Remaining independent work
--------------------------

Final public estimator/alias API; native constraints for arbitrary adaptive mirror
maps; more risk expressions, factor and mixed-integer constraints; missing/bankrupt
asset recovery and changing universes; transactional rollback/fallback; row-aligned
anchors for constrained blocks and larger evaluation windows. Strict lint, type and
docstring checks are now covered; see VALIDATION.rst for the final results.
