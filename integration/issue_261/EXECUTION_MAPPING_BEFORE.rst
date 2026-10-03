Execution ownership and update-order mapping
===========================================

Scope and status
----------------

Mapping of steps 1 and 2: separate algorithm state from execution state and
specify which holdings each update receives. Inspected at integration commit
7b9076b, based on Carlo's source c1bd1f1. This document records current behaviour,
confirmed discrepancies, and a proposed contract. It does not change production
behaviour or claim agreement on the open policy choices.

Source paths below are relative to the worktree. Line references refer to the
inspected commit, not future edits. Reproduce the numerical probes with::

    PYTHONPATH=src /Users/lucasmorin/skfolio/.venv/bin/python \
      integration/issue_261/trace_execution.py

The script asserts the current discrepancies; it is a diagnostic snapshot, not
an acceptance test requiring those discrepancies to remain. Recorded output is
in execution_trace.json. No network, trades or external writes are involved.

Notation and the intended boundary
---------------------------------

* r_t: asset returns observed during period t; x_t = 1 + r_t.
* b_t: algorithm target produced before observing r_t, intended for period t.
* h_t: actual holdings after period t, immediately before the next rebalance.
* b_(t+1): target produced using information through t for the next period.

For a simple long-only, fully invested, cost-free portfolio that actually entered
period t at b_t, drift is h_t = b_t * x_t / dot(b_t, x_t). This formula is only an
illustration; the execution layer owns the definition for other cases.

The algorithm's update center can remain b_t while the turnover constraint is
sum(abs(b_(t+1) - h_t)) <= max_turnover. These are different references. Do not
replace the engine iterate with h_t or drift an h_t supplied by the evaluator
again. The current constraint uses full L1 turnover, without a factor of one half.

Ownership map
-------------

.. list-table:: Current fields and their intended owners
   :header-rows: 1
   :widths: 29 36 35

   * - Current field or component
     - Actual meaning in this branch
     - Proposed treatment
   * - weights_, initial_weights_, n_observations_, _t
     - Current algorithm target, initial reference and observation clock
     - Keep in estimator; none proves a trade occurred.
   * - FirstOrderOCO._x_t, _G_sum, _last_grad, _prev_prediction; map state
     - Algorithm iterate, accumulated gradients, optimistic prediction and adaptive geometry
     - Keep in shared engine. Gradient matrices are not return covariance estimates.
   * - _predictor, _strategy_impl, CWMR _mu/_Sdiag, OLMAR/RMR histories
     - Algorithm forecasting and adaptive state
     - Keep in strategies, even when no trades occur during warmup.
   * - previous_weights
     - Public constructor parameter; also updated by model selection between evaluated periods
     - Keep existing public transport where possible; read freshly at a rebalance boundary.
   * - _configured_previous_weights_
     - Snapshot of previous_weights on first initialization
     - Legacy entry configuration; must not be used as the current execution anchor.
   * - _current_previous_weights_
     - Internally simulated holdings from the previous algorithm target and return
     - Remove as an execution authority. Obtain actual holdings from evaluator or caller.
   * - _last_trade_weights_
     - Snapshot of the old algorithm target before an update
     - Rename/use as previous target if needed; it is not evidence of an executed trade.
   * - ProjectionConfig.previous_weights
     - Mutable reference; often overwritten with the old target by strategy code
     - Supply a single explicit, read-only execution anchor for each constrained decision.
   * - wealth_, all_wealth_, _wealth_initialized
     - Wealth simulated inside learning, including warmup
     - Move realised accounting to Portfolio/MultiPeriodPortfolio and evaluation.
   * - all_weights_
     - Targets simulated before each learning update, including unexecuted warmup targets
     - Remove mandatory storage; optional diagnostic target history must be labelled as such.
   * - loss_, cumulative_loss_
     - Algorithm diagnostics; winner and loser use different timing
     - Define pre-update algorithm loss separately from realised execution performance.
   * - _transaction_costs_arr, _management_fees_arr
     - Used for both algorithm decisions and internal accounting
     - Split decision penalties/fee-aware forecasts from costs booked by Portfolio.
   * - covariance, variance_bound, tracking inputs
     - Legacy statistical inputs to constraints
     - Future optional prior integration; not part of the execution extraction itself.

Source anchors: online/_base.py:92,121,204,246,309; online/_foco.py:151;
online/_winner.py:630,799; online/_loser.py:582,593,656. All online paths above
are under src/skfolio/optimization/.

Current call sequences
----------------------

Winner: src/skfolio/optimization/online/_winner.py:815

1. Ensure weights, objective, projector and engine exist. The projector initially
   receives _current_previous_weights_ from the base.
2. Apply management fees to price relatives.
3. Snapshot the internal holdings as rebalance_from for legacy wealth accounting.
4. When max_turnover is configured, overwrite the projector's previous_weights
   with weights_ (the old target).
5. Compute the gradient at that target and advance the shared OCO engine.
6. Save the old target as _last_trade_weights_; install the new target, including
   optional mixing; perform additional adaptive-state updates.
7. Compute loss_ using the NEW target on the just-observed return, and advance _t.
8. Book simulated wealth with the OLD target and rebalance_from.
9. Drift the old target through the current effective relatives and store that as
   _current_previous_weights_ for the next call.

Loser: src/skfolio/optimization/online/_loser.py:675

1. Initialize components/projector and apply fees to relatives.
2. Snapshot internal previous holdings; save current target as trade_w.
3. Update the OLMAR/RMR predictor using the observation, or use the current
   relatives directly for PAMR/CWMR.
4. If the strategy is ready, run its update. Active projection paths typically
   set projector.config.previous_weights = trade_w.
5. A nonzero transaction-cost adjustment can shrink the candidate toward locally
   drifted trade_w, then reproject using trade_w as the turnover anchor.
6. Install the target. Unlike winner, compute loss from the OLD trade_w.
7. Book simulated wealth, drift trade_w and advance the observation clock.

Strategy sites requiring review: _strategies.py:174,241,312,346,480,531.
Some early-return/cold-start paths skip projection entirely. A frozen algorithm
candidate does not necessarily satisfy an execution-relative turnover constraint;
that constraint needs a consistent decision boundary outside those early returns.

Shared base: src/skfolio/optimization/online/_base.py:309

partial_fit validates a complete block, then repeats the above sequence for every
row and appends histories. There is no distinction between historical warmup,
real-time updates or a block supplied after one actual holding period.

Current evaluator: src/skfolio/model_selection/_online/_validation.py:504,645

1. Apply temporary entry_rebalancing_params BEFORE warmup.
2. Call partial_fit on the complete warmup block.
3. Yield the first test slice, predict with the resulting target, and create its
   Portfolio. Restore temporary entry parameters after this first portfolio.
4. If needs_previous_weights is true, set the estimator's public previous_weights
   to the Portfolio's ending_weights, or its named dictionary equivalent.
5. Resume the generator: consume newly available training rows via partial_fit,
   then predict the next test slice. Thus supplied ending holdings already include
   the evaluated observation that the learner is about to consume.
6. refit_last=True also updates after the final evaluation. It need not execute
   the new final target. Failed/empty portfolios do not advance the propagated anchor.

For portfolio estimators online scoring/search use the same portfolio evaluation
path (_evaluate_online:815). Arbitrary non-portfolio scoring follows another path.

needs_previous_weights is defined in optimization/_base.py:461. It is triggered
by drift-enabled portfolio_params, transaction costs, turnover or relevant
fallbacks. Without it, portfolio previous holdings are propagated after prediction,
which cannot retroactively constrain the optimizer.

Confirmed discrepancies
-----------------------

1. External holdings are not refreshed internally

   _ensure_previous_weights_state (_base.py:121) runs only once. A later
   set_params(previous_weights=...) does not refresh its cached state. Even
   refreshing that cache would be insufficient: winner and loser then overwrite
   the projection anchor with the previous target.

   Probe: an external [0.9, 0.1] anchor produces a winner projection anchored at
   approximately [0.514282, 0.485718], and a PAMR projection anchored at [0.45, 0.55].

2. Warmup bookkeeping is a separate simulated path

   Two warmup rows generate internal wealth 1.098001, before the first evaluated
   portfolio. At the next learning call the evaluator supplies holdings about
   [0.547561, 0.452439], while internal cached holdings are [0.442619, 0.557381].
   The internal value predates that evaluated return; the external value already
   includes it. Their coexistence has to be resolved using explicit timing.

3. Drift depends on the execution model

   Portfolio defaults to weight_drift=False, which returns the target as
   ending_weights. For [0.5, 0.5] and returns [0.2, -0.1], enabling drift produces
   [0.571429, 0.428571]. One-observation test windows do not remove this distinction.
   Carlo always simulates drift internally. For the intended held-weight semantics,
   the proposal is explicit weight_drift=True; otherwise honour and document the
   selected constant-weight execution model.

4. Winner diagnostic loss is evaluated after the update

   With the probe's EG configuration, loss_ is -0.087762, while the pre-observation
   target's loss is -0.048790. This does not show that its gradient update leaks
   future data; it shows that this diagnostic cannot be labelled realised loss.

5. Fee conventions differ

   Carlo multiplies each relative by (1-fee), whereas Portfolio subtracts the
   weighted fee from portfolio returns. The probe gives 3.95% versus 4.00% for the
   same pre-update target. Moving bookkeeping must not silently preserve two
   inconsistent performance series. Decisions about fee-adjusted gradients and
   predictors are separate from choosing the accounting authority.

6. Turnover projection feasibility needs explicit failure handling

   The fast projector's max_turnover<=0 branch projects the anchor into static
   constraints, potentially moving it even though zero turnover was requested.
   Its positive-cap path uses ten iterations without a final feasibility error.
   An execution anchor can be outside current bounds. Keep it unchanged and
   report infeasibility rather than modifying holdings to manufacture feasibility.
   This is a source inspection finding; not exercised by the numerical probes.

7. Engine iterate versus final target needs a consistency decision

   EG-tilde mixes the returned candidate after the engine has committed _x_t.
   Some loser post-processing similarly changes the final target after an engine
   step. Do not assume weights_ and engine._x_t are interchangeable. Decide, per
   strategy, whether the next update should use the final constrained/mixed target
   or an intentionally separate internal iterate. Preserve intentional mathematics;
   do not synchronise these fields blindly during execution extraction.

Proposed contract for the port
-----------------------------

Normal one-period evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. The evaluator deploys b_t and computes period t returns/costs and ending holdings.
2. It supplies h_t before the learner consumes r_t, as the current evaluator already
   does via previous_weights when propagation is enabled.
3. The learner consumes r_t once, updating its optional prior and algorithm state.
4. It solves for b_(t+1), with algorithm center b_t and execution anchor h_t.
   No strategy may replace h_t with b_t or drift h_t a second time.
5. The next Portfolio books the rebalance from h_t to b_(t+1) and the next return.
   The optimizer stores a target, not executed trades or realised wealth.

Keep previous_weights as the existing external parameter initially. Internally,
resolve a copied/validated execution anchor at the decision boundary and pass it
explicitly to constraints and any intentional trading penalty. This proposal
requires a way to identify which rows are decisions, not a second mutable holdings
cache. Exact private signature and metadata-routing representation remain open.

Warmup and entry
~~~~~~~~~~~~~~~~

Recommended policy, requiring maintainer agreement:

* Warmup learns causally using algorithm targets. It produces no actual trades,
  transaction costs or realised wealth. Static algorithm constraints still apply.
* Execution-relative constraints and penalties are inactive on intermediate
  historical targets, because no corresponding actual holdings exist.
* The last warmup observation produces the first deployable target constrained
  against explicitly supplied entry holdings. Consume that observation/gradient
  once; do not run a second update merely to attach an execution anchor.
* Apply entry_rebalancing_params to that entry decision/accounting, not to every
  hypothetical rebalance during warmup. Preserve initial_weights_ as algorithm
  initialization, separate from the actual entry holdings.
* Decide the missing-entry-holdings policy explicitly. Portfolio currently treats
  None as zeros; Carlo's wealth path skips the initial trading cost for None.
  A unit-budget portfolio cannot be funded from zeros with an L1 turnover cap
  below one. Require compatible entry holdings, an explicitly relaxed entry cap,
  or an error. Do not quietly invent a fully invested initial account.

This warmup policy changes the learned path relative to Carlo's simulated
turnover-constrained warmup. It is a design choice, not a parity-preserving rename.
An alternative is an explicitly named virtual execution warmup mode, whose state
must remain separate from the actual account. It adds semantics and testing and
is not the recommended first integration.

Blocks and delayed updates
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Ordinary cost-free training blocks remain equivalent to sequential learning.
* A bare block is not evidence that its intermediate targets were executed.
* One end-of-block holdings vector cannot serve as the actual anchor for every
  historical row. It must not be broadcast or drifted repeatedly inside the loop.
* A possible future contract accepts row-aligned decision markers and holdings,
  or distinguishes learning-only rows from a final deployable decision. It must
  also specify whether observed gradients refer to virtual targets or actual
  deployed decisions when they differ.
* First implementation: support warmup and normal one-period decisions explicitly;
  reject other execution-constrained blocks without unambiguous context. Do not
  silently disable requested constraints. General delayed/block execution remains
  outside scope while test_size=1 is required.
* Review purged/calendar schedules too: a one-row test slice does not prove that
  every newly consumed training row corresponds to an executed allocation.
* A caller outside model selection owns execution. With constraints enabled it
  must supply fresh, correctly aligned holdings; calling partial_fit does not trade.

Costs, penalties and failures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Book actual costs once in Portfolio. A trading-cost term in an optimization
objective is a decision incentive, not a second booked cost, so both can exist.
Review the imported 'double counting' switches and soft-threshold adjustments
under that distinction instead of removing all cost-aware algorithm logic.

A failed solve must not advance execution holdings or book a trade. Algorithm,
prior and adaptive-map state may already have mutated, so the initial safe policy
can require a reset rather than claim transactional rollback. Recovery and fallback
semantics require a separate design. The existing evaluator's preservation of the
last successful holdings does not by itself specify what happened in a failed period.

Implementation sites and order
------------------------------

A. Add acceptance tests for external-anchor freshness, warmup entry and row ordering.
   Keep the source probes as historical evidence, not expected final behaviour.
B. Base: replace internal holdings authority with per-decision execution context;
   keep previous target/engine state separate. Resolve warmup and block policies
   before removing their simulated path.
C. Winner, loser and strategies: pass the same resolved anchor through all projection
   and trading-penalty paths; cover cold-start/early-return paths. Separate reported
   algorithm loss from realised performance. Review final-target/engine synchronization.
D. Evaluator: supply entry versus ordinary decision context, retain actual ending
   holdings propagation, and make the drift choice explicit. Check entry overrides,
   named assets, final refit and schedule gaps.
E. Remove/migrate _update_wealth, both local drift helpers and mandatory histories.
   Replace legacy fit_predict/predict_online execution with common evaluation where
   their intended semantics match; preserve continuation semantics explicitly.
F. Confirm accounting and failure behaviour, then implement shared geometry-aware
   constraints as the next workstream. Do not claim this mapping completes that work.

Acceptance checks before execution extraction is complete
---------------------------------------------------------

* Distinct b_t and h_t: a constrained candidate is anchored to h_t, while the
  algorithm's gradient/center still uses its intended b_t.
* Changing only externally supplied holdings changes a turnover-constrained
  decision; the next call must not use a cached previous value.
* No double drift when h_t already includes r_t.
* Winner and loser consume each observation once; predictor warmup clocks survive.
* Warmup contributes zero realised trades/fees/wealth; entry cost and turnover
  are applied exactly once with declared initial holdings.
* A missing or infeasible required anchor raises a clear error, including a zero
  turnover cap incompatible with static bounds.
* Independent manual one-period execution matches online_predict, online_score
  and search for both winner and loser, with costs and explicit drift enabled.
* Named asset alignment is preserved; returned ending holdings are not mutated.
* Legacy algorithm gradients/iterates remain equivalent in the unconstrained,
  cost-free case; constrained differences are documented as intentional.
* Unsupported constrained blocks and schedule gaps fail explicitly; unexecuted
  intermediate targets never appear as real trades.
* refit_last learns the last return without adding a fictitious evaluated period.
* Failed updates neither fabricate execution nor silently reuse corrupted state.
