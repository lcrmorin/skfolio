Online integration extensions
=============================

Transactional recovery
----------------------

Both families accept recovery="raise" (the existing default), "rollback", or
"previous_weights". The opt-in modes snapshot the full learned state before a
partial_fit call, including the prior, adaptive matrices, predictors and diagnostic
history, while preserving constructor parameter objects.

With rollback, any failure restores that state and raises the original error.
No prefix of a failed block is retained. Correct the cause and retry the whole
block, without calling fit. This also protects prior state after a failed update.

With previous_weights, allocation failures on an initialized stream additionally
return the explicitly supplied execution holdings. The fallback makes no new trade
and does not project holdings into the failed feasible set. Input and prior failures
still raise. An uninitialized estimator or missing/invalid holdings cannot use this
fallback. A failed membership change also raises after restoring the old universe,
since its holdings cannot be returned as targets for a different asset list.
Continuing evaluation supplies new observations; rejected observations are
not automatically replayed.

Diagnostics are last_update_consumed_, recovery_error_, n_recoveries_ and fallback_.
A successful call clears the latest error/fallback and marks its block consumed.
Recovery copies learned state, so use store_weights=False and bounded prior history
for long streams. The default raise policy retains the previous reset-required
behavior and incurs no snapshot cost. Recovery cannot undo external side effects
inside user-provided estimators.

Validation: injected post-update failures verify that both estimator families restore
priors, targets, numerical state and complete blocks; corrected retries agree with
uninterrupted reference runs. Holdings fallback and input-error rejection are tested.

Named universe changes
----------------------

``universe_policy="restart"`` accepts DataFrame observations with unique string
asset names. Reordering columns preserves learning state. Membership changes
restart the kernel, predictor and optional prior; weights and observation history
then describe the current universe only. ``n_universe_restarts_`` and
``previous_universe_`` report transitions. Defaults remain strict.

Use scalar or named constraints and named execution holdings. Positional asset
parameters and matrices cannot be safely remapped and are rejected on a membership
change. Liquidated holdings outside the new universe consume the turnover budget;
Portfolio accounts for their transaction costs. Use ``recovery="rollback"`` to
restore the old universe if the new allocation fails. A restarted covariance prior
may need a new historical warmup before it can supply a finite risk estimate.

This supports explicit universe changes between calls. It does not infer membership
from missing returns or AssetPanel masks, nor transport adaptive matrices or prior
statistics between universes.

Larger evaluation windows
-------------------------

``online_predict`` and ``online_score`` now accept contiguous multi-period windows
for the online families. One target is executed per window. Once the window closes,
its observations update the learner causally and only the final update makes the
next executable decision. Intermediate learner targets are diagnostic, not trades.
The optional prior updates on every observation, with finite covariance required
at the final decision. OnlineGridSearch applies the same rule during final refit.

``weight_drift=True`` gives buy-and-hold execution inside a window. With the default
False, Portfolio retains its constant-weight return convention. End-of-window
holdings anchor the next allocation, and transaction costs are charged by Portfolio.
Purged/gapped execution remains rejected. Direct constrained partial_fit blocks
still require explicit execution information and are rejected outside warmup.

Validation of the three extensions
----------------------------------

The full repository suite passed 10,804 tests, with one skipped and one expected
failure. After the final universe validation/recovery fixes, all 924 affected online
optimization and online model-selection tests passed again, with one expected
failure. The runnable example produced 50 periods for daily plain EG, daily causal
variance-constrained EG and five-period-window EG.

Ruff lint and formatting checks pass (602 files), and changed-module numpydoc
presence checks pass. Type checking reports only the existing upstream
``AssetPanel.fields`` read-only override error in
``src/skfolio/containers/_asset_panel/_view.py:233``; no extension diagnostics remain.
No commits have been pushed.
