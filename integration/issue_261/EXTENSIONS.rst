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
