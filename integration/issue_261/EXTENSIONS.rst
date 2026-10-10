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
fallback. Continuing evaluation supplies new observations; rejected observations are
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
