Latest refresh: 2026-10-10
==========================

The branch now includes upstream main at 33923e0 (v1.8.0). The refreshed full suite
passes 10790 tests, with 1 skipped and 1 expected failure. Ruff and formatting pass.
The latest ty check has one independently verified upstream AssetPanel diagnostic.
See REFRESH_2026_10_10.rst for the full record and reproduction commands.

The October 3 results below are historical and do not describe the latest checker
status or current upstream baseline.

Local integration validation (2026-10-03)
=======================================

Full repository suite: **9260 passed, 1 skipped, 1 existing expected failure,
66 warnings**, in approximately 152 seconds with two workers.

.. code-block:: console

    PYTHONPATH=src:/private/tmp/skfolio-carlo-test-deps \
      /Users/lucasmorin/skfolio/.venv/bin/python -m pytest \
      -q -n 2 --dist loadfile --import-mode=importlib -o addopts='' tests

The final added online-evaluation metadata regression was checked separately after
full-suite collection: all **8 causal-prior tests passed**. No production code changed
after the full suite started. The mirror-map suite passed all **67 tests** after
correcting the domain of a randomized log-barrier round-trip test.

An earlier full run had one failure because additive test noise pushed a negative
log-barrier dual coordinate above zero, outside the conjugate domain. The test now
uses seeded multiplicative perturbations in the negative orthant. No tolerance was
loosened and no failing test was skipped.

Static checks
-------------

* Full repository Ruff check: pass.
* Full repository Ruff format check: pass (577 Python/config files).
* Full source ty check with the workspace Python environment: pass.
* Numpydoc GL08 validation on all 23 changed source files: pass. Existing warnings
  about nonstandard section headings remain non-fatal.
* git diff --check: pass.

Autograd is still optional at runtime. It is now included in the development group
so CI can run Carlo's analytical/autograd parity tests. The local test backend was
installed only under /private/tmp/skfolio-carlo-test-deps. Typing, the new lifecycle
and prior tests, and the example also work without that optional backend.

Coverage of the integration contract
-----------------------------------

* Pre-observation loss; next-period targets; fit reset and causal block equivalence.
* Fresh external turnover anchors and solver feasibility, including zero turnover.
* Shared budget/exposure/group expressions and unchanged convex optimizer behavior.
* Entropy projection preserves free-coordinate ratios; fast and convex KL solutions
  agree; short exposure and simultaneous variance/tracking constraints are enforced.
* Named bounds/groups are aligned to DataFrame columns, while changed column order
  or asset universes are rejected until reset.
* Optional priors are cloned and updated one row at a time. Metadata, targets and
  requested sample weights are aligned through fit and online evaluation.
* Real EmpiricalPrior/EWMu/EWCovariance warmup, finite covariance at execution time,
  no default prior, conflicting covariance sources and reset after prior failure.
* Missing/non-finite/bankrupt returns are rejected before consuming a valid prefix.
* Common Portfolio accounting, drift/fees/costs, explicit initial allocation,
  optional diagnostic history, entry overrides, chronological evaluation and search.

Runnable example
----------------

examples/online_learning/plot_4_online_exponentiated_gradient.py runs without network
access or autograd. Plain EG and EG with a causal variance constraint each produce
50 evaluation periods. A covariance-repair warning during small-sample warmup is
expected. The current execution trace is in execution_trace.json.

Review limits
-------------

This remains a local discussion integration. Passing the suite does not imply that
arbitrary adaptive mirror geometries, dynamic universes, transaction rollback,
online fallback or larger execution windows are implemented. See REVIEW_PLAN.rst
for explicit boundaries and proposed review splits. No PR or comment was published.
