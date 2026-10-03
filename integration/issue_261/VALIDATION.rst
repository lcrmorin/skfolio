Local validation after execution fixes (2026-10-03)
==================================================

Final relevant suites: 2380 passed, 1 existing expected failure, 46 warnings.
Runtime approximately 36 seconds. Command:

.. code-block:: console

    PYTHONPATH=src:/private/tmp/skfolio-carlo-test-deps \
      /Users/lucasmorin/skfolio/.venv/bin/python -m pytest \
      -q --import-mode=importlib -o addopts='' \
      tests/test_optimization/test_online tests/test_model_selection/test_online \
      tests/test_measures tests/test_portfolio

This is not a full-repository test run. Autograd is only a temporary verification
dependency; analytical operation and the 30 new execution-contract cases pass
without it. The original engine, mirror-map and objective tests remain in the
suite. Legacy tests of estimator-owned wealth were migrated to explicit Portfolio
accounting and initialization tests, rather than skipped or marked expected-fail.

New execution coverage
----------------------

* Pre-observation loss and cumulative loss timing.
* Fresh external anchors across winner, PAMR, OLMAR, RMR and CWMR, including cold starts.
* Zero/positive turnover infeasibility and unchanged actual holdings.
* Final-target synchronization after EG mixing.
* Warmup entry policy, rejection of ambiguous turnover/cost blocks, and no learner wealth.
* Independent manual execution with drift, fees, costs and turnover; online_score parity.
* Initial-period evaluation and additive fee conventions.
* Failure/nonfinite-target guards requiring reset before prediction or continuation.
* Optional target storage, named holdings and read-only anchor copies.
* Entry-only overrides, purged-gap rejection, and online search final-refit behaviour.

Reproducible current probes are in trace_execution.py, with execution_trace.json
recording their output. The previous mapping/trace are separately archived.

Static checks and remaining limits
---------------------------------

* git diff --check passes.
* Current Ruff rules pass for the new execution tests and modified model-selection
  module. Modified source was checked for imports/undefined names and formatted.
* The imported online module still has 189 full-Ruff diagnostics, mainly missing
  annotations from the original implementation. This remains a documented backlog;
  a full lint/type/docstring pass has not been claimed.
* Solver and numerical warnings inherited from the original strategy tests remain.
* Shared constraints/native geometry, priors, general block execution, dynamic
  universes and transactional rollback/fallback are separate unfinished work.

Baseline reference
------------------

Carlo's original import passed 647 tests with one expected failure before adaptation.
The previous lifecycle integration passed 2346 cases across its relevant suites.
The increased current count reflects added execution checks and migration of
obsolete accounting assertions to the current ownership model.
