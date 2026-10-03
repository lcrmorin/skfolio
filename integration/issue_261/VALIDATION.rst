Local validation (2026-10-03)
=============================

Baseline import
---------------

Carlo's original online suite on current skfolio, before lifecycle adaptation:
647 passed, 1 expected failure. Autograd was available from an isolated temporary
folder for verification tests. Original numerical warnings were not suppressed.

Lifecycle adaptation and dependencies
------------------------------------

* New lifecycle suite without autograd: 50 passed, covering eight configurations
  across both strategy families.
* Combined online, online model-selection, measures and portfolio run:
  2340 passed, 4 failed, 1 expected failure. All four failures were in portfolio
  handling of missing returns for the imported log-wealth measure. The online
  and online model-selection portion passed (852 tests, 1 expected failure).
* After adapting log-wealth to exclude missing periods, the complete measures
  and portfolio suites passed: 1494 passed, including two new regression cases.
  Bankruptcy still produces negative infinity; its NumPy warning is expected.

The resulting targeted suites account for 2346 passing cases and one existing
expected failure. This is not a full-repository test run. The final measure fix
was verified by rerunning its affected suites, not by rerunning every test.

Static checks
-------------

* git diff --check passes.
* Current Ruff rules pass for the new lifecycle tests, modified model-selection
  file and modified measure file/tests. Modified lifecycle files were formatted
  and checked for imports and undefined names.
* A full Ruff inventory of the imported online module reports 191 diagnostics,
  predominantly missing annotations inherited from the older implementation.
  The inventory is a backlog, not a passing lint check.
* Current strict typing and full docstring validation have not been completed.

Remaining parity limits
-----------------------

The original mathematical tests remain, but passing them does not establish
that Carlo's execution, turnover, risk constraints or regret semantics match
current skfolio. The new online_predict comparison covers simple cost-free
one-period execution. See README.rst for the remaining architecture work.
