Review plan for the Carlo integration branch
===========================================

This local branch builds on Carlo Nicolini's unified OCO engine, mirror maps,
strategies and projector dispatcher. It keeps Portfolio responsible for execution
and adapts the learner to skfolio's incremental estimator and evaluation contracts.
Nothing has been pushed or posted. The public estimator naming decision is open.

Implemented in this pass
------------------------

* KL projection for entropy updates, with a fast bounded-simplex path and a convex
  path for richer intersections; OGD retains Euclidean projection.
* Shared weight/exposure, budget and named linear constraint expressions used by
  both convex and online optimization. Correct short-exposure direction, simultaneous
  variance/tracking constraints, explicit infeasibility and named-input alignment.
* Optional cloned priors, updated one observation at a time; covariance only when
  requested by a risk bound. Metadata, targets and requested sample weights are sliced
  by row. Statistical warmup is allowed only before an execution decision.
* Explicit rejection of missing, non-finite and <= -100% returns; fixed asset names
  and order until a reset. A failed numerical/prior update requires a fresh fit.
* Repository lint/type/docstring cleanup, Python 3.10-compatible regret enums, corrected
  autograd dispatch, development-only gradient-verification dependency and an example.

Reviewable follow-up PR sequence
-------------------------------

1. Shared constraint helpers: review the extraction and convex regression results
   independently. This does not require agreeing on public online estimator names.
2. Online core and skfolio lifecycle: retain Carlo attribution, OMD/FTRL machinery,
   native EG geometry and the execution/turnover ownership tests. Keep this a draft
   while maintainers settle the public API and the remaining geometry boundaries.
3. Optional priors: layer causal routing and warmup semantics on the agreed lifecycle.
   Review the actual EmpiricalPrior/EWMu/EWCovariance integration and failure tests.

The current branch includes all three layers for discussion. These are proposed
review boundaries, not separately published PRs. Do not overwrite the earlier #382
branch with this unrelated integration history without agreeing on the transition.

Still to decide or implement
----------------------------

* Concrete public estimators versus family/alias APIs (deliberately unchanged).
* Native constrained projections for arbitrary adaptive/composite mirror geometries,
  further risk expressions, factor constraints and mixed-integer constraints.
* Dynamic asset universes and investability masks; policies for missing returns and
  bankrupt assets rather than merely rejecting unsupported input.
* Atomic rollback or online fallback when either the prior or numerical kernel fails.
  Current behavior is explicit reset-required failure, not transactional recovery.
* Row-aligned execution anchors for constrained blocks and larger evaluation windows.
* Migration/removal of compatibility names and legacy benchmark accounting helpers.

Draft discussion message
------------------------

I have rebuilt the local integration around Carlo's OCO engine and mirror maps.
The current draft separates algorithm state, optional causal statistical state and
Portfolio execution, and retains both OMD and FTRL.

This pass adds entropy-preserving constrained EG, shared convex/online constraint
expressions, row-by-row priors with metadata routing, and explicit input/failure
policies. It also includes regression tests and a runnable example. I have kept the
existing family interface while we discuss public estimator naming.

I suggest reviewing the shared constraint extraction first, then the online lifecycle
and execution contract, and finally optional priors. Broader adaptive geometries,
dynamic universes, automatic recovery and larger execution windows remain follow-ups.
