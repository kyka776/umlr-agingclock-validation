# Decision log

## 2026-07-29 - Project location

The newer user instruction overrides the path in the handoff brief. The project is
an independent nested Git repository at:

`GeroScout/initiatives/04-clock-bias-reference`

Phase 1 files are not modified.

## 2026-07-29 - Gate A changed the deliverable

The current aging-clock manuscript now links an official UMLR repository. The
brief's stop rule therefore prohibits a competing reference package. The deliverable
is narrowed to an independent audit harness, reproducible simulations, a permitted
real-data example, and upstream-ready evidence.

## 2026-07-29 - No post-hoc `OutcomeCalibrator`

Both manuscripts act during model fitting. A convenient post-hoc API would suggest
that an existing clock can be made equivalent to UMLR by fitting on test
chronological ages, which would risk leakage and contradict the method. The audit
instead compares post-hoc methods as explicit baselines.

## 2026-07-29 - Licensing boundary

The official UMLR repository has no license at the audited commit. Its source is
not copied into this project. CI and local audit scripts identify a pinned upstream
commit and may execute a user-fetched copy, while committed code remains original.
Raw external omics data and sample-level derived tables are gitignored.

## 2026-07-29 - Statistical claims

Chronological-age MAE/RMSE, residual-age bias and downstream association recovery
are treated as separate outcomes. A method is not recommended on the basis of one
bias metric, and a negative result remains a valid completion.

## 2026-07-29 - Gate B passed

The independent oracle satisfied both training-region constraints to numerical
tolerance, manual coefficients were recovered, train/test separation was tested,
and the ten frozen scenarios completed for 30 replications each. The open-data
example was allowed only after checksum, shape and repository-license checks.

## 2026-07-29 - Gate C is conditional, not a recommendation

UMLR reduced residual-age slope but increased chronological-age RMSE in eight of
ten scenarios. It also had the highest MAE on the frozen OmniAge holdout.
Accordingly, the project recommends a decision process—external calibration,
multiple diagnostics and downstream validation—not default adoption.

## 2026-07-29 - Upstream contribution route

Because the official source has no license and the gap includes a paper/code
boundary mismatch, an issue is safer than a pull request. A PR would require the
authors to confirm licensing and intended `R1` boundary first.
