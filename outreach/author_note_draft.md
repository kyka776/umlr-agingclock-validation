# Author email draft

**To:** ShuoChen@som.umaryland.edu

**Cc:** Hwiyoung.Lee@som.umaryland.edu

**Subject:** Independent UMLR aging-clock audit and a paper/code boundary question

Dear Dr. Chen and Dr. Lee,

I read your aging-clock preprint and the Outcome-Calibrated Regression manuscript
and built a small independent audit of the UMLR equations:

https://github.com/kyka776/umlr-agingclock-validation

The repository does not redistribute or present a replacement for your R code.
It contains an independent constrained numerical oracle, leakage tests, ten
pre-specified simulations, and one small open-data holdout.

The simulations reproduce the intended reduction in residual-age slope and show
the associated trade-offs in chronological-age RMSE and downstream association
recovery. The result is conditional rather than uniformly positive: UMLR raised
RMSE in eight of ten scenarios and did not improve the small open holdout.

I also found one boundary difference I would appreciate your guidance on. The
preprint defines the lower region as `Y_i <= mean(Y)`, while the audited
`Lasso_UP.R` commit uses `y < mean.y`, excluding observations exactly at the
mean. The same source has no explicit constant-outcome guard or solver-status /
constraint-residual check. I opened a focused issue with a pinned characterization
and tests:

https://github.com/hwiyoungstat/UMLR/issues/1

Could you confirm the intended lower-region boundary? If you would welcome a
tested patch or example, could you also add or clarify the repository license?
I would be happy to adapt the audit to your intended specification.

Best regards,

kyka776
