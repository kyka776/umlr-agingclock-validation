# v0.1.0 — independent UMLR aging-clock audit

First reproducible release of an independent, leakage-safe audit of the UMLR
aging-clock constraints.

Highlights:

- original SciPy numerical oracle for the two-region equations;
- 10 frozen simulation scenarios × 30 replications × 4 methods;
- aggregate-only open OmniAge holdout demonstration;
- 26 tests for equations, known simulation behavior, train/test separation,
  edge cases, serialization and reproducibility;
- bilingual summary, two reviewed figures and an executed notebook;
- pinned characterization of the official source without redistribution.

The release is an audit, not an official UMLR implementation or clinical tool.
The main conclusion is conditional: residual-age bias decreases, but prediction
error can increase materially.
