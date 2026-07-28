# Validation report

**Status:** ready to share as an independent audit; not ready to present as an
official UMLR implementation or a clinical method.

## Checks passed

- exact manual coefficient recovery and two-region constraint checks;
- deterministic output, scalar/vector agreement and pickle round trip;
- explicit invalid, missing, constant and extrapolation behavior;
- train/test separation for model fit, correction, scaling and feature selection;
- 30 × 10 frozen simulations with identical method splits;
- pinned open-data checksum, 56 × 2,974 shape check and aggregate-only output;
- executed notebook with four executed code cells and zero errors;
- static visual review of both figures;
- package build, clean-wheel installation, tests and style checks.

## Interpretation QA

- bias, prediction error and downstream association are reported separately;
- negative and unstable cases are retained;
- no sample-level omics, identifiers, controlled-access data or official source
  code are included;
- no clinical claims are made;
- the independent oracle is labeled as such throughout.

## Remaining external uncertainties

- only the authors can confirm whether `R1` should use the paper's `<=` boundary
  or the code's strict `<` boundary;
- only the repository owner can add a license that permits a patch or derivative
  distribution;
- controlled UK Biobank and Framingham results cannot be independently
  reproduced without authorized cohort access;
- author review is needed before describing this as compatible with the intended
  solver implementation.
