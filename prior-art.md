# Prior-art audit

**Audit date:** 2026-07-29

**Gate:** A - novelty of implementation

**Verdict:** GO only as an independent validation and upstream-gap project.

## Controlling method sources

1. Hwiyoung Lee et al. *Trustworthy ML/AI for Aging Clocks: Preventing
   Systematic Prediction Bias in Biological Age Estimation*. bioRxiv, version
   posted 2026-06-01. DOI:
   <https://doi.org/10.64898/2026.05.27.728155>. The official PDF is 26 pages,
   CC BY 4.0, and was read in full.
2. Hwiyoung Lee and Shuo Chen. *Outcome-Calibrated Regression and Predicted
   Outcome-Based Inference*. arXiv:2605.29255v1, submitted 2026-05-28:
   <https://arxiv.org/abs/2605.29255>. The 12-page paper was read in full.
3. Lee and Chen. *Systematic bias of machine learning regression models and
   correction*. IEEE TPAMI 47, 4974-4983 (2025), cited by both new manuscripts.

## What the method is

The aging-clock manuscript proposes UMLR: fit a regression model from biomarkers
to chronological age while imposing two mean-anchoring equality constraints on
complementary outcome regions. With a cutoff at the training-set mean:

```text
R1 = {i: y_i <= mean(y)}
R2 = {i: y_i >  mean(y)}

mean(f(X_i), i in R1) = mean(y_i, i in R1)
mean(f(X_i), i in R2) = mean(y_i, i in R2)
```

This is a **model-fitting constraint**. It is not the same as fitting a post-hoc
correction on predicted age. The related OCR manuscript derives a different
restricted least-squares form that enforces zero calibration intercept and unit
calibration slope directly through covariance and mean constraints.

## Official implementation found

- Repository: <https://github.com/hwiyoungstat/UMLR>
- Audited commit: `3cf11abcecc5c3c99b9748d64c04f570d51243c3`
- Default branch: `main`
- Public, not archived
- Last audited commit date: 2026-04-21
- Contents: `Lasso_UP.R`, `Readme.md`, four generated PNG files and a placeholder
- Open issues at audit time: 0
- Pull requests at audit time: 0
- Releases at audit time: 0

The aging-clock manuscript calls this a web application and mentions user-friendly
R functions. The repository currently contains one function and a long executable
example, not a web application or installable R package.

## Concrete gaps

### Blocking distribution gap

No `LICENSE` file or package license declaration is present. GitHub publication
does not itself grant permission to redistribute or create a separate package from
the code. This audit therefore does not vendor `Lasso_UP.R`.

### Reproducibility gaps

- no `DESCRIPTION`, `NAMESPACE`, lockfile or dependency installation path;
- no automated tests or CI;
- no release or version tag;
- no solver-status or constraint-residual check;
- the example hard-codes a Windows working directory;
- the example has hidden `N == Nt` assumptions in test-vector generation;
- the example standardizes simulated test biomarkers using test-set statistics;
- no serialized model object or metadata contract;
- no documented behavior for missing, constant or extrapolated ages.

### Formula/code boundary mismatch

The paper defines `R1 = {i: y_i <= mean(y)}`. The audited code uses:

```text
young <- which(y < mean.y)
old   <- which(y > mean.y)
```

Observations exactly equal to the mean are excluded from both anchoring
constraints. Integer chronological ages make equality plausible. This does not
always change a fitted solution, but it is a testable divergence from the stated
method and becomes material in small or discrete-age samples.

### Degenerate input

For constant outcomes, both constraint groups are empty and the implementation
constructs averages with zero denominators. A clear validation error is needed.

## Registry and profile checks

On 2026-07-29, the official PyPI JSON endpoint for `UMLR` returned HTTP 404 and
the current CRAN `PACKAGES.gz` index contained no `Package: UMLR` entry. The
solver plugin `ROI.plugin.qpoases` was present in CRAN, but it is a dependency,
not the method. Searches of the project name, method name and authors'
repositories found no equivalent maintained package. The only official code
surface found was `hwiyoungstat/UMLR`.

The University of Maryland profiles for
[Hwiyoung Lee](https://www.medschool.umaryland.edu/profiles/lee-hwiyoung/) and
[Shuo Chen](https://www.medschool.umaryland.edu/profiles/chen-shuo/), plus Lee's
[ORCID record](https://orcid.org/0000-0002-3855-2316), were checked for additional
software links and did not expose another UMLR implementation. Chen's profile
lists the 2025 systematic-bias paper and identifies Lee as a former trainee.

## Scope comparison

| Surface | Official UMLR | This repository |
|---|---|---|
| Method implementation | R constrained Lasso | No production implementation; independent numerical oracle only |
| License | Missing at audited commit | MIT for original audit code |
| Tests / CI | None | Equation, leakage, edge-case and simulation tests |
| Real case studies | Controlled UKB/FHS inputs | Open example only; aggregate outputs committed |
| Distribution | Source files in GitHub | Installable audit CLI, not a clinical calibrator |
| Main purpose | Demonstrate proposed method | Falsify, characterize trade-offs and prepare upstream evidence |

## Gate A decision

Creating a second UMLR package would violate the brief. A narrow validation project
is justified because the official implementation has concrete, independently
testable gaps. The next gate is allowed only after:

1. the independent oracle satisfies the stated constraints on manual examples;
2. the ten frozen simulations run without leakage;
3. the results disclose error/bias trade-offs and failure modes;
4. no official source code or unlicensed raw data are redistributed.

All four conditions are met. The resulting project remains deliberately narrower
than a reference replacement package.
