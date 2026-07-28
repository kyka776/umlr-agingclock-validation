Thank you for making the UMLR demonstration available. I built an independent
audit against the equations in the aging-clock preprint:

https://github.com/kyka776/umlr-agingclock-validation

This is not a replacement implementation and does not vendor the repository's R
source. It uses an independent numerical oracle plus pinned source
characterization.

At commit `3cf11abcecc5c3c99b9748d64c04f570d51243c3`, I found three narrow questions:

1. The preprint defines `R1 = {i: Y_i <= mean(Y)}`, but `Lasso_UP.R` uses
   `young <- which(y < mean.y)`. With discrete ages, observations exactly at the
   mean are excluded from both constraints. Is strict `<` intended, or should
   the source follow `<=`?
2. A constant outcome makes both `young` and `old` empty, so the region means
   involve zero-length groups. Would you accept an explicit validation error?
3. The source reads `opt$message$primal_solution` without checking solver status
   or the final constraint residual. Would you welcome a small post-solve
   validation?

The exact pinned characterization is in
`results/tables/upstream_characterization.json`; the equation and edge-case
tests are in `tests/test_models.py`.

The repository currently has no license file. If you would like a pull request,
could you first clarify the intended license and lower-region boundary? I am
happy to prepare a minimal tested patch after that.
