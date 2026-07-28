# Pinned upstream characterization

The official `Lasso_UP.R` is not vendored because its audited repository has no
license. To reproduce the characterization after fetching the pinned commit:

```bash
python upstream-tests/characterize_upstream.py \
  /path/to/UMLR/Lasso_UP.R \
  --output results/tables/upstream_characterization.json
```

The script verifies the exact SHA-256 before reporting source properties. It does
not execute the R optimizer and does not claim runtime parity. The independent
equation tests are in `tests/test_models.py`.
