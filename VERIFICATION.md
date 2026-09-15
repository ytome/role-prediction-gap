# Verification

Verified on 2026-09-15 from the package root.

Environment:

```text
Python 3.14.4
NumPy 2.5.2
```

Command:

```bash
python scripts/smoke_test.py
```

Result: PASS.

The smoke test regenerated:

- `outputs/smoke/readout_summary.csv`
- `outputs/smoke/readout_contrasts.csv`
- `outputs/smoke/measurement_strata.csv`
- `outputs/smoke/measurement_high_low.csv`

The regenerated files matched the expected tables numerically with tolerance `1e-12`.

Additional checks:

- The smoke test includes guard checks that duplicate keys and non-finite numeric values are rejected before table comparisons.
- Generated outputs, bytecode caches and virtual environments are ignored by git.
- The release tree was scanned for local absolute paths and credential markers.
