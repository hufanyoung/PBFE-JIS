# Numerical Illustration: Scope and Assumptions

The public code implements the reduced synthetic PBFE path

`IM -> PRP (LAI) -> Q proxy (TFP) -> DV (normalized loss)`.

It does **not** implement a calibrated agricultural damage measure/state (`DM/DS`)
and does not analyze empirical agricultural data.

## Illustrative assumptions

- Environmental-hazard states: D0--D4.
- Propagated scenario: `P(D1) = 1`.
- LAI arithmetic means: 1.8, 1.6, 1.4, 1.2, 1.0 for D0--D4.
- LAI arithmetic COV: 0.5.
- LAI propagation domain: `(0, 6]`, 300 bins of width 0.02.
- TFP proxy mean: `0.2 + LAI / 6`.
- TFP proxy arithmetic COV: 0.2.
- TFP propagation domain: `(0, 1.8]`, 360 bins of width 0.005.
- Normalized-loss mean: `1 - TFP / 2`.
- Normalized-loss arithmetic COV: 0.2.
- Loss-exceedance thresholds: 0 to 2 in increments of 0.005.

LAI and TFP use normalized CDF-difference bin masses on their finite
conditioning domains. Conditional loss exceedance is evaluated with the
analytic lognormal survival function.

All relationships, COVs, and finite bounds are synthetic assumptions for
demonstrating probabilistic propagation. They are not calibrated agronomic,
yield, productivity, or financial-loss models.
