# Numerical Illustration: Scope and Assumptions

The public code implements the reduced synthetic PBFE path

`IM -> PRP (LAI) -> Q proxy (TFP) -> DV (normalized loss)`.

It does not implement a calibrated agricultural damage measure or state (`DM/DS`) and does not analyze empirical agricultural data. The calculation conditions on `D1` with probability one, so it is a scenario calculation rather than an annual frequency or risk estimate.

## Baseline direct model

The baseline uses deterministic discrete propagation through these illustrative assumptions:

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

LAI and TFP use normalized CDF-difference bin masses on their finite conditioning domains. Conditional loss exceedance is evaluated with the analytic lognormal survival function. At the illustrative limit `d_lim = 1.00`, the authoritative result is

```text
P(DV > d_lim) = 0.09591450211754649
```

The illustrative probability levels used for target-oriented comparisons are `p_lim = 0.10` and `p_lim = 0.05`. They are demonstration thresholds, not agricultural performance standards.

## Continuous Monte Carlo verification

The extension maps three independent standard-normal drivers through continuous finite-domain conditional lognormal relationships for LAI, TFP, and normalized loss. With `N = 1,000,000` and random seed `20260810`, the independent Monte Carlo result is

```text
P(DV > 1) = 0.095851
SE          = 0.0002943867962375351
```

The direct-minus-Monte-Carlo difference is smaller than one Monte Carlo standard error, so the two results are consistent at the simulation's sampling resolution. The Monte Carlo calculation is a numerical verification of the synthetic hierarchy and does not replace the authoritative direct result.

## FORM diagnostic

The FORM performance function is

```text
g_L(u) = 1.00 - DV(u)
```

with the undesirable region defined by `g_L < 0`, equivalently `DV > 1.00`. The safe-origin check gives `g_L(0) = 0.2301615035128456 > 0`. The governing solution is:

```text
beta              = 1.2255614575907634
FORM probability  = 0.11018187469032242
design point u*   = [-0.32200378393151086,
                     -0.2670877121208008,
                      1.1519455731499912]
direction cosines = [ 0.26273986295475965,
                       0.2179308317562805,
                      -0.9399329321736364]
```

The FORM probability is approximately 14.9% above the direct discrete result. FORM is an approximation here. The directional cosines are the normalized gradient at the local `DV = 1` design point; they are not global sensitivity indices, variance fractions, or causal importance measures.

## Synthetic target-oriented multiplier families

Six one-dimensional families independently multiply the nominal mean (`m`) or arithmetic COV (`c`) in the LAI (`L`), TFP-link (`T`), and normalized-loss-link (`D`) modules:

- `r_mL`, `r_cL`;
- `r_mT`, `r_cT`; and
- `r_mD`, `r_cD`.

Each profile holds the other multipliers at one and evaluates the authoritative direct discrete probability. The resolved crossings of `p_lim = 0.05` are:

| Multiplier | Crossing |
|---|---:|
| `r_mL` | `1.649894453167797` |
| `r_mT` | `1.3065074972957837` |
| `r_mD` | `0.9302953633446296` |
| `r_cD` | `0.7200710995853127` |

No 5% crossing was resolved for `r_cL` or `r_cT` in the extended numerical search. The observed `r_cL` profile is nonmonotonic. This bounded numerical finding does not establish mathematical impossibility outside the evaluated search.

The publication-facing figure evaluates direct probabilities on a deterministic 151-point linear grid over `0.5 <= r <= 2.0`. That range is a display domain, not a calibrated or scientifically admissible parameter range. The extended root-search domain is a numerical search device and is not interpreted as a meaningful intervention range.

## Interpretation boundaries

All distributions, relationships, COVs, finite bounds, multiplier scenarios, and thresholds are synthetic. They are not calibrated agronomic, damage, yield, productivity, intervention, economic, or financial-loss models. The repository includes no empirical agricultural data, no strategy ranking, no annual agricultural risk estimate, and no end-to-end PBFE validation. The extension supports numerical uncertainty diagnostics and target-oriented illustration only.
