# Performance-Based Food Engineering (PBFE)

This repository contains reproducible code for the reduced-form numerical illustration accompanying a framework manuscript on **Performance-Based Food Engineering (PBFE)**. The full conceptual framework distinguishes environmental hazard, latent plant response, damage state, physical or operational consequence, stakeholder decision variable, strategy, context, observations, and history. The code here implements a deliberately reduced synthetic path; it is not an end-to-end PBFE validation.

## Baseline direct propagation

The deterministic baseline implements

`IM -> PRP (LAI) -> Q proxy (TFP) -> DV (normalized loss)`.

It uses finite-domain conditional distributions and direct discrete probability propagation. The authoritative baseline result at the illustrative loss limit `DV = 1` is

```text
P(DV > 1) = 0.09591450211754649
```

The baseline does not use Monte Carlo sampling. Running `src/pbfe_numerical_example.py` regenerates four baseline figures and `outputs/numerical_summary.json`.

## v1.1.0 UQ and target extension

The extension retains the direct discrete probability as authoritative and adds:

- an independent continuous Monte Carlo check;
- a first-order reliability method (FORM) approximation;
- local design-point directional cosines;
- six one-dimensional synthetic target-oriented multiplier families;
- a publication-facing target-oriented figure; and
- machine-readable diagnostics and tables.

The verified probability diagnostics are:

| Quantity | Value |
|---|---:|
| Direct discrete probability | `0.09591450211754649` |
| Continuous Monte Carlo probability | `0.095851` |
| Monte Carlo Bernoulli standard error | `0.0002943867962375351` |
| Monte Carlo sample size | `1,000,000` |
| Monte Carlo seed | `20260810` |
| FORM reliability index, beta | `1.2255614575907634` |
| FORM probability | `0.11018187469032242` |

The governing FORM design point is

```text
[-0.32200378393151086, -0.2670877121208008, 1.1519455731499912]
```

and the corresponding local directional-cosine vector is

```text
[0.26273986295475965, 0.2179308317562805, -0.9399329321736364]
```

The six multiplier families separately vary the nominal mean or coefficient of variation (COV) for the LAI, TFP-link, and normalized-loss-link modules. The resolved crossings of the illustrative 5% exceedance level are:

| Multiplier | Resolved crossing |
|---|---:|
| `r_mL` | `1.649894453167797` |
| `r_mT` | `1.3065074972957837` |
| `r_mD` | `0.9302953633446296` |
| `r_cD` | `0.7200710995853127` |

No 5% crossing was resolved for `r_cL` or `r_cT` in the extended numerical search. The `r_cL` profile was nonmonotonic. These findings do not prove that a crossing is mathematically impossible.

All parameter scenarios are synthetic and are not calibrated agricultural interventions. The FORM directional cosines are local diagnostics at the `DV = 1` boundary, not global sensitivity indices or variance fractions.

## Interpretation boundaries

This repository includes no empirical agricultural data, calibrated damage model, empirical economic model, or empirical strategy model. It establishes no strategy ranking and produces no annual agricultural risk estimate. The fixed `D1` calculation is a conditional scenario, not an annual hazard-frequency model. No end-to-end PBFE validation is claimed.

## Repository structure

```text
PBFE-JIS/
├── README.md
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── .gitignore
├── src/
│   ├── pbfe_numerical_example.py
│   ├── pbfe_uq_target_extension.py
│   └── generate_pbfe_target_scenarios_figure.py
├── tests/
│   ├── test_pbfe_numerical_example.py
│   └── test_pbfe_uq_target_extension.py
├── figures/
│   ├── illustrative_lai_distributions.png
│   ├── illustrative_tfp_distributions.png
│   ├── illustrative_loss_distributions.png
│   ├── synthetic_loss_exceedance.png
│   └── pbfe_target_scenarios.png
├── outputs/
│   ├── numerical_summary.json
│   └── uq_target_extension/
│       ├── pbfe_uq_target_diagnostics.json
│       ├── pbfe_uq_target_sensitivity_verification_table.csv
│       └── pbfe_uq_target_profiles_and_roots.csv
└── docs/
    └── model_scope.md
```

## Reproduce the results

The reference environment uses Python 3.9 and the dependency versions recorded in `requirements.txt` and `environment.yml`.

### Conda

```bash
conda env create -f environment.yml
conda activate pbfe
```

### Python virtual environment

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run these commands from the repository root:

```bash
# Baseline calculation and its five tests
python src/pbfe_numerical_example.py
python -m unittest -v tests/test_pbfe_numerical_example.py

# UQ and target extension and its sixteen tests
python src/pbfe_uq_target_extension.py
python -m unittest -v tests/test_pbfe_uq_target_extension.py

# Publication-facing target figure
python src/generate_pbfe_target_scenarios_figure.py

# Complete 21-test suite
python -m unittest discover -s tests -v
```

The extension writes only machine-readable files under `outputs/uq_target_extension/`. The target-figure generator reads those diagnostics and writes `figures/pbfe_target_scenarios.png` on the fixed display domain `0.5 <= r <= 2.0`.

## Data availability

No empirical agricultural data are required or included. All numerical inputs are synthetic and are stated in the source code and `docs/model_scope.md`.

## Citation

Please cite the associated PBFE manuscript once its final bibliographic information is available. Repository citation metadata are provided in `CITATION.cff`.

## License

No software license has yet been assigned to this repository.

## Contact

Khalid M. Mosalam (corresponding author): mosalam@berkeley.edu
