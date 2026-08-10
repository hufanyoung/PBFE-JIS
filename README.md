# Performance-Based Food Engineering (PBFE)

This repository contains the reproducible code for the reduced-form numerical
illustration accompanying a manuscript on **Performance-Based Food Engineering
(PBFE)**.

PBFE is formulated as a probabilistic framework for risk-informed assessment
of agricultural production infrastructure. The complete conceptual chain is

`IM -> PRP -> DM/DS -> Q -> DV`

with observation/state-estimation, strategy, context, and exposure history
represented separately where relevant.

## What this repository contains

The code here implements only the reduced synthetic demonstration

`IM -> PRP (LAI) -> Q proxy (TFP) -> DV (normalized loss)`.

It demonstrates conditional probability propagation. It does **not** contain:

- empirical agricultural datasets;
- a calibrated agricultural damage model;
- an empirically estimated crop-yield model;
- an empirical financial-loss model;
- an annual hazard-occurrence model;
- a validated strategy ranking; or
- an end-to-end PBFE implementation.

## Repository structure

```text
PBFE-JIS/
├── README.md
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── .gitignore
├── src/
│   └── pbfe_numerical_example.py
├── tests/
│   └── test_pbfe_numerical_example.py
├── figures/
│   ├── illustrative_lai_distributions.png
│   ├── illustrative_tfp_distributions.png
│   ├── illustrative_loss_distributions.png
│   └── synthetic_loss_exceedance.png
├── outputs/
│   └── numerical_summary.json
└── docs/
    └── model_scope.md
```

## Reproduce the numerical illustration

The reference environment used Python 3.9 with the dependency versions recorded
in `requirements.txt` and `environment.yml`.

### Conda

```bash
conda env create -f environment.yml
conda activate pbfe
python src/pbfe_numerical_example.py
python -m unittest -v tests/test_pbfe_numerical_example.py
```

### Python virtual environment

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/pbfe_numerical_example.py
python -m unittest -v tests/test_pbfe_numerical_example.py
```

Running the script regenerates the four PNG files in `figures/` and writes
machine-readable diagnostics to `outputs/revised_summary.json`.

The calculation is deterministic; it does not use Monte Carlo sampling.

## Numerical checks

The test suite checks:

1. recovery of the specified arithmetic mean and COV from the lognormal
   parameterization;
2. normalization of finite-domain LAI and TFP probability masses;
3. positivity of downstream loss means;
4. boundedness and monotonicity of the final exceedance curve; and
5. expected numerical array dimensions.

## Figures

The checked-in figures are the manuscript-facing outputs of the deterministic
script. They can be regenerated from `src/pbfe_numerical_example.py`.

## Data availability

No empirical agricultural data are required to reproduce this repository.
All numerical inputs are synthetic and are stated in the source code and
`docs/model_scope.md`.

## Citation

Please cite the associated PBFE manuscript once its final bibliographic
information is available. Repository citation metadata are provided in
`CITATION.cff`.

## License

No software license has yet been assigned to this repository.

## Contact

Khalid M. Mosalam (corresponding author): mosalam@berkeley.edu
