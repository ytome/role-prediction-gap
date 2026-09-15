# Role Prediction Gap Minimal Reproducibility Package

This repository contains the minimal public code and saved numeric data used to reproduce the post-hoc checks added during the reviewer-directed revision of the manuscript:

**Testing Role Identity as a Predictor of Prompt-Injection Behavior**.

It is a verification package for saved analysis artifacts. It does not rerun language-model generation, hidden-state extraction, probe fitting, attack judging, or stopping experiments. The data cover the original reference-set scope reported in the manuscript.

## What This Reproduces

1. Mean case-held-out log loss for four readouts:
   - `ML`: calibrated local role-coordinate readout.
   - `MLrbf`: post-hoc nonlinear RBF Nystroem readout over the same local role coordinates.
   - `MHpca4`: post-hoc four-dimensional PCA readout from the full hidden state.
   - `MH`: full pre-action hidden-state readout.

   All four readouts include the same control variables `Z`.

2. Paired semantic-case bootstrap intervals for the predefined readout contrasts:
   - `MLrbf-ML`
   - `MLrbf-MH`
   - `MLrbf-MHpca4`
   - `MHpca4-ML`
   - `MHpca4-MH`

3. Descriptive measurement-stratified checks for `ML-MH`, including:
   - overall full-state advantage;
   - five wording-role strata;
   - benign-reference alignment bins;
   - direct high-minus-low alignment-bin contrast.

The independent bootstrap unit is the semantic case. The five wording rows for a case stay together when the case-level loss gap is analysed.

## Directory Layout

```text
.
  README.md
  LICENSE
  requirements.txt
  .github/workflows/smoke.yml
  data/
    readout_case_losses.csv
    measurement_case_alignment.csv
    measurement_row_losses.csv
    expected_readout_summary.csv
    expected_readout_contrasts.csv
    expected_measurement_strata.csv
    expected_measurement_high_low.csv
  scripts/
    common.py
    reproduce_readout_checks.py
    reproduce_measurement_strata.py
    smoke_test.py
  outputs/                  # generated locally; ignored by git
```

## Setup

Use Python 3.12 or newer. NumPy 2.5.2 declares `Requires-Python: >=3.12`. The release was locally verified with Python 3.14.4 and NumPy 2.5.2.

```bash
git clone https://github.com/ytome/role-prediction-gap.git
cd role-prediction-gap
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/smoke_test.py
```

The only Python dependency is NumPy.

## Running Individual Checks

```bash
python scripts/reproduce_readout_checks.py --check
python scripts/reproduce_measurement_strata.py --check
```

By default, outputs are written to `outputs/`. To write elsewhere:

```bash
python scripts/reproduce_readout_checks.py --out-dir /tmp/readout_check --check
python scripts/reproduce_measurement_strata.py --out-dir /tmp/measurement_check --check
```

## Expected Key Values

Readout means:

| Model | ML | MLrbf | MHpca4 | MH |
|---|---:|---:|---:|---:|
| GPT-OSS | 0.534199 | 0.529330 | 0.531661 | 0.439505 |
| Qwen | 0.511044 | 0.506987 | 0.510039 | 0.466164 |

Nonlinear role readout versus full state:

| Model | Contrast | Mean | 95% CI |
|---|---|---:|---:|
| GPT-OSS | MLrbf-MH | 0.089825 | [0.051224, 0.128590] |
| Qwen | MLrbf-MH | 0.040823 | [0.009826, 0.070222] |

Overall calibrated role readout versus full state:

| Model | Contrast | Mean | 95% CI |
|---|---|---:|---:|
| GPT-OSS | ML-MH | 0.094694 | [0.058669, 0.131432] |
| Qwen | ML-MH | 0.044880 | [0.013627, 0.076928] |

High-minus-low benign alignment contrast:

| Model | Mean | 95% CI |
|---|---:|---:|
| GPT-OSS | 0.147959 | [0.049718, 0.255794] |
| Qwen | -0.016303 | [-0.081950, 0.047506] |

Positive log-loss contrasts mean the left readout has higher loss. For `MLrbf-MH` and `ML-MH`, positive values mean the full-state readout has lower loss.

## Data Schema

`data/readout_case_losses.csv`

| Column | Meaning |
|---|---|
| `model` | Model label used in the manuscript tables. |
| `case_id` | Semantic case identifier. |
| `loss_ML` | Case-mean held-out log loss for the calibrated local role readout. |
| `loss_MH` | Case-mean held-out log loss for the full-state readout. |
| `loss_MLrbf` | Case-mean held-out log loss for the RBF role readout. |
| `loss_MHpca4` | Case-mean held-out log loss for the four-dimensional hidden PCA readout. |

`data/measurement_row_losses.csv`

| Column | Meaning |
|---|---|
| `model` | Model label. |
| `condition_id` | Numeric-condition identifier used to join rows within a semantic case. |
| `case_id` | Semantic case identifier. |
| `style_role` | One of `system`, `user`, `tool`, `cot`, or `assistant`. |
| `category` | Attack category label used as a control in the readouts. |
| `attack_attempt` | Binary AttackAttempt label. |
| `pred_ML`, `pred_MH` | Saved out-of-fold probabilities. |
| `loss_ML`, `loss_MH` | Row-level log losses. |
| `gap_ML_minus_MH` | `loss_ML - loss_MH`; positive means full-state lower loss. |
| `local_alignment_overall` | Case-level benign local reference-alignment rate. |
| `global_alignment_overall` | Case-level benign global reference-alignment rate. |
| `alignment_group` | Tie-preserving benign-alignment bin. |

`data/measurement_case_alignment.csv` is included to expose the case-level alignment values and bin assignments used by `measurement_row_losses.csv`.

## Data Provenance and Scope

The CSV files are compact numeric exports from saved reviewer-repair artifacts. Case identifiers are analysis identifiers, and the package does not include raw prompts or model transcripts.

The reference-alignment fields follow the manuscript scope. GPT-OSS uses its saved original reference measurements. Qwen alignment uses five fixed validation texts from the global training bank, held out from the two local anchors per role.

## Statistical Details

All intervals are percentile bootstrap intervals.

For readout contrasts, the script samples semantic cases with replacement and recomputes the mean case-level loss contrast. GPT-OSS uses 5,000 draws; Qwen uses 2,000 draws. Both use seed `20260915`, matching the reviewer-repair readout analysis.

For measurement strata, the script first averages row-level `ML-MH` gaps within each semantic case and stratum, then samples semantic cases with replacement. Each stratum uses 5,000 draws. The high-minus-low alignment comparison samples cases independently within the high and low alignment groups.

The overall `ML-MH` mean is the same quantity shown by the main Table 1 linear role versus full-state comparison. Some confidence-interval endpoints differ slightly because the measurement-strata audit uses its own fixed seeds and 5,000 draws for both models, while the primary Table 1 Qwen interval used the original 2,000-draw analysis record.

For GPT-OSS alignment bins, the high-minus-low comparison compares the predefined high and low benign-alignment groups only: high `n=36`, low `n=65`. The mid group (`n=93`) remains in `expected_measurement_strata.csv` and is not part of that two-group descriptive contrast.

These are marginal post-hoc intervals conditional on the saved out-of-fold predictions, with no multiplicity correction. They do not include uncertainty from refitting models, selecting hyperparameters, rerunning generation, or rerunning the attack judge.

## What Is Not Included

This package intentionally omits:

- raw prompts, injected texts, model outputs, and judge transcripts;
- API keys, server settings, and local absolute paths;
- full hidden-state arrays and upstream feature-extraction code;
- scripts for rerunning LLM generation or stopping gates;
- author names, affiliations, and ORCID fields.

The package is therefore suitable for checking the reported numeric tables from saved artifacts. It is not a full experimental rerun package.

## Continuous Verification

GitHub Actions runs the smoke test on pushes and pull requests. The workflow installs `requirements.txt` and runs:

```bash
python scripts/smoke_test.py
```

## License

The code, documentation, and bundled numeric CSV exports in this repository are released under the MIT License. See `LICENSE`.
