# Insurerisk evaluation notes

**Status:** Documented notebook checkpoint, not a new training run.

## Snapshot

- Repository: dhananjaynerkar/insurerisk
- Repository commit containing the documented notebook output: 5c5b99564313c9c84021e8e1dce263e1eba99ad6
- Notebook: notebooks/Insurance_Fraud_ML_Pipeline_new.ipynb
- Split recorded in the notebook: 7,000 train rows, 1,500 validation rows, and 1,500 test rows
- Split strategy: temporal ordering with feature construction intended to avoid future-label leakage
- Selected checkpoint in the notebook: Autoencoder (Anomaly)
- Selected threshold: 0.20

## Test-window metrics

| Metric | Result |
| --- | ---: |
| PR-AUC | 0.044732 |
| ROC-AUC | 0.483760 |
| Recall | 0.898551 |
| Precision | 0.046757 |
| F2 score | 0.193508 |
| Business loss | 982,000 |

The notebook defines business-loss costs of 50,000 for a false negative and
500 for a false positive. These are scenario parameters in the project, not
verified insurer costs or business savings.

## Interpretation boundary

The checkpoint demonstrates an evaluation-aware pipeline and a recall-focused
thresholding workflow. It does not demonstrate a useful production fraud
ranker: the selected test PR-AUC is low, the ROC-AUC is below 0.5, and precision
is low at the selected threshold. No business impact, production deployment,
or external validation is verified.

The Streamlit interface is documented as a local scoring interface. Do not
describe it as a public or production deployment without separate evidence.

## What future results must include

Record the commit, input versions, date range, split boundaries, candidate
models, threshold-selection data, final test window, metric definitions, and
failure/skipped-run details. Keep cross-validation, validation, and final test
results in separate tables.

