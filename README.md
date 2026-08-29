# Insurance Fraud Detection and Claim Risk Scoring

## About

An evaluation-aware insurance fraud and claim-risk machine-learning pipeline with temporal features, leakage auditing, imbalance handling, threshold tuning, business-loss analysis, serialized artifacts, and a Streamlit scoring interface.

> **Status:** Evaluation-aware local ML pipeline; no production success is claimed.  
> **Stack:** Python · pandas · scikit-learn · Streamlit · YAML configuration  
> **Proof:** Temporal validation, leakage auditing, imbalance handling, threshold tuning, and business-loss analysis.


This project builds an end-to-end machine learning pipeline for insurance fraud detection and claim severity prediction. It uses time-aware validation, leakage-safe feature engineering, imbalance handling, threshold tuning, business-cost evaluation, and a Streamlit interface for claim risk scoring.

## Project Structure

- `app/`: Streamlit dashboard.
- `configs/`: YAML configuration for paths, split settings, and model parameters.
- `src/data/`: Data loading and dataset creation.
- `src/features/`: Feature engineering logic.
- `src/models/`: Training, prediction, risk scoring, and anomaly model helpers.
- `scripts/`: Local validation and model inspection scripts.
- `tests/`: Automated smoke tests.
- `notebooks/Insurance_Fraud_ML_Pipeline_new.ipynb`: Main ML notebook.

## Local-Only Artifacts

The following folders are intentionally ignored by Git:

- `data/`: raw/interim/processed datasets.
- `models/`: trained model binaries and metadata.
- `reports/`: generated metrics and analysis outputs.
- `.venv/`: local Python environment.

Do not commit private datasets, model binaries, cache folders, or generated reports.

See [dataset provenance notes](docs/DATASET_PROVENANCE.md) for the input-data publication boundary.

## Setup

Run these commands from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Place these files in `data/raw/`:

- `insurance_data.csv`
- `employee_data.csv`
- `vendor_data.csv`

## Train Models

```powershell
python -m src.models.train
```

Training creates local artifacts under `data/processed/`, `models/`, and `reports/`.

## Run Streamlit App

```powershell
streamlit run app/streamlit_app.py --server.port 8501
```

Open:

```text
http://localhost:8501
```

## Validate

```powershell
python scripts/smoke_load.py
python scripts/inspect_model.py
pytest -q
```



## Evaluation snapshot

The documented notebook output at commit 5c5b99564313c9c84021e8e1dce263e1eba99ad6 records a temporal split of 7,000 train rows, 1,500 validation rows, and 1,500 test rows. The selected checkpoint was Autoencoder (Anomaly) with a threshold of 0.20.

| Metric | Test-window result |
| --- | ---: |
| PR-AUC | 0.044732 |
| ROC-AUC | 0.483760 |
| Recall | 0.898551 |
| Precision | 0.046757 |
| F2 score | 0.193508 |
| Business loss | 982,000 |

These values are a documented notebook checkpoint, not a new run or a claim of production success. The high recall and low precision show the review-capacity trade-off. The notebook's false-negative and false-positive costs are scenario parameters, not verified insurer costs or business savings. See [evaluation notes](docs/EVALUATION_NOTES.md) for the interpretation boundary and required run metadata.

## Model Selection

The final fraud model is selected using fraud-focused and business-focused metrics, not accuracy. The comparison prioritizes PR-AUC, recall, F2-score, threshold-tuned business loss, and top-K investigation lift. This matches the real use case: ranking suspicious claims for investigation while controlling false-review cost.
