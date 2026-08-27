from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    fraud_path = Path("models/fraud_model.pkl")
    sev_path = Path("models/severity_model.pkl")
    meta_path = Path("models/metadata.json")

    if not (fraud_path.exists() and sev_path.exists() and meta_path.exists()):
        print("Missing model artifacts. Run: python -m src.models.train")
        return

    fraud = joblib.load(fraud_path)
    sev = joblib.load(sev_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    print(f"Fraud model: {type(fraud).__name__}")
    print(f"Severity model: {type(sev).__name__}")
    print(f"Class feature count: {len(meta.get('class_features', []))}")
    print(f"Reg feature count: {len(meta.get('reg_features', []))}")
    print(f"Threshold: {meta.get('threshold')}")


if __name__ == "__main__":
    main()
