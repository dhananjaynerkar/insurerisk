from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.risk_service import score_claims


def main() -> None:
    p = Path("data/processed/claims_features.csv")
    if not p.exists():
        p = Path("data/interim/claims_merged.csv")
    if not p.exists():
        raise FileNotFoundError("No data found for post-load inspection.")

    df = pd.read_csv(p).head(50)
    top = score_claims(df, top_n=10)
    cols = [c for c in ["fraud_probability", "predicted_claim_amount", "risk_score"] if c in top.columns]
    print(top[cols].to_string(index=False))


if __name__ == "__main__":
    main()
