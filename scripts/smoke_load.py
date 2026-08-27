from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.predict import predict_batch


def main() -> None:
    path = Path("data/processed/claims_features.csv")
    if not path.exists():
        path = Path("data/interim/claims_merged.csv")
    if not path.exists():
        raise FileNotFoundError("No processed/interim data found for smoke test.")

    df = pd.read_csv(path).head(5)
    out = predict_batch(df)
    print("Smoke load success.")
    print(out.head().to_string(index=False))


if __name__ == "__main__":
    main()
