from __future__ import annotations

import pandas as pd

from src.models.predict import predict_batch


def score_claims(claims_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    scored = predict_batch(claims_df)
    out = claims_df.reset_index(drop=True).copy()
    out = pd.concat([out, scored], axis=1)
    out = out.sort_values("risk_score", ascending=False).head(top_n).reset_index(drop=True)
    return out

