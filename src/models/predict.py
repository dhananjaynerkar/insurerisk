from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.build_features import add_features


DEFAULT_METADATA_PATH = Path("models/metadata.json")
DEFAULT_FRAUD_MODEL_PATH = Path("models/fraud_model.pkl")
DEFAULT_SEVERITY_MODEL_PATH = Path("models/severity_model.pkl")


def _load_metadata(path: str | Path = DEFAULT_METADATA_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _apply_defaults(df: pd.DataFrame, required_cols: list[str], defaults: dict) -> pd.DataFrame:
    out = df.copy()
    for col in required_cols:
        if col not in out.columns:
            out[col] = defaults.get(col)
        else:
            default_value = defaults.get(col)
            if default_value is not None:
                out[col] = out[col].fillna(default_value)
    return out


def _coerce_dtypes(df: pd.DataFrame, dtypes: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c) for c in out.columns]

    for col, dtype in dtypes.items():
        if col not in out.columns:
            continue
        if dtype.startswith("datetime64"):
            out[col] = pd.to_datetime(out[col], errors="coerce")
        elif dtype.startswith("int") or dtype.startswith("float"):
            out[col] = pd.to_numeric(out[col], errors="coerce")
        elif dtype == "bool":
            out[col] = out[col].astype("bool")
        else:
            out[col] = out[col].astype("object")
    return out


def _normalize_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.replace({pd.NA: np.nan})
    dt_cols = list(out.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns)
    for col in dt_cols:
        out[col] = out[col].astype("int64")
        out[col] = out[col].replace(-9223372036854775808, np.nan).astype(float)
    return out


def _prepare_for_model(
    df: pd.DataFrame,
    required_cols: list[str],
    defaults: dict[str, object] | None = None,
    dtypes: dict[str, str] | None = None,
) -> pd.DataFrame:
    out = add_features(df)
    out = _apply_defaults(out, required_cols, defaults or {})
    if dtypes:
        out = _coerce_dtypes(out, dtypes)
    out = out.reindex(columns=required_cols)
    out = _normalize_datetimes(out)
    return out


def load_models(
    fraud_model_path: str | Path = DEFAULT_FRAUD_MODEL_PATH,
    severity_model_path: str | Path = DEFAULT_SEVERITY_MODEL_PATH,
):
    fraud_model = joblib.load(fraud_model_path)
    severity_model = joblib.load(severity_model_path)
    metadata = _load_metadata()
    return fraud_model, severity_model, metadata


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    fraud_model, severity_model, metadata = load_models()

    Xc = _prepare_for_model(
        df=df,
        required_cols=metadata["class_features"],
        defaults=metadata.get("class_defaults"),
        dtypes=metadata.get("class_dtypes"),
    )
    Xr = _prepare_for_model(
        df=df,
        required_cols=metadata["reg_features"],
        defaults=metadata.get("reg_defaults"),
        dtypes=metadata.get("reg_dtypes"),
    )

    fraud_prob = fraud_model.predict_proba(Xc)[:, 1]
    severity_log = severity_model.predict(Xr)
    severity = np.expm1(severity_log)
    risk_score = fraud_prob * severity

    return pd.DataFrame(
        {
            "fraud_probability": fraud_prob,
            "predicted_claim_amount": severity,
            "risk_score": risk_score,
        }
    )

