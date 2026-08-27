from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
import warnings

import joblib
import numpy as np
import pandas as pd
import yaml
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier

    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    from catboost import CatBoostClassifier

    HAS_CAT = True
except Exception:
    HAS_CAT = False

try:
    import shap

    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

from src.data.make_dataset import build_base_table
from src.features.build_features import add_features
from src.models.anomaly_models import (
    AutoencoderAnomalyDetector,
    IsolationForestAnomalyDetector,
)
from src.utils.io import save_csv


FN_COST = 50_000.0
FP_COST = 500.0
TOPK_CAPACITIES = (0.01, 0.05, 0.10, 0.20)

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")


def _load_config(config_path: str | Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _safe_save_csv(df: pd.DataFrame, path: Path, fallback_dir: Path) -> Path:
    try:
        save_csv(df, path)
        return path
    except PermissionError:
        fallback_dir.mkdir(parents=True, exist_ok=True)
        alt = fallback_dir / path.name
        save_csv(df, alt)
        return alt


def _safe_dump_joblib(obj: object, path: Path, fallback_dir: Path) -> Path:
    try:
        joblib.dump(obj, path)
        return path
    except PermissionError:
        fallback_dir.mkdir(parents=True, exist_ok=True)
        alt = fallback_dir / path.name
        joblib.dump(obj, alt)
        return alt


def _safe_write_text(text: str, path: Path, fallback_dir: Path) -> Path:
    try:
        path.write_text(text, encoding="utf-8")
        return path
    except PermissionError:
        fallback_dir.mkdir(parents=True, exist_ok=True)
        alt = fallback_dir / path.name
        alt.write_text(text, encoding="utf-8")
        return alt


def _normalize_features(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    out = out.replace({pd.NA: np.nan})
    dt_cols = list(
        out.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
    )
    for c in dt_cols:
        out[c] = out[c].astype("int64")
        out[c] = out[c].replace(-9223372036854775808, np.nan).astype(float)
    return out


def _safe_roc_auc(y_true: pd.Series, y_prob: np.ndarray) -> float:
    if len(pd.Series(y_true).dropna().unique()) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _safe_pr_auc(y_true: pd.Series, y_prob: np.ndarray) -> float:
    if len(pd.Series(y_true).dropna().unique()) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def _business_loss(y_true: pd.Series, y_pred: np.ndarray) -> float:
    y_arr = np.asarray(y_true).astype(int)
    p_arr = np.asarray(y_pred).astype(int)
    fn = int(((y_arr == 1) & (p_arr == 0)).sum())
    fp = int(((y_arr == 0) & (p_arr == 1)).sum())
    return float(fn * FN_COST + fp * FP_COST)


def _tune_threshold(y_true: pd.Series, y_prob: np.ndarray) -> float:
    best_thr = 0.5
    best_f2 = -1.0
    best_loss = float("inf")

    for thr in np.arange(0.05, 0.51, 0.01):
        pred = (y_prob >= thr).astype(int)
        f2 = float(fbeta_score(y_true, pred, beta=2, zero_division=0))
        loss = _business_loss(y_true, pred)
        if (f2 > best_f2) or (np.isclose(f2, best_f2) and loss < best_loss):
            best_f2 = f2
            best_loss = loss
            best_thr = float(thr)
    return best_thr


def _topk_table(
    y_true: pd.Series, y_prob: np.ndarray, capacities: tuple[float, ...] = TOPK_CAPACITIES
) -> pd.DataFrame:
    y_arr = np.asarray(y_true).astype(int)
    p_arr = np.asarray(y_prob).astype(float)
    n = len(y_arr)
    baseline_rate = float(y_arr.mean()) if n else 0.0
    order = np.argsort(-p_arr)
    sorted_y = y_arr[order]

    rows: list[dict[str, Any]] = []
    for frac in capacities:
        k = max(1, int(np.ceil(n * frac)))
        top_y = sorted_y[:k]
        fraud_caught = int(top_y.sum())
        precision = float(top_y.mean()) if k else 0.0
        recall = float(fraud_caught / max(int(y_arr.sum()), 1))
        lift = float(precision / baseline_rate) if baseline_rate > 0 else float("nan")
        pred = np.zeros(n, dtype=int)
        pred[:k] = 1
        pred_original_order = np.zeros(n, dtype=int)
        pred_original_order[order] = pred
        loss = _business_loss(y_arr, pred_original_order)
        rows.append(
            {
                "review_capacity": f"top_{int(frac * 100)}pct",
                "claims_reviewed": int(k),
                "precision": precision,
                "recall": recall,
                "fraud_caught": fraud_caught,
                "lift": lift,
                "business_loss": loss,
                "baseline_fraud_rate": baseline_rate,
            }
        )
    return pd.DataFrame(rows)


def _build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["TXN_DATE_TIME"] = pd.to_datetime(out["TXN_DATE_TIME"], errors="coerce")
    out = out.sort_values("TXN_DATE_TIME").reset_index(drop=True)

    for c in ("LOSS_DT", "REPORT_DT", "POLICY_EFF_DT", "DATE_OF_JOINING"):
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce")

    if "LOSS_DT" in out.columns and "REPORT_DT" in out.columns:
        out["days_to_report"] = (out["REPORT_DT"] - out["LOSS_DT"]).dt.days
        out["late_report_flag"] = (out["days_to_report"] > 30).astype(float)
    else:
        out["days_to_report"] = np.nan
        out["late_report_flag"] = np.nan

    if "CLAIM_AMOUNT" in out.columns and "PREMIUM_AMOUNT" in out.columns:
        out["claim_to_premium_ratio"] = out["CLAIM_AMOUNT"] / out["PREMIUM_AMOUNT"].replace(
            0, np.nan
        )
    else:
        out["claim_to_premium_ratio"] = np.nan

    if "TXN_DATE_TIME" in out.columns and "POLICY_EFF_DT" in out.columns:
        out["policy_age_years"] = (
            out["TXN_DATE_TIME"] - out["POLICY_EFF_DT"]
        ).dt.days / 365.25
    else:
        out["policy_age_years"] = np.nan

    if "TXN_DATE_TIME" in out.columns and "DATE_OF_JOINING" in out.columns:
        out["agent_experience_years"] = (
            out["TXN_DATE_TIME"] - out["DATE_OF_JOINING"]
        ).dt.days / 365.25
    else:
        out["agent_experience_years"] = np.nan

    return out


def _window_count_previous(ts: pd.Series, window_days: int) -> np.ndarray:
    arr = pd.to_datetime(ts, errors="coerce").to_numpy()
    out = np.zeros(len(arr), dtype=float)
    valid = ~pd.isna(arr)
    if not valid.any():
        return out
    arr = arr.astype("datetime64[ns]")
    idx = np.arange(len(arr))
    left = np.searchsorted(
        arr, arr - np.timedelta64(window_days, "D"), side="left"
    )
    counts = idx - left
    counts[~valid] = 0
    return counts.astype(float)


def _group_history_features(
    df: pd.DataFrame, group_col: str, target_col: str, prefix: str, global_rate: float
) -> pd.DataFrame:
    out = df.copy()
    key = out[group_col].astype("string").fillna("Unknown")
    out[f"{prefix}_claim_count"] = out.groupby(key).cumcount().astype(float)
    out[f"{prefix}_days_since_last_claim"] = (
        out.groupby(key)["TXN_DATE_TIME"].diff().dt.days
    ).fillna(999.0)
    out[f"{prefix}_claim_velocity"] = 1.0 / (out[f"{prefix}_days_since_last_claim"] + 1.0)

    prev_target = out.groupby(key)[target_col].shift(1)
    cum_sum = prev_target.groupby(key).cumsum().fillna(0.0)
    cum_cnt = out.groupby(key).cumcount().astype(float)
    k = 20.0
    out[f"{prefix}_rolling_fraud_rate"] = (cum_sum + k * global_rate) / (cum_cnt + k)

    if prefix == "customer":
        out["customer_7d_claim_count"] = (
            out.groupby(key, group_keys=False)["TXN_DATE_TIME"]
            .apply(lambda s: pd.Series(_window_count_previous(s, 7), index=s.index))
            .astype(float)
        )
        out["customer_30d_claim_count"] = (
            out.groupby(key, group_keys=False)["TXN_DATE_TIME"]
            .apply(lambda s: pd.Series(_window_count_previous(s, 30), index=s.index))
            .astype(float)
        )
    if prefix in {"agent", "vendor"}:
        out[f"{prefix}_30d_claim_count"] = (
            out.groupby(key, group_keys=False)["TXN_DATE_TIME"]
            .apply(lambda s: pd.Series(_window_count_previous(s, 30), index=s.index))
            .astype(float)
        )

    return out


def _add_leakage_safe_behavioral_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values("TXN_DATE_TIME").reset_index(drop=True)
    global_rate = float(out[target_col].mean())

    for col in ("CUSTOMER_ID", "AGENT_ID", "VENDOR_ID", "INCIDENT_STATE"):
        if col not in out.columns:
            out[col] = "Unknown"

    out = _group_history_features(out, "CUSTOMER_ID", target_col, "customer", global_rate)
    out = _group_history_features(out, "AGENT_ID", target_col, "agent", global_rate)
    out = _group_history_features(out, "VENDOR_ID", target_col, "vendor", global_rate)

    state_key = out["INCIDENT_STATE"].astype("string").fillna("Unknown")
    prev_target = out.groupby(state_key)[target_col].shift(1)
    cum_sum = prev_target.groupby(state_key).cumsum().fillna(0.0)
    cum_cnt = out.groupby(state_key).cumcount().astype(float)
    k = 20.0
    out["state_fraud_rate"] = (cum_sum + k * global_rate) / (cum_cnt + k)
    out["agent_claim_volume"] = out.groupby(out["AGENT_ID"].astype("string").fillna("Unknown")).cumcount().astype(float)

    prev_global = out[target_col].shift(1)
    gsum = prev_global.cumsum().fillna(0.0)
    gcnt = pd.Series(np.arange(len(out)), index=out.index, dtype=float)
    out["rolling_fraud_rate"] = (gsum + k * global_rate) / (gcnt + k)

    out["days_since_last_claim"] = out["customer_days_since_last_claim"]
    out["claim_velocity"] = out["customer_claim_velocity"]
    out["customer_claim_count"] = out["customer_claim_count"]

    return out


def _temporal_split(
    df: pd.DataFrame, test_size: float, val_size: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out["TXN_DATE_TIME"] = pd.to_datetime(out["TXN_DATE_TIME"], errors="coerce")
    out = out.sort_values("TXN_DATE_TIME").reset_index(drop=True)
    n = len(out)
    n_test = max(1, int(np.ceil(n * test_size)))
    n_val = max(1, int(np.ceil(n * val_size)))
    train = out.iloc[: n - (n_val + n_test)].copy()
    val = out.iloc[n - (n_val + n_test) : n - n_test].copy()
    test = out.iloc[n - n_test :].copy()
    return train, val, test


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = list(X.select_dtypes(include=["object", "category", "string"]).columns)
    num_cols = [c for c in X.columns if c not in cat_cols]
    num_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ohe",
                OneHotEncoder(
                    handle_unknown="ignore", min_frequency=0.01, sparse_output=True
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)], remainder="drop"
    )


def _leakage_audit_table(
    columns: list[str], task: str, target_col: str = "target"
) -> pd.DataFrame:
    high_leak_keywords = ("paid", "settlement", "approved", "final", "status")
    rows: list[dict[str, Any]] = []
    for col in columns:
        c = str(col)
        c_lower = c.lower()
        available = "Yes"
        risk = "Low"
        action = "Keep"
        rationale = "No obvious leakage signal."

        if c in {target_col, "CLAIM_STATUS"}:
            available = "No"
            risk = "High"
            action = "Drop"
            rationale = "Direct target leakage."
        elif c in {
            "TRANSACTION_ID",
            "POLICY_NUMBER",
            "SSN",
            "ACCT_NUMBER",
            "ROUTING_NUMBER",
            "EMP_ACCT_NUMBER",
            "EMP_ROUTING_NUMBER",
            "CUSTOMER_NAME",
            "ADDRESS_LINE1",
            "ADDRESS_LINE1_AGENT",
            "ADDRESS_LINE1_VENDOR",
        }:
            available = "Maybe"
            risk = "High"
            action = "Drop"
            rationale = "ID/PII-like field with overfit/privacy risk."
        elif any(k in c_lower for k in high_leak_keywords):
            available = "Maybe"
            risk = "High"
            action = "Review/Drop"
            rationale = "Possible post-outcome or decision-derived field."
        elif task == "regression" and c in {"CLAIM_AMOUNT", "claim_to_premium_ratio"}:
            available = "No"
            risk = "High"
            action = "Drop"
            rationale = "Contains target or direct target transformation."
        elif task == "classification" and c == "CLAIM_AMOUNT":
            available = "Yes"
            risk = "Medium"
            action = "Keep with caution"
            rationale = "Can be valid at FNOL, but may be unstable across time."
        rows.append(
            {
                "feature": c,
                "task": task,
                "available_before_decision": available,
                "leakage_risk": risk,
                "action": action,
                "rationale": rationale,
            }
        )
    return pd.DataFrame(rows).sort_values(["leakage_risk", "feature"], ascending=[True, True])


def _build_experiments(
    pos_weight: float, random_state: int, enable_boosting_experiments: bool
) -> list[dict[str, Any]]:
    anomaly_contam = float(np.clip(1.0 / (1.0 + max(pos_weight, 1e-6)), 0.01, 0.20))
    experiments: list[dict[str, Any]] = [
        {
            "name": "LogReg_balanced_no_resample",
            "model": LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=random_state
            ),
            "sampler": None,
        },
        {
            "name": "LogReg_balanced_smote",
            "model": LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=random_state
            ),
            "sampler": SMOTE(random_state=random_state),
        },
        {
            "name": "LogReg_balanced_borderline_smote",
            "model": LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=random_state
            ),
            "sampler": BorderlineSMOTE(random_state=random_state),
        },
        {
            "name": "LogReg_balanced_adasyn",
            "model": LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=random_state
            ),
            "sampler": ADASYN(random_state=random_state),
        },
        {
            "name": "LogReg_balanced_smoteenn",
            "model": LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=random_state
            ),
            "sampler": SMOTEENN(random_state=random_state),
        },
        {
            "name": "RF_balanced_subsample",
            "model": RandomForestClassifier(
                n_estimators=160,
                max_depth=12,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=1,
                random_state=random_state,
            ),
            "sampler": None,
        },
        {
            "name": "IsolationForest_anomaly",
            "model": IsolationForestAnomalyDetector(
                n_estimators=180,
                contamination=anomaly_contam,
                random_state=random_state,
            ),
            "sampler": None,
        },
        {
            "name": "Autoencoder_anomaly",
            "model": AutoencoderAnomalyDetector(
                svd_components=40,
                hidden_layer_sizes=(24, 8, 24),
                max_iter=80,
                contamination=anomaly_contam,
                random_state=random_state,
            ),
            "sampler": None,
        },
    ]

    if enable_boosting_experiments and HAS_XGB:
        experiments.append(
            {
                "name": "XGB_scale_pos_weight",
                "model": XGBClassifier(
                    n_estimators=180,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_alpha=0.0,
                    reg_lambda=1.0,
                    min_child_weight=1,
                    gamma=0.0,
                    objective="binary:logistic",
                    eval_metric="aucpr",
                    scale_pos_weight=pos_weight,
                    n_jobs=2,
                    random_state=random_state,
                ),
                "sampler": None,
            }
        )
    if enable_boosting_experiments and HAS_LGBM:
        experiments.append(
            {
                "name": "LGBM_scale_pos_weight",
                "model": LGBMClassifier(
                    n_estimators=180,
                    learning_rate=0.05,
                    num_leaves=63,
                    min_child_samples=30,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    scale_pos_weight=pos_weight,
                    verbosity=-1,
                    n_jobs=2,
                    random_state=random_state,
                ),
                "sampler": None,
            }
        )
    if enable_boosting_experiments and HAS_CAT:
        experiments.append(
            {
                "name": "CatBoost_class_weights",
                "model": CatBoostClassifier(
                    iterations=150,
                    learning_rate=0.05,
                    depth=6,
                    eval_metric="AUC",
                    verbose=False,
                    logging_level="Silent",
                    allow_writing_files=False,
                    random_seed=random_state,
                    class_weights=(1.0, pos_weight),
                ),
                "sampler": None,
            }
        )
    return experiments


def _fit_pipeline(
    preprocessor: ColumnTransformer,
    experiment: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Any:
    model = clone(experiment["model"])
    sampler = experiment["sampler"]
    if sampler is None:
        pipe = Pipeline([("prep", clone(preprocessor)), ("model", model)])
    else:
        pipe = ImbPipeline(
            [("prep", clone(preprocessor)), ("sampler", clone(sampler)), ("model", model)]
        )
    pipe.fit(X_train, y_train)
    return pipe


def _evaluate_experiment_cv(
    preprocessor: ColumnTransformer,
    experiment: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int,
) -> dict[str, Any]:
    splitter = TimeSeriesSplit(n_splits=n_splits)
    rows: list[dict[str, float]] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(splitter.split(X_train), start=1):
        X_tr = X_train.iloc[tr_idx]
        y_tr = y_train.iloc[tr_idx]
        X_va = X_train.iloc[va_idx]
        y_va = y_train.iloc[va_idx]

        pipe = _fit_pipeline(preprocessor, experiment, X_tr, y_tr)
        proba = pipe.predict_proba(X_va)[:, 1]
        thr = _tune_threshold(y_va, proba)
        pred = (proba >= thr).astype(int)
        rows.append(
            {
                "fold": float(fold_idx),
                "pr_auc": _safe_pr_auc(y_va, proba),
                "roc_auc": _safe_roc_auc(y_va, proba),
                "recall": float(recall_score(y_va, pred, zero_division=0)),
                "precision": float(precision_score(y_va, pred, zero_division=0)),
                "f2": float(fbeta_score(y_va, pred, beta=2, zero_division=0)),
                "business_loss": _business_loss(y_va, pred),
                "threshold": thr,
            }
        )

    fold_df = pd.DataFrame(rows)
    return {
        "name": experiment["name"],
        "sampler": "None" if experiment["sampler"] is None else type(experiment["sampler"]).__name__,
        "mean_pr_auc": float(fold_df["pr_auc"].mean()),
        "std_pr_auc": float(fold_df["pr_auc"].std(ddof=0)),
        "mean_roc_auc": float(fold_df["roc_auc"].mean()),
        "std_roc_auc": float(fold_df["roc_auc"].std(ddof=0)),
        "mean_recall": float(fold_df["recall"].mean()),
        "std_recall": float(fold_df["recall"].std(ddof=0)),
        "mean_precision": float(fold_df["precision"].mean()),
        "std_precision": float(fold_df["precision"].std(ddof=0)),
        "mean_f2": float(fold_df["f2"].mean()),
        "std_f2": float(fold_df["f2"].std(ddof=0)),
        "mean_business_loss": float(fold_df["business_loss"].mean()),
        "std_business_loss": float(fold_df["business_loss"].std(ddof=0)),
        "mean_threshold": float(fold_df["threshold"].mean()),
        "fold_details": fold_df,
    }


def _select_best_experiment(cv_results: pd.DataFrame) -> str:
    ranked = cv_results.sort_values(
        ["mean_business_loss", "mean_f2", "mean_pr_auc"],
        ascending=[True, False, False],
    )
    return str(ranked.iloc[0]["name"])


def _evaluate_split(
    pipe: Any, X_eval: pd.DataFrame, y_eval: pd.Series, threshold: float
) -> dict[str, float]:
    proba = pipe.predict_proba(X_eval)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "pr_auc": _safe_pr_auc(y_eval, proba),
        "roc_auc": _safe_roc_auc(y_eval, proba),
        "recall": float(recall_score(y_eval, pred, zero_division=0)),
        "precision": float(precision_score(y_eval, pred, zero_division=0)),
        "f2": float(fbeta_score(y_eval, pred, beta=2, zero_division=0)),
        "business_loss": _business_loss(y_eval, pred),
    }


def _split_stability_table(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    best_pipe: Any,
    threshold: float,
    train_cv_metrics: dict[str, float] | None = None,
) -> pd.DataFrame:
    val_metrics = _evaluate_split(best_pipe, val_df, y_val, threshold)
    test_metrics = _evaluate_split(best_pipe, test_df, y_test, threshold)
    rows: list[dict[str, float | str]] = []
    if train_cv_metrics is not None:
        rows.append(
            {
                "split": "train_cv",
                "fraud_rate": float(y_train.mean()),
                "pr_auc": float(train_cv_metrics["mean_pr_auc"]),
                "roc_auc": float(train_cv_metrics["mean_roc_auc"]),
                "recall": float(train_cv_metrics["mean_recall"]),
                "precision": float(train_cv_metrics["mean_precision"]),
                "f2": float(train_cv_metrics["mean_f2"]),
                "business_loss": float(train_cv_metrics["mean_business_loss"]),
            }
        )
    else:
        train_metrics = _evaluate_split(best_pipe, train_df, y_train, threshold)
        rows.append({"split": "train_window", "fraud_rate": float(y_train.mean()), **train_metrics})

    rows.append({"split": "validation_window", "fraud_rate": float(y_val.mean()), **val_metrics})
    rows.append({"split": "final_test_window", "fraud_rate": float(y_test.mean()), **test_metrics})
    return pd.DataFrame(rows)


def _compute_drift_report(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    out = df.copy()
    out["TXN_DATE_TIME"] = pd.to_datetime(out["TXN_DATE_TIME"], errors="coerce")
    out["txn_month"] = out["TXN_DATE_TIME"].dt.to_period("M").astype("string")

    monthly = (
        out.groupby("txn_month", dropna=False)
        .agg(
            rows=("txn_month", "size"),
            fraud_rate=(target_col, "mean"),
            claim_amount_mean=("CLAIM_AMOUNT", "mean"),
            premium_amount_mean=("PREMIUM_AMOUNT", "mean"),
        )
        .reset_index()
    )
    monthly["fraud_rate_drift_abs"] = (
        monthly["fraud_rate"] - float(out[target_col].mean())
    ).abs()
    monthly["type"] = "monthly"

    state = (
        out.groupby("INCIDENT_STATE", dropna=False)
        .agg(rows=("INCIDENT_STATE", "size"), fraud_rate=(target_col, "mean"))
        .reset_index()
        .rename(columns={"INCIDENT_STATE": "segment"})
    )
    state["segment"] = "state::" + state["segment"].astype("string")
    state["type"] = "state"
    state["fraud_rate_drift_abs"] = (state["fraud_rate"] - float(out[target_col].mean())).abs()

    vendor = (
        out.groupby("VENDOR_ID", dropna=False)
        .agg(rows=("VENDOR_ID", "size"), fraud_rate=(target_col, "mean"))
        .reset_index()
        .rename(columns={"VENDOR_ID": "segment"})
    )
    vendor = vendor.sort_values("rows", ascending=False).head(25)
    vendor["segment"] = "vendor::" + vendor["segment"].astype("string")
    vendor["type"] = "vendor_top25"
    vendor["fraud_rate_drift_abs"] = (
        vendor["fraud_rate"] - float(out[target_col].mean())
    ).abs()

    monthly_out = monthly.rename(columns={"txn_month": "segment"})[
        ["type", "segment", "rows", "fraud_rate", "fraud_rate_drift_abs", "claim_amount_mean", "premium_amount_mean"]
    ]
    state_out = state[["type", "segment", "rows", "fraud_rate", "fraud_rate_drift_abs"]]
    state_out["claim_amount_mean"] = np.nan
    state_out["premium_amount_mean"] = np.nan
    vendor_out = vendor[["type", "segment", "rows", "fraud_rate", "fraud_rate_drift_abs"]]
    vendor_out["claim_amount_mean"] = np.nan
    vendor_out["premium_amount_mean"] = np.nan

    return pd.concat([monthly_out, state_out, vendor_out], ignore_index=True)


def _extract_defaults_and_dtypes(X: pd.DataFrame) -> tuple[dict[str, Any], dict[str, str]]:
    defaults: dict[str, Any] = {}
    dtypes: dict[str, str] = {}
    for c in X.columns:
        s = X[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            dtypes[c] = "datetime64[ns]"
            defaults[c] = None
        elif pd.api.types.is_numeric_dtype(s):
            dtypes[c] = "float64"
            defaults[c] = None
        elif pd.api.types.is_bool_dtype(s):
            dtypes[c] = "bool"
            defaults[c] = False
        else:
            dtypes[c] = "object"
            defaults[c] = "Unknown"
    return defaults, dtypes


def train(config_path: str | Path = "configs/config.yaml") -> dict[str, str]:
    cfg = _load_config(config_path)

    raw_dir = Path(cfg["data"]["raw_dir"])
    interim_dir = Path(cfg["data"]["interim_dir"])
    processed_dir = Path(cfg["data"]["processed_dir"])
    reports_dir = Path(cfg["reports"]["reports_dir"])
    fraud_model_path = Path(cfg["models"]["fraud_model_path"])
    severity_model_path = Path(cfg["models"]["severity_model_path"])
    metadata_path = Path(cfg["models"]["metadata_path"])
    test_size = float(cfg["training"].get("test_size", 0.15))
    val_size = float(cfg["training"].get("val_size", 0.15))
    rs = int(cfg["training"].get("random_state", 42))
    run_shap = bool(cfg["training"].get("run_shap", False))
    enable_boosting_experiments = bool(
        cfg["training"].get("enable_boosting_experiments", False)
    )
    requested_cv_splits = int(cfg["training"].get("cv_splits", 5))
    regression_search_n_iter = int(cfg["training"].get("regression_search_n_iter", 2))
    regression_cv_splits = int(cfg["training"].get("regression_cv_splits", 3))
    max_reg_search_rows = int(cfg["training"].get("max_reg_search_rows", 3000))

    for p in [interim_dir, processed_dir, reports_dir, fraud_model_path.parent]:
        p.mkdir(parents=True, exist_ok=True)
    fallback_dir = reports_dir / "runtime_fallback"
    fallback_dir.mkdir(parents=True, exist_ok=True)

    base = build_base_table(raw_dir)
    merged_out = _safe_save_csv(base, interim_dir / "claims_merged.csv", fallback_dir)
    feat = add_features(base)
    feat = _build_temporal_features(feat)
    features_out = _safe_save_csv(feat, processed_dir / "claims_features.csv", fallback_dir)

    cls_df = feat[feat["CLAIM_STATUS"].isin(["A", "D"])].copy()
    cls_df["target"] = (cls_df["CLAIM_STATUS"] == "D").astype(int)
    cls_df = _add_leakage_safe_behavioral_features(cls_df, target_col="target")
    cls_train, cls_val, cls_test = _temporal_split(cls_df, test_size=test_size, val_size=val_size)

    drop_base = {
        "CLAIM_STATUS",
        "target",
        "TRANSACTION_ID",
        "POLICY_NUMBER",
        "SSN",
        "CUSTOMER_NAME",
        "ACCT_NUMBER",
        "ROUTING_NUMBER",
        "EMP_ACCT_NUMBER",
        "EMP_ROUTING_NUMBER",
        "ADDRESS_LINE1",
        "ADDRESS_LINE1_AGENT",
        "ADDRESS_LINE1_VENDOR",
    }

    Xc_train = cls_train.drop(columns=[c for c in drop_base if c in cls_train.columns], errors="ignore")
    yc_train = cls_train["target"].astype(int)
    Xc_val = cls_val.drop(columns=[c for c in drop_base if c in cls_val.columns], errors="ignore")
    yc_val = cls_val["target"].astype(int)
    Xc_test = cls_test.drop(columns=[c for c in drop_base if c in cls_test.columns], errors="ignore")
    yc_test = cls_test["target"].astype(int)

    Xc_train = _normalize_features(Xc_train)
    Xc_val = _normalize_features(Xc_val)
    Xc_test = _normalize_features(Xc_test)

    leakage_cls = _leakage_audit_table(list(Xc_train.columns), task="classification")
    leakage_cls_out = _safe_save_csv(
        leakage_cls, reports_dir / "leakage_audit_classification.csv", fallback_dir
    )

    neg = int((yc_train == 0).sum())
    pos = int((yc_train == 1).sum())
    pos_weight = float(neg / max(pos, 1))

    preprocessor = _build_preprocessor(Xc_train)
    experiments = _build_experiments(
        pos_weight=pos_weight,
        random_state=rs,
        enable_boosting_experiments=enable_boosting_experiments,
    )

    n_splits = min(max(2, requested_cv_splits), max(2, len(Xc_train) // 250))
    if n_splits >= len(Xc_train):
        n_splits = max(2, len(Xc_train) - 1)

    cv_records: list[dict[str, Any]] = []
    fold_details: list[pd.DataFrame] = []
    failed_experiments: list[dict[str, str]] = []
    for exp in experiments:
        try:
            result = _evaluate_experiment_cv(
                preprocessor=preprocessor,
                experiment=exp,
                X_train=Xc_train,
                y_train=yc_train,
                n_splits=n_splits,
            )
            fold_df = result.pop("fold_details")
            fold_df.insert(0, "model", result["name"])
            fold_details.append(fold_df)
            cv_records.append(result)
        except Exception as exc:
            failed_experiments.append(
                {"model": str(exp["name"]), "error": str(exc).split("\n")[0][:260]}
            )

    if not cv_records:
        raise RuntimeError(
            "All classification experiments failed. Check preprocessing and model compatibility."
        )

    cv_df = pd.DataFrame(cv_records).sort_values(
        ["mean_business_loss", "mean_f2", "mean_pr_auc"],
        ascending=[True, False, False],
    )
    cv_out = _safe_save_csv(cv_df, reports_dir / "classification_cv_metrics.csv", fallback_dir)
    folds_out = _safe_save_csv(
        pd.concat(fold_details, ignore_index=True),
        reports_dir / "classification_cv_fold_metrics.csv",
        fallback_dir,
    )
    failed_df = pd.DataFrame(failed_experiments, columns=["model", "error"])
    failed_path = _safe_save_csv(
        failed_df,
        reports_dir / "classification_failed_experiments.csv",
        fallback_dir,
    )
    failed_exp_out = str(failed_path)

    best_name = _select_best_experiment(cv_df)
    best_exp = next(e for e in experiments if e["name"] == best_name)
    best_pipe = _fit_pipeline(preprocessor, best_exp, Xc_train, yc_train)
    best_cv_row = cv_df.loc[cv_df["name"] == best_name].iloc[0].to_dict()

    val_proba = best_pipe.predict_proba(Xc_val)[:, 1]
    best_thr = _tune_threshold(yc_val, val_proba)
    test_proba = best_pipe.predict_proba(Xc_test)[:, 1]
    test_pred = (test_proba >= best_thr).astype(int)

    split_stability = _split_stability_table(
        train_df=Xc_train,
        val_df=Xc_val,
        test_df=Xc_test,
        y_train=yc_train,
        y_val=yc_val,
        y_test=yc_test,
        best_pipe=best_pipe,
        threshold=best_thr,
        train_cv_metrics=best_cv_row,
    )
    split_stability_out = _safe_save_csv(
        split_stability, reports_dir / "split_stability_metrics.csv", fallback_dir
    )

    topk_df = _topk_table(yc_test, test_proba, TOPK_CAPACITIES)
    topk_out = _safe_save_csv(topk_df, reports_dir / "topk_investigation_metrics.csv", fallback_dir)

    drift_df = _compute_drift_report(cls_df, target_col="target")
    drift_out = _safe_save_csv(drift_df, reports_dir / "drift_report.csv", fallback_dir)

    perm_rows = min(len(Xc_test), 2500)
    if perm_rows < len(Xc_test):
        perm_idx = np.random.RandomState(rs).choice(len(Xc_test), perm_rows, replace=False)
        X_perm = Xc_test.iloc[perm_idx].copy()
        y_perm = yc_test.iloc[perm_idx].copy()
    else:
        X_perm = Xc_test
        y_perm = yc_test
    perm = permutation_importance(
        best_pipe,
        X_perm,
        y_perm,
        scoring="average_precision",
        n_repeats=3,
        random_state=rs,
        n_jobs=1,
    )
    perm_df = pd.DataFrame(
        {"feature": Xc_test.columns.astype(str), "importance": perm.importances_mean}
    ).sort_values("importance", ascending=False)
    perm_out = _safe_save_csv(
        perm_df.head(50), reports_dir / "permutation_importance_top50.csv", fallback_dir
    )

    shap_note = "SHAP skipped (run_shap is False)."
    if HAS_SHAP and run_shap:
        try:
            X_tx = best_pipe.named_steps["prep"].transform(Xc_test)
            model_obj = best_pipe.named_steps["model"]
            sample_n = min(500, X_tx.shape[0])
            idx = np.random.RandomState(rs).choice(X_tx.shape[0], sample_n, replace=False)
            X_sample = X_tx[idx]
            explainer = shap.Explainer(model_obj, X_sample)
            shap_values = explainer(X_sample)
            try:
                vals = np.abs(np.asarray(shap_values.values))
                mean_abs = vals.mean(axis=0)
                feat_names = (
                    best_pipe.named_steps["prep"].get_feature_names_out().astype(str)
                    if hasattr(best_pipe.named_steps["prep"], "get_feature_names_out")
                    else np.array([f"feature_{i}" for i in range(mean_abs.shape[0])], dtype=object)
                )
                shap_df = pd.DataFrame(
                    {"feature": feat_names[: len(mean_abs)], "mean_abs_shap": mean_abs}
                ).sort_values("mean_abs_shap", ascending=False)
                _safe_save_csv(shap_df.head(50), reports_dir / "shap_summary_top50.csv", fallback_dir)
                shap_note = "SHAP summary saved."
            except Exception:
                shap_note = "SHAP computed, but summary extraction skipped."
        except Exception:
            shap_note = "SHAP skipped due to runtime/model compatibility."

    reg_df = feat[feat["CLAIM_AMOUNT"].notna()].copy()
    reg_df = reg_df.sort_values("TXN_DATE_TIME").reset_index(drop=True)
    reg_train, _, reg_test = _temporal_split(reg_df, test_size=test_size, val_size=val_size)

    drop_reg = set(drop_base) | {"CLAIM_AMOUNT", "claim_to_premium_ratio"}
    Xr_train = reg_train.drop(columns=[c for c in drop_reg if c in reg_train.columns], errors="ignore")
    Xr_test = reg_test.drop(columns=[c for c in drop_reg if c in reg_test.columns], errors="ignore")

    yr_train = reg_train["CLAIM_AMOUNT"].astype(float)
    yr_test = reg_test["CLAIM_AMOUNT"].astype(float)
    yr_train_log = np.log1p(yr_train)
    yr_test_log = np.log1p(yr_test)

    Xr_train = _normalize_features(Xr_train)
    Xr_test = _normalize_features(Xr_test)

    leakage_reg = _leakage_audit_table(list(Xr_train.columns), task="regression")
    leakage_reg_out = _safe_save_csv(
        leakage_reg, reports_dir / "leakage_audit_regression.csv", fallback_dir
    )

    pre_reg = _build_preprocessor(Xr_train)
    ridge_pipe = Pipeline(
        [
            ("prep", clone(pre_reg)),
            ("model", RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0])),
        ]
    )
    rf_reg_pipe = Pipeline(
        [
            ("prep", clone(pre_reg)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=180,
                    max_depth=14,
                    min_samples_leaf=2,
                    n_jobs=1,
                    random_state=rs,
                ),
            ),
        ]
    )

    ridge_pipe.fit(Xr_train, yr_train_log)
    if regression_search_n_iter > 0 and len(Xr_train) >= 500:
        if len(Xr_train) > max_reg_search_rows:
            reg_idx = np.arange(len(Xr_train))[-max_reg_search_rows:]
            Xr_search = Xr_train.iloc[reg_idx].copy()
            yr_search = yr_train_log.iloc[reg_idx].copy()
        else:
            Xr_search = Xr_train
            yr_search = yr_train_log

        reg_splits = min(max(2, regression_cv_splits), max(2, len(Xr_search) // 350))
        if reg_splits >= len(Xr_search):
            reg_splits = max(2, len(Xr_search) - 1)

        rf_reg_search = RandomizedSearchCV(
            rf_reg_pipe,
            param_distributions={
                "model__n_estimators": [120, 180, 240],
                "model__max_depth": [8, 12, 14, None],
                "model__min_samples_leaf": [1, 2, 5],
            },
            n_iter=regression_search_n_iter,
            cv=TimeSeriesSplit(n_splits=reg_splits),
            scoring="neg_mean_absolute_error",
            n_jobs=1,
            random_state=rs,
        )
        rf_reg_search.fit(Xr_search, yr_search)
        best_rf_reg = rf_reg_search.best_estimator_
    else:
        rf_reg_pipe.fit(Xr_train, yr_train_log)
        best_rf_reg = rf_reg_pipe

    reg_rows: list[dict[str, Any]] = []
    reg_models = {"RidgeCV": ridge_pipe, "RandomForestRegressor": best_rf_reg}
    for name, model in reg_models.items():
        pred_log = model.predict(Xr_test)
        pred = np.expm1(pred_log)
        reg_rows.append(
            {
                "Model": name,
                "Test_MAE": float(mean_absolute_error(yr_test, pred)),
                "Test_RMSE": float(np.sqrt(mean_squared_error(yr_test, pred))),
                "Test_R2_log": float(r2_score(yr_test_log, pred_log)),
            }
        )
    reg_metrics_df = pd.DataFrame(reg_rows).sort_values(["Test_MAE", "Test_RMSE"], ascending=[True, True])
    reg_metrics_out = _safe_save_csv(
        reg_metrics_df, reports_dir / "regression_test_metrics.csv", fallback_dir
    )
    best_reg_name = str(reg_metrics_df.iloc[0]["Model"])
    best_regressor = reg_models[best_reg_name]

    fraud_out = _safe_dump_joblib(best_pipe, fraud_model_path, fallback_dir)
    severity_out = _safe_dump_joblib(best_regressor, severity_model_path, fallback_dir)

    cls_defaults, cls_dtypes = _extract_defaults_and_dtypes(Xc_train)
    reg_defaults, reg_dtypes = _extract_defaults_and_dtypes(Xr_train)
    metadata = {
        "class_features": list(Xc_train.columns),
        "reg_features": list(Xr_train.columns),
        "class_dtypes": cls_dtypes,
        "reg_dtypes": reg_dtypes,
        "class_defaults": cls_defaults,
        "reg_defaults": reg_defaults,
        "threshold": float(best_thr),
        "reject_threshold": 0.70,
        "split": "time_based",
        "fraud_model_name": best_name,
        "severity_model_name": best_reg_name,
    }
    metadata_out = _safe_write_text(json.dumps(metadata, indent=2), metadata_path, fallback_dir)

    test_metrics = _evaluate_split(best_pipe, Xc_test, yc_test, best_thr)
    cls_test_df = pd.DataFrame(
        [
            {
                "Model": best_name,
                "Threshold": float(best_thr),
                "Accuracy": float((test_pred == yc_test.to_numpy()).mean()),
                "Precision": test_metrics["precision"],
                "Recall": test_metrics["recall"],
                "F2": test_metrics["f2"],
                "ROC_AUC": test_metrics["roc_auc"],
                "PR_AUC": test_metrics["pr_auc"],
                "Business_Loss": test_metrics["business_loss"],
            }
        ]
    )
    cls_test_out = _safe_save_csv(
        cls_test_df, reports_dir / "classification_test_metrics.csv", fallback_dir
    )

    summary = {
        "best_classifier": best_name,
        "best_threshold": float(best_thr),
        "best_sampler": str(cv_df.iloc[0]["sampler"]),
        "classification_test_metrics": cls_test_df.iloc[0].to_dict(),
        "best_regressor": best_reg_name,
        "regression_test_metrics": reg_metrics_df.iloc[0].to_dict(),
        "topk_summary": topk_df.to_dict(orient="records"),
        "notes": [
            "Model selection was performed with temporal CV.",
            "Final threshold was tuned on validation window only.",
            "Business loss uses FN cost 50000 and FP cost 500.",
            shap_note,
        ],
    }
    summary_out = _safe_write_text(
        json.dumps(summary, indent=2), reports_dir / "final_model_summary.json", fallback_dir
    )

    metrics_out = _safe_write_text(
        json.dumps(
            {
                "classification": test_metrics,
                "regression": reg_metrics_df.iloc[0].to_dict(),
                "best_classifier": best_name,
                "best_regressor": best_reg_name,
                "threshold": float(best_thr),
            },
            indent=2,
        ),
        reports_dir / "metrics.json",
        fallback_dir,
    )

    model_card = [
        "# Model Card",
        "",
        "## Champions",
        f"- Fraud: {best_name}",
        f"- Severity: {best_reg_name}",
        "",
        "## Validation Policy",
        "- Time-based train/validation/test split.",
        "- TimeSeriesSplit used for model comparison on train window.",
        "- Threshold tuned on validation window with F2 + business-loss objective.",
        "",
        "## Key Holdout Metrics",
        f"- PR-AUC: {test_metrics['pr_auc']:.4f}",
        f"- ROC-AUC: {test_metrics['roc_auc']:.4f}",
        f"- Recall: {test_metrics['recall']:.4f}",
        f"- F2: {test_metrics['f2']:.4f}",
        f"- Business Loss: {test_metrics['business_loss']:.2f}",
        f"- Regression MAE: {float(reg_metrics_df.iloc[0]['Test_MAE']):.2f}",
        f"- Regression RMSE: {float(reg_metrics_df.iloc[0]['Test_RMSE']):.2f}",
    ]
    model_card_out = _safe_write_text(
        "\n".join(model_card), reports_dir / "model_card.md", fallback_dir
    )

    return {
        "claims_merged_path": str(merged_out),
        "claims_features_path": str(features_out),
        "fraud_model_path": str(fraud_out),
        "severity_model_path": str(severity_out),
        "metadata_path": str(metadata_out),
        "metrics_path": str(metrics_out),
        "model_card_path": str(model_card_out),
        "cv_metrics_path": str(cv_out),
        "cv_fold_metrics_path": str(folds_out),
        "classification_test_metrics_path": str(cls_test_out),
        "regression_test_metrics_path": str(reg_metrics_out),
        "split_stability_path": str(split_stability_out),
        "topk_metrics_path": str(topk_out),
        "drift_report_path": str(drift_out),
        "leakage_audit_classification_path": str(leakage_cls_out),
        "leakage_audit_regression_path": str(leakage_reg_out),
        "permutation_importance_path": str(perm_out),
        "summary_path": str(summary_out),
        "failed_experiments_path": failed_exp_out,
    }


if __name__ == "__main__":
    out = train()
    print("Training complete.")
    for k, v in out.items():
        print(f"{k}: {v}")
