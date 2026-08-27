import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "LOSS_DT" in df.columns:
        df["LOSS_DT"] = pd.to_datetime(df["LOSS_DT"], errors="coerce")
    if "REPORT_DT" in df.columns:
        df["REPORT_DT"] = pd.to_datetime(df["REPORT_DT"], errors="coerce")
    if "LOSS_DT" in df.columns and "REPORT_DT" in df.columns:
        df["days_to_report"] = (df["REPORT_DT"] - df["LOSS_DT"]).dt.days
    else:
        df["days_to_report"] = np.nan

    if "CLAIM_AMOUNT" in df.columns and "PREMIUM_AMOUNT" in df.columns:
        df["claim_to_premium_ratio"] = df["CLAIM_AMOUNT"] / df[
            "PREMIUM_AMOUNT"
        ].replace(0, np.nan)
    else:
        df["claim_to_premium_ratio"] = np.nan

    if "TXN_DATE_TIME" in df.columns:
        df["TXN_DATE_TIME"] = pd.to_datetime(df["TXN_DATE_TIME"], errors="coerce")
    if "DATE_OF_JOINING" in df.columns:
        df["DATE_OF_JOINING"] = pd.to_datetime(df["DATE_OF_JOINING"], errors="coerce")
    if "POLICY_EFF_DT" in df.columns:
        df["POLICY_EFF_DT"] = pd.to_datetime(df["POLICY_EFF_DT"], errors="coerce")

    if "TXN_DATE_TIME" in df.columns and "DATE_OF_JOINING" in df.columns:
        df["agent_experience_years"] = (
            df["TXN_DATE_TIME"] - df["DATE_OF_JOINING"]
        ).dt.days / 365.25
    else:
        df["agent_experience_years"] = np.nan

    if "TXN_DATE_TIME" in df.columns and "POLICY_EFF_DT" in df.columns:
        df["policy_age_years"] = (
            df["TXN_DATE_TIME"] - df["POLICY_EFF_DT"]
        ).dt.days / 365.25
    else:
        df["policy_age_years"] = np.nan

    return df
