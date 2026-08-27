from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED_SOURCE_COLUMNS = {
    "insurance_data.csv": {"AGENT_ID", "VENDOR_ID"},
    "employee_data.csv": {"AGENT_ID"},
    "vendor_data.csv": {"VENDOR_ID"},
}


def _assert_file_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required source file: {path}")


def _assert_required_columns(df: pd.DataFrame, required_cols: set[str], source: str) -> None:
    missing = sorted(required_cols.difference(df.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def load_raw_data(
    data_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)

    insurance_path = data_dir / "insurance_data.csv"
    employee_path = data_dir / "employee_data.csv"
    vendor_path = data_dir / "vendor_data.csv"

    for source_path in (insurance_path, employee_path, vendor_path):
        _assert_file_exists(source_path)

    insurance = pd.read_csv(insurance_path)
    employee = pd.read_csv(employee_path)
    vendor = pd.read_csv(vendor_path)

    _assert_required_columns(
        insurance,
        REQUIRED_SOURCE_COLUMNS["insurance_data.csv"],
        "insurance_data.csv",
    )
    _assert_required_columns(
        employee,
        REQUIRED_SOURCE_COLUMNS["employee_data.csv"],
        "employee_data.csv",
    )
    _assert_required_columns(
        vendor,
        REQUIRED_SOURCE_COLUMNS["vendor_data.csv"],
        "vendor_data.csv",
    )

    return insurance, employee, vendor


def merge_data(
    insurance: pd.DataFrame, employee: pd.DataFrame, vendor: pd.DataFrame
) -> pd.DataFrame:
    merged = insurance.merge(
        employee, on="AGENT_ID", how="left", suffixes=("", "_AGENT")
    )
    merged = merged.merge(vendor, on="VENDOR_ID", how="left", suffixes=("", "_VENDOR"))
    return merged


def standardize_strings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        series = df[col].astype("string").str.strip()
        df[col] = series.where(series.notna(), np.nan).astype("object")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "VENDOR_ID" in df.columns:
        df["VENDOR_ID"] = df["VENDOR_ID"].fillna("Unknown")

    if "CUSTOMER_EDUCATION_LEVEL" in df.columns:
        df["CUSTOMER_EDUCATION_LEVEL"] = df["CUSTOMER_EDUCATION_LEVEL"].fillna(
            "Unknown"
        )

    if "AUTHORITY_CONTACTED" in df.columns:
        df["AUTHORITY_CONTACTED_MISSING"] = df["AUTHORITY_CONTACTED"].isna().astype(int)
        df["AUTHORITY_CONTACTED"] = df["AUTHORITY_CONTACTED"].fillna("Unknown")

    if "ADDRESS_LINE2" in df.columns:
        df = df.drop(columns=["ADDRESS_LINE2"])

    if "TRANSACTION_ID" in df.columns:
        df = df.drop_duplicates(subset=["TRANSACTION_ID"])

    return df


def build_base_table(data_dir: str | Path) -> pd.DataFrame:
    insurance, employee, vendor = load_raw_data(data_dir)
    merged = merge_data(insurance, employee, vendor)
    merged = standardize_strings(merged)
    merged = clean_data(merged)
    return merged
