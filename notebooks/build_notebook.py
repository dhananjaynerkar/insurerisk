import nbformat as nbf
import os
from pathlib import Path

nb = nbf.v4.new_notebook()

cells = []

def add_md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def add_code(text):
    cells.append(nbf.v4.new_code_cell(text))

add_md("""# Full ML Pipeline: Fraud Detection and Severity Prediction

This notebook demonstrates a complete, end-to-end Machine Learning pipeline optimized for highly imbalanced data. 
We will go through:
1. Data Loading
2. Exploratory Data Analysis (EDA)
3. Data Preprocessing & Cleaning (using Modular Functions)
4. Feature Engineering
5. Advanced Feature Engineering (Post Train/Test Split to prevent Data Leakage)
6. Visualization
7. Model Building with Pipelines (using RandomUnderSampler for extreme speed & balance)
8. Optimal Threshold Tuning (maximizing recall/F1 instead of default 0.5)
9. Evaluation & Explainability (SHAP, ROC-AUC, PR-AUC)
10. Saving Final Production-Ready Models""")

add_code("""import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
)

# Imblearn for Pipeline with Undersampling
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, mean_absolute_error,
    mean_squared_error, r2_score, roc_curve, precision_recall_curve, auc, f1_score
)

import shap
from scipy.stats import ks_2samp""")

add_md("""## 1. Load Data
We load the raw datasets: `insurance_data`, `employee_data`, and `vendor_data`.""")

add_code("""project_root = Path.cwd()
if not (project_root / 'data' / 'raw' / 'insurance_data.csv').exists():
    project_root = project_root.parent

data_dir = project_root / 'data' / 'raw'

# Load CSVs
insurance = pd.read_csv(data_dir / 'insurance_data.csv')
employee = pd.read_csv(data_dir / 'employee_data.csv')
vendor = pd.read_csv(data_dir / 'vendor_data.csv')

print('Loaded datasets:')
print(f'Insurance shape: {insurance.shape}')
print(f'Employee shape: {employee.shape}')
print(f'Vendor shape: {vendor.shape}')

# Merge datasets
merged = insurance.merge(employee, on="AGENT_ID", how="left", suffixes=("", "_AGENT"))
merged = merged.merge(vendor, on="VENDOR_ID", how="left", suffixes=("", "_VENDOR"))
print(f"Merged shape: {merged.shape}")""")

add_md("""## 2. Exploratory Data Analysis (EDA)
Let's explore the target variables and identify missing values.""")

add_code("""# Define helper function for basic data checks
def summarize_dataframe(df):
    print(f"Total rows: {len(df)}")
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if len(missing) > 0:
        print("\\nMissing values per column:")
        print(missing.head(10))
    else:
        print("\\nNo missing values found.")
        
    dupes = df.duplicated(subset=["TRANSACTION_ID"]).sum()
    print(f"\\nDuplicate TRANSACTION_IDs: {dupes}")

summarize_dataframe(merged)""")

add_code("""# Plot Target Distribution
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.countplot(data=merged, x='CLAIM_STATUS', palette='viridis')
plt.title('Distribution of Claim Status (Fraud vs Legit)')

plt.subplot(1, 2, 2)
sns.histplot(merged['CLAIM_AMOUNT'].dropna(), bins=50, kde=True, color='blue')
plt.title('Distribution of Claim Amounts (Severity)')

plt.tight_layout()
plt.show()""")

add_md("""## 3. Data Preprocessing & Cleaning
We use functions to encapsulate our data cleaning steps. This makes the code modular, reusable, and easy to explain.
- Fill missing categorical values.
- Create missing indicator columns.
- Remove high-missing columns.
- Remove duplicates.
- Cap Outliers using the Interquartile Range (IQR) method.""")

add_code("""def clean_data(df):
    cleaned = df.copy()
    
    # Strip whitespace from string columns and convert empties to NaN
    for col in cleaned.select_dtypes(include=["object", "string"]).columns:
        series = cleaned[col].astype("string").str.strip()
        cleaned[col] = series.where(series.notna(), np.nan).astype("object")

    # Fill specific categoricals
    if "VENDOR_ID" in cleaned.columns:
        cleaned["VENDOR_ID"] = cleaned["VENDOR_ID"].fillna("Unknown")
    
    if "CUSTOMER_EDUCATION_LEVEL" in cleaned.columns:
        cleaned["CUSTOMER_EDUCATION_LEVEL"] = cleaned["CUSTOMER_EDUCATION_LEVEL"].fillna("Unknown")
        
    # Create missing indicator for authority contacted
    if "AUTHORITY_CONTACTED" in cleaned.columns:
        cleaned["AUTHORITY_CONTACTED_MISSING"] = cleaned["AUTHORITY_CONTACTED"].isna().astype(int)
        cleaned["AUTHORITY_CONTACTED"] = cleaned["AUTHORITY_CONTACTED"].fillna("Unknown")
        
    # Drop largely empty or unneeded columns
    if "ADDRESS_LINE2" in cleaned.columns:
        cleaned = cleaned.drop(columns=["ADDRESS_LINE2"])
        
    # Drop duplicate transaction records
    if "TRANSACTION_ID" in cleaned.columns:
        cleaned = cleaned.drop_duplicates(subset=["TRANSACTION_ID"])
        
    return cleaned

def cap_outliers_iqr(df, cols, multiplier=1.5):
    capped_df = df.copy()
    for col in cols:
        if col in capped_df.columns and capped_df[col].dtype in [np.float64, np.int64]:
            q1 = capped_df[col].quantile(0.25)
            q3 = capped_df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - multiplier * iqr
            upper = q3 + multiplier * iqr
            capped_df[col] = capped_df[col].clip(lower, upper)
    return capped_df

cleaned = clean_data(merged)
outlier_cols = ["CLAIM_AMOUNT", "PREMIUM_AMOUNT", "AGE", "TENURE"]
cleaned = cap_outliers_iqr(cleaned, outlier_cols)

print("Data Cleaning Complete.")
summarize_dataframe(cleaned)""")

add_md("""## 4. Feature Engineering
We will engineer temporal features based on dates (e.g., policy age, agent experience).""")

add_code("""def engineer_features(df):
    featured = df.copy()
    
    featured["LOSS_DT"] = pd.to_datetime(featured["LOSS_DT"], errors="coerce")
    featured["REPORT_DT"] = pd.to_datetime(featured["REPORT_DT"], errors="coerce")
    featured["days_to_report"] = (featured["REPORT_DT"] - featured["LOSS_DT"]).dt.days
    
    featured["claim_to_premium_ratio"] = featured["CLAIM_AMOUNT"] / featured["PREMIUM_AMOUNT"].replace(0, np.nan)
    
    featured["TXN_DATE_TIME"] = pd.to_datetime(featured["TXN_DATE_TIME"], errors="coerce")
    featured["DATE_OF_JOINING"] = pd.to_datetime(featured["DATE_OF_JOINING"], errors="coerce")
    featured["POLICY_EFF_DT"] = pd.to_datetime(featured["POLICY_EFF_DT"], errors="coerce")
    
    featured["agent_experience_years"] = (featured["TXN_DATE_TIME"] - featured["DATE_OF_JOINING"]).dt.days / 365.25
    featured["policy_age_years"] = (featured["TXN_DATE_TIME"] - featured["POLICY_EFF_DT"]).dt.days / 365.25
    
    return featured

featured = engineer_features(cleaned)
featured[["days_to_report", "claim_to_premium_ratio", "agent_experience_years", "policy_age_years"]].head()""")

add_md("""## 5. Train / Test Split (Crucial for Preventing Target Leakage)
Before we calculate target-based statistical features (like agent fraud rates), we MUST split the dataset. If we compute average fraud rate per agent using the whole dataset, our model will inadvertently see test-set labels during training!""")

add_code("""# Define columns to drop
PII_COLS = ["CUSTOMER_NAME", "SSN", "ACCT_NUMBER", "ROUTING_NUMBER", "EMP_ACCT_NUMBER", "EMP_ROUTING_NUMBER", "ADDRESS_LINE1"]
ID_COLS = ["TRANSACTION_ID", "CUSTOMER_ID", "POLICY_NUMBER"]
DATE_COLS = ["TXN_DATE_TIME", "POLICY_EFF_DT", "LOSS_DT", "REPORT_DT", "DATE_OF_JOINING"]

drop_common = [c for c in (PII_COLS + ID_COLS + DATE_COLS) if c in featured.columns]

# Classification Dataset Setup
df_class = featured[featured["CLAIM_STATUS"].isin(["A", "D"])].copy()
y_class = df_class["CLAIM_STATUS"].map({"A": 0, "D": 1})
X_class = df_class.drop(columns=["CLAIM_STATUS"] + drop_common, errors="ignore")
X_class = X_class.replace({pd.NA: np.nan})

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42, stratify=y_class
)

# Regression Dataset Setup
df_reg = featured[featured["CLAIM_AMOUNT"].notna()].copy()
y_reg = df_reg["CLAIM_AMOUNT"]
X_reg = df_reg.drop(columns=["CLAIM_STATUS", "CLAIM_AMOUNT", "claim_to_premium_ratio"] + drop_common, errors="ignore")
X_reg = X_reg.replace({pd.NA: np.nan})

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

print(f"Classification Train: {X_train_cls.shape}, Test: {X_test_cls.shape}")
print(f"Regression Train: {X_train_reg.shape}, Test: {X_test_reg.shape}")""")

add_md("""## 6. Advanced Feature Engineering (Post-Split)
Now that the data is split, we calculate fraud behavioral signals purely on the training set, and map those values to both the training and test set.""")

add_code("""def add_behavioral_features(X_train, y_train, X_test):
    # Combine train X and y to compute rates safely
    train_df = X_train.copy()
    train_df['TARGET'] = y_train
    
    # Calculate Agent Fraud Rate
    agent_stats = train_df.groupby('AGENT_ID')['TARGET'].agg(['sum', 'count'])
    agent_stats['agent_fraud_rate'] = agent_stats['sum'] / agent_stats['count']
    
    # Calculate Vendor Fraud Rate
    vendor_stats = train_df.groupby('VENDOR_ID')['TARGET'].agg(['sum', 'count'])
    vendor_stats['vendor_fraud_rate'] = vendor_stats['sum'] / vendor_stats['count']
    
    # Calculate State Fraud Rate
    state_stats = train_df.groupby('INCIDENT_STATE')['TARGET'].agg(['sum', 'count'])
    state_stats['state_fraud_rate'] = state_stats['sum'] / state_stats['count']
    
    # Calculate Agent Claim Velocity
    agent_vol = train_df.groupby('AGENT_ID').size().reset_index(name='agent_volume')

    # Apply mapping function
    def map_features(df):
        df = df.copy()
        df['agent_fraud_rate'] = df['AGENT_ID'].map(agent_stats['agent_fraud_rate']).fillna(agent_stats['agent_fraud_rate'].mean())
        df['vendor_fraud_rate'] = df['VENDOR_ID'].map(vendor_stats['vendor_fraud_rate']).fillna(vendor_stats['vendor_fraud_rate'].mean())
        df['state_fraud_rate'] = df['INCIDENT_STATE'].map(state_stats['state_fraud_rate']).fillna(state_stats['state_fraud_rate'].mean())
        
        df = df.merge(agent_vol, on='AGENT_ID', how='left')
        df['agent_volume'] = df['agent_volume'].fillna(0)
        return df
        
    X_train_enhanced = map_features(X_train)
    X_test_enhanced = map_features(X_test)
    
    return X_train_enhanced, X_test_enhanced

X_train_cls, X_test_cls = add_behavioral_features(X_train_cls, y_train_cls, X_test_cls)
print("Behavioral features successfully added without data leakage!")
X_train_cls[['AGENT_ID', 'agent_fraud_rate', 'vendor_fraud_rate', 'agent_volume']].head()""")

add_md("""## 7. Modeling Setup: Preprocessing & Pipelines
We use `imblearn.pipeline.Pipeline` which allows us to add `RandomUnderSampler` directly into the cross-validation pipeline.
**Why Undersampling instead of SMOTE?**
- SMOTE takes very long on large datasets and interpolates between points, which can be noisy for categorical one-hot encoded variables.
- Undersampling balances the classes instantly by dropping the majority class to match the minority class size. This results in **lightning fast** training times while drastically boosting fraud recall.""")

add_code("""# Define feature types
cat_cols = list(X_train_cls.select_dtypes(include=["object", "category", "string"]).columns)
num_cols = [c for c in X_train_cls.columns if c not in cat_cols]

# Create transformers
num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", num_transformer, num_cols),
    ("cat", cat_transformer, cat_cols)
])

# Display preprocessor
preprocessor""")

add_md("""## 8. Legacy Baseline Removed
The earlier notebook-only modeling path was removed to keep this notebook clean and avoid duplicate logic.
We now focus on one production-style advanced workflow in Section 14.""")

add_md("""## 14. Advanced End-to-End Workflow (Interview-Ready)
This section performs a production-style classification workflow for `CLAIM_STATUS`:
- Deep data understanding and EDA
- Leakage-aware preprocessing + feature engineering
- Imbalance benchmarking (`SMOTE`, `SMOTETomek`, `SMOTEENN`, `ADASYN`, `RandomOverSampler`)
- Model comparison (`LogReg`, `DecisionTree`, `RandomForest`, `XGBoost`, `LightGBM`, `CatBoost`)
- Threshold-aware evaluation + explainability + deployment artifacts

This is intentionally practical and compact so it remains runnable in an interview setting.""")

add_code("""import warnings
warnings.filterwarnings('ignore')

import time
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    recall_score,
    precision_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.combine import SMOTETomek, SMOTEENN

# Optional boosted libraries
HAS_XGB = HAS_LGBM = HAS_CAT = False
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    pass
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    pass
try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except Exception:
    pass

print(f"Optional libs loaded -> XGBoost: {HAS_XGB}, LightGBM: {HAS_LGBM}, CatBoost: {HAS_CAT}")""")

add_code("""# ---------------------------
# 14.1 Data Understanding
# ---------------------------
df_adv = featured.copy()
df_adv = df_adv[df_adv['CLAIM_STATUS'].isin(['A', 'D'])].copy()
df_adv['TARGET_FRAUD'] = df_adv['CLAIM_STATUS'].map({'A': 0, 'D': 1})

print("Dataset shape:", df_adv.shape)
print("\\nData types summary:")
print(df_adv.dtypes.value_counts())

missing_summary = df_adv.isna().sum().sort_values(ascending=False)
missing_summary = missing_summary[missing_summary > 0]
print("\\nTop missing columns:")
print(missing_summary.head(15))

dup_count = int(df_adv.duplicated(subset=['TRANSACTION_ID']).sum()) if 'TRANSACTION_ID' in df_adv.columns else int(df_adv.duplicated().sum())
print("\\nDuplicate rows (transaction-level where possible):", dup_count)

cardinality = df_adv.nunique(dropna=True).sort_values(ascending=False)
print("\\nTop 20 high-cardinality columns:")
print(cardinality.head(20))

num_cols_all = df_adv.select_dtypes(include=[np.number]).columns.tolist()
skew_series = df_adv[num_cols_all].skew(numeric_only=True).sort_values(ascending=False)
print("\\nTop skewed numerical features:")
print(skew_series.head(10))

# Simple outlier scan using IQR
outlier_counts = {}
for c in num_cols_all:
    s = df_adv[c].dropna()
    if len(s) < 5:
        continue
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outlier_counts[c] = int(((s < lower) | (s > upper)).sum())
outlier_df = pd.Series(outlier_counts).sort_values(ascending=False)
print("\\nTop outlier-prone features (IQR count):")
print(outlier_df.head(10))

obj_cols = df_adv.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
datetime_cols = [c for c in df_adv.columns if 'DT' in c or 'DATE' in c or 'TIME' in c]
binary_cols = [c for c in df_adv.columns if df_adv[c].dropna().nunique() == 2]
low_card_cat = [c for c in obj_cols if df_adv[c].nunique(dropna=True) <= 15]
high_card_cat = [c for c in obj_cols if df_adv[c].nunique(dropna=True) > 15]
ordinal_candidates = [c for c in ['INCIDENT_SEVERITY', 'RISK_SEGMENTATION', 'SOCIAL_CLASS'] if c in df_adv.columns]

print("\\nColumn groups:")
print("Numerical:", len(num_cols_all))
print("Categorical:", len(obj_cols))
print("Low-card categorical:", len(low_card_cat))
print("High-card categorical:", len(high_card_cat))
print("Binary-like:", len(binary_cols))
print("Datetime-like:", datetime_cols)
print("Ordinal candidates:", ordinal_candidates)

leakage_cols = [c for c in [
    'CLAIM_STATUS',
    'TARGET_FRAUD',
    'TRANSACTION_ID',
    'CUSTOMER_ID',
    'POLICY_NUMBER',
    'CUSTOMER_NAME',
    'SSN',
    'ACCT_NUMBER',
    'ROUTING_NUMBER',
    'EMP_ACCT_NUMBER',
    'EMP_ROUTING_NUMBER',
    'ADDRESS_LINE1',
    'TXN_DATE_TIME',
    'POLICY_EFF_DT',
    'LOSS_DT',
    'REPORT_DT',
    'DATE_OF_JOINING',
] if c in df_adv.columns]

target_rate = float(df_adv['TARGET_FRAUD'].mean())
ratio = (1 - target_rate) / max(target_rate, 1e-9)
print(f"\\nTarget fraud rate: {target_rate:.4f} ({target_rate*100:.2f}%) | Majority:Minority ~= {ratio:.1f}:1")""")

add_code("""# ---------------------------
# 14.2 EDA Visuals + Interpretation
# ---------------------------

# 1) Missing heatmap
plt.figure(figsize=(14, 5))
sns.heatmap(df_adv[missing_summary.index[:20]].isna(), cbar=False)
plt.title("Missing Value Heatmap (Top Missing Columns)")
plt.show()
print("Purpose: Find systematic data-quality gaps. Interpretation: clustered gaps suggest process-level missingness. Business impact: missing-heavy fields may weaken risk scoring if untreated.")

# 2) Target distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='TARGET_FRAUD', data=df_adv)
plt.title("Target Distribution (0=Non-Fraud, 1=Fraud)")
plt.show()
print("Purpose: quantify imbalance severity. Interpretation: minority fraud class is small. Business impact: naive accuracy is misleading; recall/PR focus is necessary.")

# 3) Correlation heatmap (top numerical subset)
num_for_corr = [c for c in num_cols_all if c not in ['TARGET_FRAUD']]
num_for_corr = num_for_corr[:20]
corr_df = df_adv[num_for_corr + ['TARGET_FRAUD']].corr(numeric_only=True)
plt.figure(figsize=(12, 8))
sns.heatmap(corr_df, cmap='coolwarm', center=0)
plt.title("Correlation Heatmap (Numerical + Target)")
plt.show()
print("Purpose: detect linear relationships and redundancy. Interpretation: weak linear links indicate non-linear models may help. Business impact: feature interactions matter.")

# 4) Numerical distributions
dist_cols = [c for c in ['CLAIM_AMOUNT', 'PREMIUM_AMOUNT', 'AGE', 'TENURE', 'days_to_report'] if c in df_adv.columns]
if dist_cols:
    df_adv[dist_cols].hist(figsize=(12, 6), bins=30)
    plt.suptitle("Numerical Feature Distributions")
    plt.tight_layout()
    plt.show()
    print("Purpose: inspect skewness/tails. Interpretation: heavy tails justify robust scaling/log-like treatment. Business impact: prevents instability from extreme claims.")

# 5) Boxplots for outliers
if dist_cols:
    plt.figure(figsize=(12, 5))
    df_adv[dist_cols].boxplot(rot=45)
    plt.title("Outlier Detection via Boxplots")
    plt.show()
    print("Purpose: visualize outlier spread. Interpretation: claim-related columns have outlier concentration. Business impact: robust preprocessing protects model reliability.")

# 6) Feature vs target
if 'claim_to_premium_ratio' in df_adv.columns:
    plt.figure(figsize=(7, 4))
    sns.boxplot(x='TARGET_FRAUD', y='claim_to_premium_ratio', data=df_adv)
    plt.title("Claim-to-Premium Ratio vs Fraud Target")
    plt.show()
    print("Purpose: compare risk signal by class. Interpretation: higher ratio tends to align with fraud risk. Business impact: useful review trigger.")

# 7) Categorical frequency plots
cat_plot_cols = [c for c in ['INSURANCE_TYPE', 'INCIDENT_SEVERITY', 'INCIDENT_STATE'] if c in df_adv.columns]
for c in cat_plot_cols:
    plt.figure(figsize=(8, 4))
    vc = df_adv[c].astype('string').value_counts().head(12)
    sns.barplot(x=vc.index, y=vc.values)
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Top Categories: {c}")
    plt.show()
print("Purpose: find dominant categories and rare levels. Business impact: informs low-card vs high-card encoding strategy.")

# 8) Pairplot (limited important features)
pair_cols = [c for c in ['CLAIM_AMOUNT', 'PREMIUM_AMOUNT', 'days_to_report', 'claim_to_premium_ratio', 'TARGET_FRAUD'] if c in df_adv.columns]
if len(pair_cols) >= 3:
    sns.pairplot(df_adv[pair_cols].dropna().sample(min(800, len(df_adv)), random_state=42), hue='TARGET_FRAUD', diag_kind='hist')
    plt.show()
    print("Purpose: quickly inspect class separability and interactions. Business impact: helps choose tree-based models for non-linear boundaries.")""")

add_code("""# ---------------------------
# 14.3 Setup: Features + Preprocessor + Train/Test Split
# ---------------------------
from sklearn.utils.validation import check_is_fitted

class HighCardEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing=20.0):
        self.smoothing = smoothing

    def fit(self, X, y):
        X = pd.DataFrame(X).copy()
        y = pd.Series(y).reset_index(drop=True)
        self.cols_ = list(X.columns)
        self.global_mean_ = float(y.mean())
        self.freq_maps_ = {}
        self.te_maps_ = {}
        for c in self.cols_:
            s = X[c].astype('string').fillna('UNK')
            self.freq_maps_[c] = s.value_counts(normalize=True).to_dict()
            stats = pd.DataFrame({'k': s, 'y': y}).groupby('k')['y'].agg(['mean', 'count'])
            smooth = (stats['mean'] * stats['count'] + self.global_mean_ * self.smoothing) / (stats['count'] + self.smoothing)
            self.te_maps_[c] = smooth.to_dict()
        return self

    def transform(self, X):
        check_is_fitted(self, ['cols_', 'global_mean_', 'freq_maps_', 'te_maps_'])
        X = pd.DataFrame(X).copy()
        out = []
        for c in self.cols_:
            s = X[c].astype('string').fillna('UNK')
            out.append(s.map(self.freq_maps_[c]).fillna(0.0).astype(float).to_numpy().reshape(-1, 1))
            out.append(s.map(self.te_maps_[c]).fillna(self.global_mean_).astype(float).to_numpy().reshape(-1, 1))
        return np.hstack(out) if out else np.empty((len(X), 0))

target_col = 'TARGET_FRAUD'
drop_cols = leakage_cols
X = df_adv[[c for c in df_adv.columns if c not in drop_cols]].copy()
y = df_adv[target_col].copy()

for c in [d for d in datetime_cols if d in X.columns]:
    dt = pd.to_datetime(X[c], errors='coerce')
    X[f"{c}_year"] = dt.dt.year
    X[f"{c}_month"] = dt.dt.month
    X[f"{c}_day"] = dt.dt.day
    X = X.drop(columns=[c])

# Normalize missing tokens for sklearn compatibility (prevents pandas <NA> ambiguity in imputers)
X = X.replace({pd.NA: np.nan, '': np.nan, 'NA': np.nan, 'N/A': np.nan})
for c in X.columns:
    if str(X[c].dtype) == 'boolean':
        X[c] = X[c].astype(float)
    elif pd.api.types.is_string_dtype(X[c]):
        X[c] = X[c].astype('object')
    elif pd.api.types.is_object_dtype(X[c]):
        X[c] = X[c].replace({pd.NA: np.nan})

cat_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]
low_card_cols = [c for c in cat_cols if X[c].nunique(dropna=True) <= 12]
high_card_cols = [c for c in cat_cols if X[c].nunique(dropna=True) > 12]

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', RobustScaler())]), num_cols),
    ('low_cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), low_card_cols),
    ('high_cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('highenc', HighCardEncoder(smoothing=20.0))]), high_card_cols),
], remainder='drop')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Train/Test shapes:", X_train.shape, X_test.shape)""")

add_code("""# ---------------------------
# 14.4 Sampler Benchmark (applied only on training folds)
# ---------------------------
samplers = {
    'SMOTE': SMOTE(random_state=42),
    'SMOTETomek': SMOTETomek(random_state=42),
    'SMOTEENN': SMOTEENN(random_state=42),
    'ADASYN': ADASYN(random_state=42),
    'RandomOverSampler': RandomOverSampler(random_state=42),
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
baseline = LogisticRegression(max_iter=1200, class_weight='balanced')
sampler_results = []

for s_name, s_obj in samplers.items():
    fold_scores = {'recall': [], 'f1': [], 'roc_auc': [], 'pr_auc': []}
    ok = True
    for tr_idx, va_idx in cv.split(X_train, y_train):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
        pipe = ImbPipeline([('prep', clone(preprocessor)), ('sampler', clone(s_obj)), ('model', clone(baseline))])
        try:
            pipe.fit(X_tr, y_tr)
            p = pipe.predict(X_va)
            pp = pipe.predict_proba(X_va)[:, 1]
            fold_scores['recall'].append(recall_score(y_va, p))
            fold_scores['f1'].append(f1_score(y_va, p))
            fold_scores['roc_auc'].append(roc_auc_score(y_va, pp))
            fold_scores['pr_auc'].append(average_precision_score(y_va, pp))
        except Exception:
            ok = False
            break
    if ok and fold_scores['f1']:
        sampler_results.append({
            'sampler': s_name,
            'recall_mean': float(np.mean(fold_scores['recall'])),
            'f1_mean': float(np.mean(fold_scores['f1'])),
            'roc_auc_mean': float(np.mean(fold_scores['roc_auc'])),
            'pr_auc_mean': float(np.mean(fold_scores['pr_auc'])),
        })

if sampler_results:
    sampler_df = pd.DataFrame(sampler_results).sort_values(['f1_mean', 'recall_mean', 'pr_auc_mean'], ascending=False)
else:
    sampler_df = pd.DataFrame(columns=['sampler', 'recall_mean', 'f1_mean', 'roc_auc_mean', 'pr_auc_mean'])
display(sampler_df)
best_sampler_name = 'SMOTE' if sampler_df.empty else sampler_df.iloc[0]['sampler']
best_sampler = samplers[best_sampler_name]
print(f"Best sampler selected: {best_sampler_name}")""")

add_code("""# Visual comparison of sampler performance
if not sampler_df.empty:
    plt.figure(figsize=(10, 4))
    plot_df = sampler_df.melt(id_vars='sampler', value_vars=['recall_mean', 'f1_mean', 'pr_auc_mean'])
    sns.barplot(data=plot_df, x='sampler', y='value', hue='variable')
    plt.xticks(rotation=30, ha='right')
    plt.title('Sampler Benchmark (CV Mean Metrics)')
    plt.show()
print("Purpose: select imbalance method with best minority retrieval quality. Business impact: higher fraud recall at stable precision reduces missed suspicious claims.")""")

add_code("""# ---------------------------
# 14.5 Model Candidates + Tuning
# ---------------------------
models = {
    'LogisticRegression': (LogisticRegression(max_iter=1500, class_weight='balanced'), {'model__C': np.logspace(-2, 1, 8)}),
    'DecisionTree': (DecisionTreeClassifier(class_weight='balanced', random_state=42), {'model__max_depth': [3, 5, 8, 12, None], 'model__min_samples_leaf': [1, 2, 5, 10]}),
    'RandomForest': (RandomForestClassifier(class_weight='balanced_subsample', random_state=42, n_jobs=-1), {'model__n_estimators': [150, 300, 500], 'model__max_depth': [8, 14, None], 'model__min_samples_leaf': [1, 2, 5]}),
}

if HAS_XGB:
    models['XGBoost'] = (
        XGBClassifier(random_state=42, n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.9, objective='binary:logistic', eval_metric='aucpr', n_jobs=2),
        {'model__max_depth': [4, 6, 8], 'model__learning_rate': [0.03, 0.05, 0.08], 'model__subsample': [0.8, 0.9, 1.0]}
    )
if HAS_LGBM:
    models['LightGBM'] = (
        LGBMClassifier(random_state=42, n_estimators=500, learning_rate=0.05, objective='binary', class_weight='balanced', n_jobs=2),
        {'model__num_leaves': [31, 63, 127], 'model__learning_rate': [0.03, 0.05, 0.08], 'model__min_child_samples': [20, 40, 80]}
    )
if HAS_CAT:
    models['CatBoost'] = (
        CatBoostClassifier(random_state=42, verbose=False, loss_function='Logloss', eval_metric='AUC', iterations=600, learning_rate=0.05),
        {'model__depth': [4, 6, 8], 'model__l2_leaf_reg': [1, 3, 5, 7]}
    )

model_rows = []
failed_models = []
best_bundle = None
best_score = -1.0
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
print("Models to benchmark:", list(models.keys()))""")

add_code("""# Train + tune all candidate models with the selected sampler
for m_name, (m_obj, param_dist) in models.items():
    t0 = time.perf_counter()
    pipe = ImbPipeline([('prep', clone(preprocessor)), ('sampler', clone(best_sampler)), ('model', clone(m_obj))])
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=min(8, max(3, len(param_dist) * 2)),
        scoring='f1',
        cv=inner_cv,
        random_state=42,
        n_jobs=1,
        refit=True,
    )
    try:
        search.fit(X_train, y_train)
    except Exception as e:
        failed_models.append({'model': m_name, 'error': str(e).split('\\n')[0][:220]})
        print(f"Skipped {m_name} due to fit error.")
        continue
    fit_seconds = time.perf_counter() - t0

    best_pipe = search.best_estimator_
    prob = best_pipe.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, prob)
    f1_values = (2 * precision * recall) / (precision + recall + 1e-9)
    idx = int(np.argmax(f1_values))
    thr = float(thresholds[idx]) if idx < len(thresholds) else 0.5
    pred = (prob >= thr).astype(int)

    row = {
        'model': m_name,
        'best_cv_f1': float(search.best_score_),
        'test_recall': float(recall_score(y_test, pred)),
        'test_precision': float(precision_score(y_test, pred, zero_division=0)),
        'test_f1': float(f1_score(y_test, pred)),
        'test_roc_auc': float(roc_auc_score(y_test, prob)),
        'test_pr_auc': float(average_precision_score(y_test, prob)),
        'best_threshold': thr,
        'fit_seconds': float(fit_seconds),
    }
    model_rows.append(row)

    rank_score = row['test_f1'] + 0.5 * row['test_recall']
    if rank_score > best_score:
        best_score = rank_score
        best_bundle = {'name': m_name, 'search': search, 'pipe': best_pipe, 'prob': prob, 'pred': pred, 'threshold': thr}

if not model_rows:
    raise RuntimeError("All model candidates failed. Please inspect preprocessing/dtypes and retry.")

model_df = pd.DataFrame(model_rows).sort_values(['test_f1', 'test_recall', 'test_pr_auc'], ascending=False)
display(model_df)
if failed_models:
    print("Some models were skipped due to runtime/compatibility issues:")
    display(pd.DataFrame(failed_models))
print(f"Best model selected (recall+f1 objective): {best_bundle['name']}")""")

add_code("""# ---------------------------
# 14.6 Final Evaluation
# ---------------------------
best_name = best_bundle['name']
best_pipe = best_bundle['pipe']
best_prob = best_bundle['prob']
best_pred = best_bundle['pred']
best_thr = best_bundle['threshold']

print("Classification Report (Threshold-tuned):")
print(classification_report(y_test, best_pred, digits=4))

cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

fpr, tpr, _ = roc_curve(y_test, best_prob)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_test, best_prob):.4f}")
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.title('ROC Curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.show()

prec, rec, _ = precision_recall_curve(y_test, best_prob)
plt.figure(figsize=(6, 4))
plt.plot(rec, prec, label=f"PR-AUC={average_precision_score(y_test, best_prob):.4f}")
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()

print("Business tradeoff:")
print("- False negatives: missed fraud -> direct payout leakage risk.")
print("- False positives: extra manual review cost and customer friction.")
print("- Threshold tuning lets ops choose review-load vs fraud-catch balance.")""")

add_code("""# ---------------------------
# 14.7 Explainability
# ---------------------------
perm = permutation_importance(best_pipe, X_test, y_test, scoring='f1', n_repeats=5, random_state=42)
# Permutation importance on a pipeline maps to original input columns
perm_feat_names = np.array(X_test.columns.astype(str))
imp_df = pd.DataFrame({'feature': perm_feat_names, 'importance': perm.importances_mean}).sort_values('importance', ascending=False).head(20)

plt.figure(figsize=(10, 6))
sns.barplot(data=imp_df, x='importance', y='feature')
plt.title('Top 20 Permutation Importances')
plt.show()

try:
    try:
        feat_names = best_pipe.named_steps['prep'].get_feature_names_out()
    except Exception:
        X_tmp = best_pipe.named_steps['prep'].transform(X_test.head(5))
        feat_names = np.array([f'feature_{i}' for i in range(X_tmp.shape[1])])
    X_test_tx = best_pipe.named_steps['prep'].transform(X_test)
    model_obj = best_pipe.named_steps['model']
    sample_n = min(250, X_test_tx.shape[0])
    idx = np.random.RandomState(42).choice(X_test_tx.shape[0], sample_n, replace=False)
    X_shap = X_test_tx[idx]
    expl = shap.Explainer(model_obj, X_shap)
    shap_vals = expl(X_shap)
    shap.summary_plot(shap_vals, X_shap, feature_names=feat_names, max_display=12)
except Exception as e:
    print("SHAP skipped due to runtime/model compatibility issue:", e)""")

add_code("""# ---------------------------
# 14.8 Save Artifacts
# ---------------------------
models_dir = Path('models')
reports_dir = Path('reports')
models_dir.mkdir(parents=True, exist_ok=True)
reports_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(best_pipe, models_dir / 'fraud_model_advanced.pkl')

advanced_metrics = {
    'best_model': best_name,
    'best_sampler': best_sampler_name,
    'threshold': float(best_thr),
    'test_recall': float(recall_score(y_test, best_pred)),
    'test_precision': float(precision_score(y_test, best_pred, zero_division=0)),
    'test_f1': float(f1_score(y_test, best_pred)),
    'test_roc_auc': float(roc_auc_score(y_test, best_prob)),
    'test_pr_auc': float(average_precision_score(y_test, best_prob)),
    'sampler_benchmark': sampler_df.to_dict(orient='records'),
    'model_benchmark': model_df.to_dict(orient='records'),
    'feature_importance_top20': imp_df.to_dict(orient='records'),
}
with (reports_dir / 'advanced_workflow_metrics.json').open('w', encoding='utf-8') as f:
    json.dump(advanced_metrics, f, indent=2)

print("Final outputs saved:")
print("- models/fraud_model_advanced.pkl")
print("- reports/advanced_workflow_metrics.json")""")

nb['cells'] = cells

output_path = Path(__file__).resolve().parent / "Full_Flow_Notebook.ipynb"
fallback_path = Path(__file__).resolve().parent / "Full_Flow_Notebook_updated.ipynb"
fallback_path_2 = Path.cwd() / "Full_Flow_Notebook_updated.ipynb"

try:
    with output_path.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook written: {output_path}")
except PermissionError:
    try:
        with fallback_path.open("w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print(f"Primary notebook is locked. Updated copy written: {fallback_path}")
    except PermissionError:
        with fallback_path_2.open("w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print(f"Notebook folder is locked. Updated copy written: {fallback_path_2}")
