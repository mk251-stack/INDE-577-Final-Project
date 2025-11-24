"""
Reusable preprocessing utilities for tabular datasets.

Contains:
- load_data(path)
- preprocess_credit_dataset(df)
"""

import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# -------------------------------------------------------------------
# Robust load_data with automatic resolution to project root
# -------------------------------------------------------------------
def load_data(path):
    path = Path(path)

    # (1) Try direct path
    if path.exists():
        return pd.read_csv(path)

    # (2) Try relative to project root (three levels up from src/data)
    project_root = Path(__file__).resolve().parents[3]
    alt_path = project_root / path

    if alt_path.exists():
        return pd.read_csv(alt_path)

    raise FileNotFoundError(
        f"Dataset not found.\n"
        f"Tried: {path}\n"
        f"Then: {alt_path}"
    )


# -------------------------------------------------------------------
# Preprocessing for UCI Credit Card dataset
# -------------------------------------------------------------------
def preprocess_credit_dataset(df):
    """
    Cleans and preprocesses the UCI Credit Card dataset.

    Returns:
        X (DataFrame): transformed features
        y (Series): binary target variable
        meta (dict): metadata (column names, transformers, etc.)
    """

    # Target column
    if "default.payment.next.month" not in df.columns:
        raise KeyError("Target column 'default.payment.next.month' not found.")

    y = df["default.payment.next.month"]

    # Drop the target from features
    X = df.drop(columns=["default.payment.next.month"])

    # Identify categorical variables
    categorical_like = [
        "SEX", "EDUCATION", "MARRIAGE",
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"
    ]

    cat_cols = [c for c in categorical_like if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Basic numeric imputation
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())

    # Create a OneHotEncoder compatible with multiple sklearn versions
    try:
        ohe = OneHotEncoder(drop="first", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(drop="first", sparse=False)

    # Preprocessing transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", Pipeline([
                ("onehot", ohe)
            ]), cat_cols)
        ]
    )

    X_transformed = preprocessor.fit_transform(X)

    # Extract feature names manually
    num_feature_names = num_cols
    cat_feature_names = (
        preprocessor.named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(cat_cols)
        .tolist()
    )

    feature_names = num_feature_names + cat_feature_names

    # Create DataFrame
    X_final = pd.DataFrame(X_transformed, columns=feature_names)

    meta = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
    }

    return X_final, y, meta

