import numpy as np
import pandas as pd
from rice_ml.processing.KNN_imputation import KNNImputer, knn_impute

def test_knn_imputer_basic_imputation():
    X = np.array([
        [1.0, 2.0],
        [2.0, np.nan],
        [1.0, 2.0]
    ])

    imputer = KNNImputer(n_neighbors=2)
    out = imputer.fit_transform(X)

    assert out[1, 1] == 2.0

def test_knn_impute_dataframe():
    df = pd.DataFrame({
        "a": [1, 2, np.nan],
        "b": [5.0, np.nan, 5.0]
    })

    out = knn_impute(df)

    assert out.isna().sum().sum() == 0
