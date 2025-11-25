import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

__all__ = ["compute_vif", "reduce_multicollinearity"]

def compute_vif(df_X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor (VIF) for each feature in the DataFrame.

    Parameters
    ----------
    df_X : pd.DataFrame
        DataFrame containing only predictor variables (numeric).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ["feature", "VIF"], sorted by descending VIF.
    """
    vif_df = pd.DataFrame()
    vif_df["feature"] = df_X.columns
    vif_df["VIF"] = [variance_inflation_factor(df_X.values, i) for i in range(df_X.shape[1])]
    return vif_df.sort_values("VIF", ascending=False).reset_index(drop=True)

def reduce_multicollinearity(df_X: pd.DataFrame, threshold: float = 10.0) -> pd.DataFrame:
    """
    Iteratively remove features with VIF above a threshold until all are below threshold.

    Parameters
    ----------
    df_X : pd.DataFrame
        DataFrame containing only predictor variables (numeric).
    threshold : float, optional
        VIF threshold for multicollinearity, by default 10.0

    Returns
    -------
    pd.DataFrame
        DataFrame with reduced set of predictors.
    """
    X_iter = df_X.copy()
    iteration = 1

    while True:
        vif_table = compute_vif(X_iter)
        max_vif = vif_table.loc[0, "VIF"]
        worst_feature = vif_table.loc[0, "feature"]

        if max_vif <= threshold:
            break

        X_iter = X_iter.drop(columns=[worst_feature])
        iteration += 1

    return X_iter
