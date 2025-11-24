import pandas as pd
from rice_ml.processing.multicollinearity import compute_vif, reduce_multicollinearity

def test_compute_vif_returns_dataframe():
    df = pd.DataFrame({
        "x1": [1,2,3,4,5],
        # Nearly collinear with x1 but with small noise to avoid divide-by-zero warnings
        "x2": [2.0,4.1,6.1,8.2,10.3]
    })

    vif_df = compute_vif(df)
    assert "feature" in vif_df.columns
    assert "VIF" in vif_df.columns
    assert len(vif_df) == 2

def test_reduce_multicollinearity_removes_high_vif():
    df = pd.DataFrame({
        "x1": [1,2,3,4,5],
        # Keep x2 strongly correlated with x1 while avoiding perfect collinearity
        "x2": [2.0,4.1,6.1,8.2,10.3],
        "x3": [5,3,6,2,7]    # independent
    })

    reduced = reduce_multicollinearity(df, threshold=5.0)

    # At least one of x1 or x2 should be removed
    assert not ({"x1", "x2"}.issubset(reduced.columns))
