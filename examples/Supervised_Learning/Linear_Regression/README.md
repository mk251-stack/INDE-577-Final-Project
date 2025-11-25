# Linear Regression

This directory contains example code and notes for the Linear Regression algorithm
in supervised learning. The example focuses on **housing price prediction** using a
complete machine learning pipeline and several custom-built Python modules.

---

## Algorithm

Linear Regression models the relationship between a continuous target variable and
a set of input features by fitting a linear function that minimizes the sum of
squared residuals.

In this project, the goal is to fit a model that predicts **median housing price**
using tabular real-estate features. The implementation includes:

- Ordinary Least Squares (OLS)
- Closed-form Normal Equation solver
- Comparison with `sklearn.linear_model.LinearRegression`
- Residual diagnostics (normality, heteroskedasticity, outliers)
- Cross-validation with MSE and R²
- Feature selection via Variance Inflation Factor (VIF)

---

## ML Pipeline Overview

This example implements a full end-to-end machine learning workflow:

1. **Load dataset**  
   Using the provided `BostonHousing.csv`.

2. **Handle missing values**  
   Custom KNN-based imputation using `knn_impute.py`.

3. **Reduce multicollinearity**  
   Variance Inflation Factor (VIF) analysis via `vif.py`.

4. **Train/test split**

5. **Feature scaling**  
   Standardization with `StandardScaler` from `scaling.py`.

6. **Model fitting**  
   - Custom OLS (`linear_regression.py`)  
   - Scikit-learn LinearRegression for comparison

7. **Model evaluation**
   - Residual plots  
   - Q–Q plots  
   - Shapiro–Wilk test  
   - MSE and R²  
   - 5-fold cross-validation  

8. **Interpretability**
   - Coefficient analysis  
   - Ranked feature importance  

---

## Data

**File:** `datasets/BostonHousing.csv`

This dataset contains real-estate features such as:

- crime rate  
- number of rooms  
- nitric oxide concentration  
- pupil–teacher ratio  
- property-tax rate  
- proportion of lower-status population  
- and more  

**Target variable:**  
- `MEDV` — Median value of owner-occupied homes.

Preprocessing includes missing-value imputation, type coercion,
VIF-based feature removal, and outlier diagnostics.

---

## Custom Modules

This example uses several custom modules located under:

```
src/rice_ml/processing
src/rice_ml/supervised_learning
```

### `KNN_imputation.py`
NumPy-only KNN imputation module that computes distances using shared
non-missing features.  
Handles missing values by finding k-nearest neighbors based on overlapping attributes.

### `multicoliniarity.py`
Computes Variance Inflation Factors (VIF) and iteratively removes
highly collinear features to stabilize the regression model.

### `scaling.py`
Implements `StandardScaler` for mean–variance feature scaling.

### `linear_regression.py`
Full OLS implementation with:
- Normal Equation solver  
- Prediction  
- Residual extraction  
- R² and MSE metrics  

---
