# Logistic Regression (from scratch)

This folder contains:
- `logistic_regression.ipynb`: end-to-end notebook using a dataset from `datasets/`
- `logistic_regression.py`: reusable class implementing binary logistic regression with gradient descent

## How to run
1. Place your dataset (CSV) under the top-level `datasets/` folder of the repo.
2. Open the notebook and set `DATA_PATH` to your CSV (e.g., `datasets/census_dataset.csv`).
3. Run all cells. The notebook will:
   - Clean & encode features
   - Train/test split
   - Train a from-scratch logistic regression model
   - Compare with `sklearn.linear_model.LogisticRegression`
   - Report accuracy, precision, recall, F1, and ROC AUC
   - Plot a ROC curve
