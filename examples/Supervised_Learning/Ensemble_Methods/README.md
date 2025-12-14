# Ensemble Methods

This example demonstrates how different ensemble classifiers compare on the UCI Credit Card Default dataset. Baseline models (logistic regression, decision tree, KNN) are evaluated alongside ensembles (random forest for bagging, AdaBoost for boosting, bagging with decision trees, histogram gradient boosting, and a soft-voting ensemble that blends logistic regression, random forest, and histogram gradient boosting).

## Algorithm
Ensemble methods combine multiple learners to reduce variance and bias compared to a single model. In this notebook:

- **Bagging:** Random Forest and Bagging(DT) average many tree learners trained on bootstrapped samples, improving stability and generalization.
- **Boosting:** AdaBoost and Histogram Gradient Boosting focus successive learners on prior errors to lift performance on hard cases.
- **Voting:** A soft-voting classifier averages calibrated probabilities from heterogeneous models (logistic regression, random forest, histogram gradient boosting) to capture complementary decision boundaries.

Evaluation centers on **ROC–AUC** as the primary, threshold-free ranking metric. **F1-score** and threshold tuning are used to explore the precision/recall tradeoff for the minority default class.


## Data
- **Source:** UCI Credit Card Default (30,000 clients).
- **Features:**
  - Demographics: sex, education, marriage, age
  - Payment history: `PAY_0, PAY_2, …, PAY_6`
  - Billing amounts: `BILL_AMT1`–`BILL_AMT6`
  - Payment amounts: `PAY_AMT1`–`PAY_AMT6`
- **Target:** `default.payment.next.month` (1 if the client defaults on the next payment).
- **Class balance:** Defaults are the minority (~22%), so stratified splits and recall-oriented metrics are emphasized.

## Preprocessing

The notebook builds a preprocessing pipeline that:

1. Splits numeric and categorical-like columns.
2. Imputes missing numeric values with the **median**.
3. Standardizes numeric features to zero mean and unit variance.
4. One-hot encodes categorical variables.
5. Returns the transformed feature matrix `X`, target vector `y`, and metadata for feature names.

## Training and evaluation

- **Split:** Stratified train/test split (75% / 25%) performed before cross-validation to prevent leakage.
- **Cross-validation:** Stratified 5-fold CV on the training set evaluating Accuracy and ROC–AUC.
- **Top CV performers (by ROC–AUC):**
  1) Voting(LR + RF + HGB) ≈ 0.786
  2) Histogram Gradient Boosting ≈ 0.784
  3) Random Forest ≈ 0.773
  4) Logistic Regression / AdaBoost / Bagging(DT) ≈ 0.763–0.770
  5) KNN / Decision Tree (significantly lower)

After CV, models are refit on the full training set and evaluated on the held-out test set.

## Key results (Voting ensemble)

- **Test ROC–AUC:** ≈ 0.775
- **Test accuracy:** ≈ 0.819
- **Confusion matrix @ 0.5 threshold:** TN=5,540; FP=301; FN=1,053; TP=606
- **Threshold tuning:** Lowering the decision threshold (≈0.01 in this run) greatly boosts recall for defaulters at the expense of more false alarms. The optimal cutoff should reflect business costs of false positives vs false negatives.

## How to run

Open and execute `examples/Supervised_Learning/Ensemble_Methods/Ensemple_Methods.ipynb`. The notebook uses repository utilities for preprocessing (`src/data/preprocessing.py`), ensemble model definitions (`src/rice_ml/supervised_learning/ensemble_methods.py`), and evaluation helpers (`src/data/postprocessing.py`).
