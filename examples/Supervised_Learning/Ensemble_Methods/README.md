# Ensemble Methods

This example demonstrates how different ensemble classifiers compare on the **UCI Credit Card Default** dataset. Baseline models (Logistic Regression, Decision Tree, KNN) are evaluated alongside ensemble approaches, including
Random Forest (bagging), AdaBoost (boosting), Bagging with Decision Trees, Histogram Gradient Boosting, and a soft-voting ensemble that blends Logistic Regression, Random Forest, and Histogram Gradient Boosting.

---

## Notebook quick reference
- **Dataset:** UCI Credit Card Default (`datasets/UCI_Credit_Card.csv`) with binary default indicator
- **Expected runtime:** ~8–10 minutes on a modern laptop for all ensemble variants
- **Key parameters to tweak:** `n_estimators`, learning rate (for boosting), tree depth constraints, and voting weights
- **Demonstrates:** side-by-side comparison of bagging, boosting, and voting ensembles on an imbalanced classification problem

## Algorithm

Ensemble methods combine multiple learners to reduce variance and bias compared to a single model. In this notebook:

- **Bagging:** Random Forest and Bagging(DT) average many tree learners trained on bootstrapped samples, improving stability and generalization.
- **Boosting:** AdaBoost and Histogram Gradient Boosting focus successive learners on prior errors to improve performance on hard-to-classify cases.
- **Voting:** A soft-voting classifier averages calibrated probabilities from heterogeneous models (Logistic Regression, Random Forest, Histogram Gradient Boosting) to capture complementary decision boundaries.

Evaluation centers on **ROC–AUC** as the primary, threshold-free ranking metric.
**F1-score** and threshold tuning are used to explore the precision/recall trade-off for the minority default class.

---

## Data

- **Source:** UCI Credit Card Default dataset (30,000 clients).
- **Features:**
  - Demographics: sex, education, marriage, age
  - Payment history: `PAY_0, PAY_2, …, PAY_6`
  - Billing amounts: `BILL_AMT1`–`BILL_AMT6`
  - Payment amounts: `PAY_AMT1`–`PAY_AMT6`
- **Target:** `default.payment.next.month`
  - `1` = client defaults on the next payment
  - `0` = client does not default
- **Class balance:** Defaults are the minority (~22%), motivating stratified splits and recall-oriented evaluation metrics.

---

## Preprocessing

The notebook builds a preprocessing pipeline that:

1. Splits numeric and categorical-like columns.
2. Imputes missing numeric values with the **median**.
3. Standardizes numeric features to zero mean and unit variance.
4. One-hot encodes categorical variables.
5. Returns the transformed feature matrix `X`, target vector `y`, and metadata for feature name tracking.

---

## Training and evaluation

- **Split:** Stratified train/test split (75% / 25%) performed before cross-validation to prevent leakage.
- **Cross-validation:** Stratified 5-fold CV on the training set evaluating Accuracy and ROC–AUC.
- **Top CV performers (by ROC–AUC):**
  1) **Voting(LR + RF + HGB)** ≈ 0.786  
  2) **Histogram Gradient Boosting** ≈ 0.784  
  3) **Random Forest** ≈ 0.773  
  4) **Logistic Regression / AdaBoost / Bagging(DT)** ≈ 0.763–0.770  
  5) **KNN / Decision Tree** (significantly lower)

After cross-validation, models are refit on the full training set and evaluated on the held-out test set.

---

## Key results (Voting ensemble)

- **Test ROC–AUC:** ≈ 0.775  
- **Test accuracy:** ≈ 0.819  
- **Confusion matrix @ 0.5 threshold:**  
  TN = 5,540 · FP = 301 · FN = 1,053 · TP = 606
- **Threshold tuning:** Lowering the decision threshold (≈ 0.01 in this run) substantially increases recall for defaulters at the cost of more false positives. The optimal cutoff should reflect business costs of false alarms versus missed defaults.

---

## Key insights

- Ensemble models **consistently outperform single learners** on this imbalanced credit-risk task.
- The **soft-voting ensemble** benefits from combining linear structure (Logistic Regression) with nonlinear interactions (Random Forest and Gradient Boosting), yielding stronger probability ranking.
- Histogram Gradient Boosting performs nearly as well as the voting ensemble, confirming its effectiveness on large tabular datasets.
- Single models such as Decision Trees and KNN exhibit higher variance or sensitivity to imbalance, leading to weaker ROC–AUC performance.

---

## Conclusion

This example highlights the practical advantages of ensemble methods for real-world, imbalanced classification problems such as credit default prediction. By leveraging complementary model strengths, ensemble approaches achieve more robust generalization and superior ranking performance than individual classifiers.

While ROC–AUC confirms strong separability, final deployment decisions should consider **threshold tuning** based on business costs. In credit risk settings, the trade-off between false positives and false negatives is often more important than raw accuracy.
