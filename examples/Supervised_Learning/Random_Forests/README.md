# Random Forests

Random Forest is an ensemble learning method that extends decision trees by training many trees on different subsets of the data and combining their predictions. Each tree is built using bootstrap sampling, and at every split only a random subset of features is considered. This randomness reduces variance and mitigates overfitting compared to a single decision tree.

This notebook uses a Random Forest Classifier to predict whether a person’s income is greater than $50K or less than or equal to $50K using the Adult Census Income dataset. Random Forests are a natural next step after Decision Trees because they preserve interpretability while significantly improving generalization.

Random Forests are well suited for this dataset because they:
- Reduce overfitting by averaging predictions across many trees
- Capture nonlinear interactions and complex feature relationships
- Remain stable in the presence of noisy or correlated variables
- Provide reliable, aggregated feature importance estimates

The primary goal is to evaluate whether Random Forest improves recall for the high-income class and stabilizes performance under class imbalance.

---

## Dataset Comments

The same preprocessing steps apply here as in the Decision Tree section. The dataset is imbalanced, with a substantially larger proportion of individuals earning ≤ $50K. A stratified train–test split is therefore used to preserve class proportions.

The `fnlwgt` column is dropped prior to training, as it does not provide meaningful predictive signal. After one-hot encoding categorical variables, the final feature matrix contains 99 features, which are used directly by the Random Forest model.

---

## Model Configuration and Training

The Random Forest model is trained using the following hyperparameters:

- `n_estimators = 300`
- `max_depth = 15`
- `min_samples_split = 20`
- `min_samples_leaf = 10`
- `n_jobs = -1` (full parallelism)

Constraining tree depth and minimum leaf size helps control variance and prevents individual trees from overfitting, while the large number of estimators ensures sufficient ensemble diversity.

---

## Evaluation and Results

Model performance is evaluated using:
- Test-set accuracy
- Precision, recall, and F1-score for each income class
- Macro and weighted averages
- Confusion matrix

The Random Forest achieves a test accuracy of approximately **0.86**, improving slightly over the single Decision Tree. Recall for the high-income class increases relative to the pruned tree, although predicting high income remains more challenging due to class imbalance.

Overall, the Random Forest demonstrates more stable predictions and reduced sensitivity to individual splits, reflecting effective variance reduction.

---

## Feature Importance

Feature importance is computed as the average reduction in impurity across all trees in the forest, making it more robust than importance derived from a single tree.

The most influential features include:
- `marital_status_Married-civ-spouse`
- `capital_gain`
- `education_num`
- `age`
- `marital_status_Never-married`
- `hours_per_week`

These features align with socioeconomic expectations and reinforce patterns observed in the Decision Tree analysis. Compared to a single tree, importance is distributed more broadly, reflecting interactions among multiple predictors rather than dominance by a few splits.

The confusion matrix confirms strong performance for the lower-income class and moderate performance for the higher-income class, consistent with the underlying class imbalance.

---

## Outlook

Random Forest provides a meaningful improvement over Decision Trees by addressing overfitting and split dominance while maintaining interpretability. It serves as a strong baseline for tabular classification tasks.

Future extensions could include tuning `max_features`, applying class weights, or exploring boosting-based methods such as Gradient Boosting or XGBoost to further improve minority-class recall.