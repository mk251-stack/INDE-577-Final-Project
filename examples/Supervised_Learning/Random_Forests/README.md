# Random Forests

Random Forest is an ensemble learning method that extends decision trees by training many trees on different subsets of the data and combining their predictions. Each tree is built using bootstrap sampling, and at every split only a random subset of features is considered. This randomness reduces variance and mitigates overfitting compared to a single decision tree.

This notebook uses a Random Forest Classifier to predict whether a person’s income is greater than $50K or less than or equal to $50K using the Adult Census Income dataset. Random Forests are a natural next step after Decision Trees because they preserve interpretability while significantly improving generalization.

*The Objective*

The objective here is to evaluate whether Random Forest improves stability and recall for the higher income class in a highly imbalanced tabular dataset.

Random Forests are well suited for this dataset because they:
- Reduce overfitting by averaging predictions across many trees
- Capture nonlinear interactions and complex feature relationships
- Remain stable in the presence of noisy or correlated variables
- Provide reliable, aggregated feature importance estimates

# Data Set Comments:
The model is trained on the Adult Census Income dataset. The prediction task is binary classification:

>Income less than or equal to fifty thousand dollars
>Income greater than fifty thousand dollars

The same preprocessing steps apply here as in Decision Trees. The dataset is imbalanced, with far more individuals earning less than fifty thousand dollars. Therefore the stratified train test split remains essential to maintain consistent class proportions. 
The fnlwgt column is again dropped before training because it does not contribute meaningful signal to the model.

# Key Observations and Interpretation:

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

*Key Insights:*

>Random Forest improves prediction stability and reduces sensitivity to noise compared to individual trees.
>Precision for the higher income class increases, showing better discrimination of minority outcomes.
>Recall for high income remains challenging due to class imbalance and population diversity.
>Feature importance rankings are more reliable and less driven by individual split artifacts.

# Feature Importance Interpretation:

Model performance is evaluated using:
- Test-set accuracy
- Precision, recall, and F1-score for each income class
- Macro and weighted averages
- Confusion matrix

The Random Forest achieves a test accuracy of approximately **0.86**, improving slightly over the single Decision Tree. Recall for the high-income class increases relative to the pruned tree, although predicting high income remains more challenging due to class imbalance.

Overall, the Random Forest demonstrates more stable predictions and reduced sensitivity to individual splits, reflecting effective variance reduction.

---

## Feature Importance

# Conclusion and Outlook:

Random Forest provides a clear improvement over a standalone decision tree by reducing variance, stabilizing predictions, and producing more robust feature importance estimates. While the model still struggles to fully capture the diverse patterns that define high income individuals, it generalizes better and behaves more consistently across runs. The importance estimates are also more reliable since they reflect aggregated behavior across many models rather than the structure of a single tree.

Random Forest forms a strong baseline for tabular classification. Future extensions may include tuning hyperparameters such as max_depth, max_features and class weights or exploring Gradient Boosting methods, which could push performance further especially for the high income group.

In any case, Random Forests extracted the most usable signal from this dataset and proved to be a successful implementation. If nothing else, throwing more trees at the problem actually worked this time. ^^
