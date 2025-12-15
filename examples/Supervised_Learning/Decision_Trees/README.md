# Decision Trees

Decision Trees are supervised learning models that can be used for both classification and regression.

They repeatedly split the data based on feature values. Each internal node represents a decision rule on a single feature, each branch represents the outcome of that rule, and each leaf node stores a predicted output. The tree structure makes the model easy to visualize and interpret.

This notebook uses a Decision Tree Classifier to predict whether a person’s income is greater than $50K or less than or equal to $50K based on demographic and employment features from the Adult Census Income dataset.

Decision Trees are a good fit for this task because they:
- Handle both numerical and categorical variables after simple encoding
- Capture nonlinear relationships and feature interactions automatically
- Do not require feature scaling
- Provide clear interpretability through tree structure and feature importance scores

In this project, I focus on a depth-limited tree that balances interpretability and generalization rather than a fully grown tree, which would likely overfit. Given the skewed nature of the dataset, I expect the model to perform better on the lower-income class and to assign most importance to a small subset of highly informative features.

---

## Dataset Comments

The original dataset contains the column `fnlwgt`. For this project, `fnlwgt` is kept in the raw data but dropped from the model features because it behaves like a sampling weight and does not provide meaningful predictive signal in a simple Decision Tree.

The dataset is imbalanced, with most individuals belonging to the lower-income class. To ensure reliable evaluation, a stratified train–test split is used so that class proportions remain consistent across training and testing sets.

---

## Model Configuration and Evaluation

The Decision Tree Classifier is trained using the Gini impurity criterion. To reduce overfitting, model complexity is controlled using the following hyperparameters:

- `max_depth = 8`
- `min_samples_split = 20`
- `min_samples_leaf = 10`

Model performance is evaluated using:
- Test-set accuracy
- Confusion matrix
- Precision, recall, and F1-score for each class
- Macro and weighted averages to account for class imbalance

In this run, the model achieved a test accuracy of approximately **0.86**. Recall for the lower-income class is substantially higher than for the higher-income class, reflecting the underlying class imbalance and the model’s conservative behavior when predicting high income.

---

## Feature Importance

Decision Trees compute feature importance based on the total reduction in Gini impurity contributed by each feature across all splits where it appears.

The most important predictors in this model are:
- `marital_status_Married_civ_spouse`
- `education_num`
- `capital_gain`
- `capital_loss`
- `age`
- `hours_per_week`

These features align well with socioeconomic expectations. Income is strongly related to marital status, education level, and capital gains, while age and hours worked naturally correlate with earning potential. The Decision Tree effectively uses these variables to separate higher-income individuals from lower-income ones.

The model performs especially well for the lower-income class, which dominates the dataset, and provides a clear ranking of influential features. Predicting high income remains more difficult due to the presence of social, economic, and unobserved factors not captured in the dataset.

---

## Outlook

Random Forest is the next logical choice because it reduces the overfitting that single decision trees are prone to by averaging predictions across many independently trained trees. Each tree sees different subsets of data and features, allowing the ensemble to capture minority-class patterns and stabilize noisy splits. This leads to higher overall generalization performance, especially on imbalanced datasets like the census income data. Feature importance estimates also become more reliable because they are aggregated across many models rather than driven by a few dominant splits. As a result Random Forest typically achieves better recall, higher F1 scores and more robust predictions compared to a standalone Decision Tree. At least thats the thought , running random forests on the dataset should give us insights into this :)