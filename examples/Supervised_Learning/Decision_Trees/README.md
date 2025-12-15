# Decision Trees

Decision Trees are supervised learning models that can be used for both classification and regression.

They repeatedly split the data based on feature values. Each internal node represents a decision rule on a single feature, each branch represents the outcome of that rule, and each leaf node stores a predicted output. The tree structure makes the model easy to visualize and interpret.

*The Objective:* <br>

The objective is to evaluate how a single interpretable model performs on an imbalanced, real world classification task and to understand which features most strongly influence income predictions.

I used a Decision Tree Classifier to predict whether a person’s income is greater than fifty thousand dollars or less than or equal to fifty thousand dollars based on demographic and employment features from the Adult Census Income dataset.

*Algorithm:* <br>

Decision Trees are supervised learning models that recursively partition the feature space using decision rules that maximize class separation. Each internal node represents a split on a single feature, branches represent outcomes of that rule, and leaf nodes store predicted class labels. 

Decision Trees are a good fit here because they:
>Handle both numerical and categorical variables after simple encoding.
>Capture nonlinear relationships and feature interactions automatically.
>Do not require feature scaling.
>Provide clear interpretability through the tree diagram and feature importance scores.

In this project, I focus on a depth-limited tree that balances interpretability and generalization rather than a fully grown tree, which would likely overfit. Given the skewed nature of the dataset, I expect the model to perform better on the lower-income class and to assign most importance to a small subset of highly informative features.



# DataSet Overview & Comments:

*Source* <br>
U.S. Adult Census Income dataset

*Target variable* <br>
Income level
Greater than fifty thousand dollars versus less than or equal to fifty thousand dollars

*Class balance* <br>
The dataset is imbalanced, with the lower income class forming the majority of observations. This motivates stratified splitting and careful evaluation beyond accuracy alone.

*Feature handling* <br>
The dataset includes a column named fnlwgt, which represents a sampling weight rather than an intrinsic personal attribute. While retained in the raw data, fnlwgt is excluded from the model features because it does not provide meaningful predictive signal for a simple Decision Tree classifier.
The original dataset also contains the column fnlwgt. For this project I kept fnlwgt in the raw data but dropped it from the model features because it behaves like a sampling weight and does not provide meaningful predictive signal in a simple Decision Tree.

The original dataset contains the column `fnlwgt`. For this project, `fnlwgt` is kept in the raw data but dropped from the model features because it behaves like a sampling weight and does not provide meaningful predictive signal in a simple Decision Tree.

# Training configuration: 
The Decision Tree classifier is trained using the Gini impurity criterion. To reduce overfitting and control model complexity, the following hyperparameters are applied:

>max_depth set to eight <br>
>min_samples_split set to twenty <br>
>min_samples_leaf set to ten <br>

These constraints prevent the tree from memorizing noise while preserving its ability to capture meaningful decision boundaries.

A stratified train test split is used so that class proportions remain consistent across training and testing sets, leading to a more reliable evaluation.

# Evaluation:
Because the dataset is imbalanced, accuracy alone is insufficient to fully characterize model performance. Evaluation therefore focuses on a combination of metrics that reflect both overall correctness and class specific behavior.

>Overall accuracy on the test set <br>
>Confusion matrix for the two income classes <br>
>Precision, recall and F1 score for each class <br>
>Macro and weighted averages to account for class imbalance <br>

Model performance is evaluated using:
- Test-set accuracy
- Confusion matrix
- Precision, recall, and F1-score for each class
- Macro and weighted averages to account for class imbalance

In this run, the model achieved a test accuracy of approximately **0.86**. Recall for the lower-income class is substantially higher than for the higher-income class, reflecting the underlying class imbalance and the model’s conservative behavior when predicting high income.

Decision Trees compute feature importance scores based on the reduction in Gini impurity achieved by each feature across all splits where it appears. Features that consistently produce strong splits receive higher importance values.

## Feature Importance

>marital_status_Married_civ_spouse <br>
>education_num <br>
>capital_gain <br>
>capital_loss <br>
>age <br>
>hours_per_week <br>

The most important predictors in this model are:
- `marital_status_Married_civ_spouse`
- `education_num`
- `capital_gain`
- `capital_loss`
- `age`
- `hours_per_week`

These features align well with socioeconomic expectations. Income is strongly related to marital status, education level, and capital gains, while age and hours worked naturally correlate with earning potential. The Decision Tree effectively uses these variables to separate higher-income individuals from lower-income ones.

# Conclusion & Outlook:

This project demonstrates that a depth limited Decision Tree can achieve strong performance while remaining highly interpretable on the Census Income classification task. The model performs particularly well on the dominant lower income class and provides a clear ranking of features that influence income predictions.

Random Forest is the next logical choice because it reduces the overfitting that single decision trees are prone to by averaging predictions across many independently trained trees. Each tree sees different subsets of data and features, allowing the ensemble to capture minority-class patterns and stabilize noisy splits. This leads to higher overall generalization performance, especially on imbalanced datasets like the census income data. Feature importance estimates also become more reliable because they are aggregated across many models rather than driven by a few dominant splits. As a result Random Forest typically achieves better recall, higher F1 scores and more robust predictions compared to a standalone Decision Tree. At least thats the thought, running random forests on the dataset should give us insights into this :)
