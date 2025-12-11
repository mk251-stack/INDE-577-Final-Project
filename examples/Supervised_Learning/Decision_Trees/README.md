# Decision Trees

Decision Trees are supervised learning models that can be used for both classification and regression.

They repeatedly split the data based on feature values. Each internal node represents a decision rule on a single feature, each branch represents the outcome of that rule, and each leaf node stores a predicted output. The tree structure makes the model easy to visualize and interpret.

I used a Decision Tree Classifier to predict whether a person’s income is greater than fifty thousand dollars or less than or equal to fifty thousand dollars based on demographic and employment features from the Adult Census Income dataset.

>Decision Trees are a good fit here because they:
>Handle both numerical and categorical variables after simple encoding.
>Capture nonlinear relationships and feature interactions automatically.
>Do not require feature scaling.
>Provide clear interpretability through the tree diagram and feature importance scores.

In this project I focus on a depth limited tree that balances interpretability and generalization instead of using a fully grown tree that would overfit or tweaking it as my expectations from this dataset with decision trees is that the skewed nature of the dataset would offer an expected result of it performing better with the lower income category and the estimation of how only a few features would be given most importance.

# Data Set Comments:

The original dataset also contains the column fnlwgt. For this project I kept fnlwgt in the raw data but dropped it from the model features because it behaves like a sampling weight and does not provide meaningful predictive signal in a simple Decision Tree.

The dataset is imbalanced. Most people fall into the lower income class. Therefore to make it further balanced, I used a stratified train test split so that the proportion of each income class is similar in both training and testing sets, which leads to a more reliable evaluation.

# Key Comments On Running Code:

Training the Decision Tree model
I train a Decision Tree Classifier using the Gini impurity criterion. To reduce overfitting I limit the complexity using parameters such as:

>max_depth set to eight
>min_samples_split set to twenty
>min_samples_leaf set to ten

Evaluation metrics I look at when seeing the output is:

>Overall accuracy on the test set
>Confusion matrix for the two income classes
>Precision, recall and F1 score for each class
>Macro and weighted averages to account for class imbalance

In my run the model achieved a test accuracy of about zero point eighty six.
Recall for the lower income class is much higher than for the higher income class, which reflects the underlying imbalance and shows that the tree is more conservative when predicting high income.

# Feature Importance after Results:

Tree based models compute a feature importance score based on how much each feature reduces impurity across all splits where it appears. Features that consistently create strong splits and reduce Gini impurity obtain higher importance values.

For this census income model the most important features are:

>marital_status_Married_civ_spouse
>education_num
>capital_gain
>capital_loss
>age
>hours_per_week

This makes sense socioeconomically. Income is strongly related to marital status, education level and capital gains. Age and hours worked per week also naturally correlate with earning potential. The Decision Tree is essentially using these variables to separate higher income individuals from lower income ones.

The model performs especially well for the lower income class, which dominates the dataset, and it provides a clear ranking of which attributes matter most for predicting income. Furthermore, realistically, we see that hihgher income population has a lot more social and intangible/tangible factors.

# Outlook:

Random Forest is the next logical choice because it reduces the overfitting that single decision trees are prone to by averaging predictions across many independently trained trees. Each tree sees different subsets of data and features, allowing the ensemble to capture minority-class patterns and stabilize noisy splits. This leads to higher overall generalization performance, especially on imbalanced datasets like the census income data. Feature importance estimates also become more reliable because they are aggregated across many models rather than driven by a few dominant splits. As a result Random Forest typically achieves better recall, higher F1 scores and more robust predictions compared to a standalone Decision Tree. At least thats the thought , running random forests on the dataset should give us insights into this :)
