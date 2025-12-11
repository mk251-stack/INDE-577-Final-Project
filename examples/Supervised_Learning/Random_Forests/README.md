# Random Forests

Random Forest is an ensemble learning method that extends decision trees by training many trees on different subsets of the data and combining their predictions. Each tree is built using bootstrap sampling, and at every split only a random subset of features is considered. This randomness reduces variance and prevents the model from memorizing noise or overfitting to dominant patterns in the dataset.

I use a Random Forest Classifier to predict whether a person’s income is greater than fifty thousand dollars or less than or equal to fifty thousand dollars using the same Adult Census Income dataset as the Decision Tree section. Random Forests are a natural next step because they correct several limitations of single trees while keeping the interpretability of tree based models.

Random Forests are well suited for this dataset because they:
>Reduce overfitting by averaging predictions across many trees.
>Handle nonlinear interactions and complex feature relationships more effectively.
>Remain stable even when the dataset contains noisy or correlated variables.
>Provide reliable feature importance estimates aggregated across the entire ensemble.

The goal was to see whether the Random Forest improves recall for the high income class and stabilizes model performance given the skewed distribution of income labels.

# Data Set Comments:

The same preprocessing steps apply here as in Decision Trees. The dataset is imbalanced, with far more individuals earning less than fifty thousand dollars. Therefore the stratified train test split remains essential to maintain consistent class proportions. The fnlwgt column is again dropped before training because it does not contribute meaningful signal to the model.

After one hot encoding the categorical variables the final feature matrix contains ninety nine columns, and the Random Forest is trained directly on these encoded inputs.

# Key Comments On Running Code:

Training the Random Forest model. I train a Random Forest Classifier using the following hyperparameters:

    n_estimators set to three hundred
    max_depth left as None so each tree can grow naturally
    min_samples_split set to two
    min_samples_leaf set to one
    n_jobs set to minus one for full parallelism

This configuration produces a reasonably strong baseline model without pruning individual trees.

Evaluation metrics I focus on to analyse is:

    Test set accuracy
    Precision, recall and F1 score for each income class
    Macro and weighted averages to evaluate balance
    Confusion matrix to observe misclassification patterns

In my run the model achieved a test accuracy of roughly zero point eighty six, slightly higher than the Decision Tree but following similar behavior. Recall for the higher income class improves compared to the pruned single tree, but the model still reflects the underlying imbalance where predicting high income remains more difficult. However the Random Forest benefits from variance reduction and produces more stable predictions.

# Feature Importance after Results:

Because Random Forest computes importance across many trees, the resulting ranking is more robust and less sensitive to individual splits. Importance reflects the average reduction in impurity contributed by each feature throughout the forest.

For this dataset the most influential predictors are:

    marital_status_Married_civ_spouse
    capital_gain
    education_num
    age
    marital_status_Never_married
    hours_per_week

These features align with socioeconomic expectations and how the values have slightly been tweaked from the decision tree algo run. Capital gains and education level strongly correlate with high earning potential. Marital status indicators capture household and financial stability. Age and weekly work hours contribute naturally to income differences. The Random Forest reinforces these themes and spreads weight across a broader set of features than a single tree, showing that earning potential is influenced by multiple interacting factors.

The confusion matrix confirms the model performs very well for the lower income class and moderately for the higher income class. This is expected due to class imbalance and the nature of the underlying population distribution. Nonetheless Random Forest provides better generalization and smoother decision boundaries compared to the single tree.

# Outlook:

Random Forest serves as a practical improvement over Decision Trees by addressing the overfitting and split dominance problems of individual trees. It benefits from ensemble diversity and produces more stable, higher recall predictions, especially for minority outcomes like high income in this dataset. The importance estimates are also more reliable since they reflect aggregated behavior across many models rather than the structure of a single tree.

Random Forest forms a strong baseline for tabular classification. Future extensions may include tuning hyperparameters such as max_depth, max_features and class weights or exploring Gradient Boosting methods, which could push performance further especially for the high income group.