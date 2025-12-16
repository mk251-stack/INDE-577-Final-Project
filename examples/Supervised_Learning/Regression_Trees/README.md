# Regression Trees

Regression Trees are supervised learning models used to predict continuous numerical values. Instead of assigning a class label, the model repeatedly splits the feature space into regions and outputs the average target value within each leaf node.

Each internal node represents a decision rule on a single feature, each branch represents the outcome of that rule, and each leaf node stores a predicted numeric value. This structure allows regression trees to naturally capture nonlinear relationships and feature interactions without requiring explicit feature engineering.

In this project, I use a Regression Tree to predict median home values using the Boston Housing dataset. The goal is to explore how a depth constrained tree balances interpretability, predictive performance, and generalization on a relatively small but well studied regression dataset.

Regression Trees are a good fit here because they handle nonlinear relationships well, are robust to feature scaling, and provide intuitive explanations through tree structure and feature importance. Rather than training a fully grown tree that would almost certainly overfit, the focus is placed on tuning tree depth to identify a model that performs well on unseen data while remaining interpretable.

Rather than training a fully grown tree that would almost certainly overfit, I focus on tuning tree depth to identify a model that performs well on unseen data while remaining interpretable.

## Notebook quick reference
- **Dataset:** Boston Housing (`datasets/BostonHousing.csv`) predicting median home value
- **Expected runtime:** ~5–7 minutes on a modern laptop
- **Key parameters to tweak:** `max_depth`, `min_samples_leaf`, and feature preprocessing choices
- **Demonstrates:** regression tree fitting, depth tuning to avoid overfitting, and feature importance interpretation for tabular regression

# Data Set

The Boston Housing dataset contains socioeconomic and geographic indicators such as crime rate, average number of rooms, distance to employment centers, and tax rates, with the target variable being median home value.

One important observation is that the dataset contains a small number of missing values in the rm feature. Since this dataset is shared across the repository, no modifications were made to the raw CSV. Instead, missing values were handled directly within the notebook using median imputation. This ensures reproducibility while preserving dataset integrity for other models in the project.

The dataset is relatively small, which makes regression trees particularly sensitive to overfitting. This further motivated the use of depth tuning and explicit train test splits rather than relying on a fully grown tree.

 # Key Process:

The regression tree is trained using a depth limited configuration. Instead of manually selecting a depth, I evaluate performance across a range of depths and select the value that minimizes test error.

Key modeling decisions include:

- Limiting maximum depth to control variance
- Using Mean Squared Error (MSE) and Mean Absolute Error (MAE) for evaluation
- Tracking R² to assess explanatory power on the test set

The tuning process evaluates how error changes as tree depth increases. As expected, shallow trees underfit while deeper trees begin to overfit, producing a clear minimum in test error that indicates an appropriate tradeoff point.

# Evaluation Metrics and Results

The primary metrics used to evaluate performance are:

>Mean Squared Error to penalize larger prediction errors
>Mean Absolute Error for interpretability in target units
>R² to measure the proportion of variance explained by the model

After tuning, the selected model achieves:

>Low test MSE and MAE
>An R² value indicating strong explanatory power relative to a baseline mean predictor

These results suggest that the regression tree is capturing meaningful structure in the data while maintaining reasonable generalization to unseen samples.

# Feature Importance Interpretation

Regression Trees compute feature importance based on how much each feature reduces variance across all splits where it appears. Features that consistently produce informative splits receive higher importance scores.

For this dataset, the most influential features align well with domain expectations. Variables related to housing size, neighborhood quality, and location tend to dominate the splits, while less informative socioeconomic indicators contribute marginally.

This behavior reinforces the interpretability advantage of tree based models. The model is not only accurate but also transparent in how it arrives at predictions.

# Model Diagnostics and Visualization

In addition to numeric metrics, a predicted vs actual scatter plot is used to visualize model performance. Points clustered along the diagonal indicate accurate predictions, while deviations highlight regions where the model struggles.

This visualization helps reveal:

>Slight regression to the mean at extreme housing values
>Increased error variance for high priced homes
>Overall consistency across the bulk of the dataset

These patterns are expected for regression trees trained on limited data and provide useful intuition about model limitations.

# Conclusion & Outlook

While a single Regression Tree performs well and offers strong interpretability, it remains sensitive to data splits and noise. A natural extension is to apply ensemble methods such as Random Forests, which average predictions across many trees trained on different subsets of data and features.

This typically leads to:

- Lower variance
- Improved generalization
- More stable feature importance estimates

Running Random Forests on this dataset wouldve provided a useful comparison point and helps quantify how much performance is gained by sacrificing some interpretability for robustness. BUT this is an afterthought sadly, I did RF on the census dataset :( but this wouldve been nice to see too
