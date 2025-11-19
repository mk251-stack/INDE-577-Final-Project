Decision Tree Classifier on Census Dataset (Predicting Workclass)

About Decision Trees
Decision Trees are a supervised machine learning algorithm used for both classification and regression tasks.
 They work by splitting data into branches based on feature conditions, eventually reaching a predicted outcome at the leaf nodes. The structure makes the model highly interpretable and easy to visualize.
I selected Decision Trees for this project because they handle both numerical and categorical variables, capture nonlinear relationships, require no feature scaling, and naturally reveal feature interactions.
This makes them well-suited for a real-world dataset like census data, where the goal is to predict a person’s workclass based on demographic and employment characteristics.
Because workclass is a multi-class categorical variable, this problem is treated as a multi-class classification task. To compare model complexity and generalization, I trained both a fully grown decision tree and a pruned version with a depth limit.

Summary of the Dataset
The dataset used in this project is the Adult Census Dataset, which includes demographic and income-related features.
Rows: Approximately 32,000


Features: A mix of numerical and categorical variables


Target: Workclass


Examples include Private, Federal-gov, Local-gov, Self-emp-not-inc, State-gov, Never-worked, Without-pay



Key Features Used
age
education and education_num
marital_status
occupation
relationship
race
sex
hours_per_week
capital_gain and capital_loss
native_country


The dataset contains class imbalance, with the Private workclass being the most common. Stratification was used during the train–test split to ensure that the proportion of each workclass category remained consistent between the training and testing sets, preserving the original class distribution and providing a fair evaluation

Key Steps and Visualizations
1. Data Loading and Preprocessing
The dataset was loaded from CSV format.
 Initial steps included reviewing missing values, cleaning rare workclass entries, and encoding categorical variables.
 Exploratory data inspection was performed to understand distributions and class frequencies.
2. Train–Test Split
The dataset was divided into 80 percent training and 20 percent testing. A stratified split was used to maintain consistent class proportions between training and testing sets.
3. Training the Decision Tree Models
Two models were trained:
Full Decision Tree
No depth limitation
Used to observe the natural complexity of the data
Expected to overfit due to unrestricted growth
Pruned Decision Tree (max_depth between 5 and 8)


More generalizable
Reduces unnecessary splits
Produces a simpler, more interpretable structure


Both models used the Gini Impurity criterion.
4. Evaluation
Model evaluation included:
Accuracy on the test set
Confusion matrix to observe class-level performance
Precision, recall, and F1-score for each workclass category
Macro-averaged and weighted metrics to account for imbalance
Visualizations included:
A plot of the full and pruned decision trees
A normalized confusion matrix
A bar chart of feature importances
A plot showing training and testing accuracy at different tree depths


5. Feature Importance
The model highlighted several features as strong predictors of workclass.
 Typical high-importance features included:
occupation
education_num
age
hours_per_week
marital_status


These features helped differentiate categories such as private sector employment, government roles, and self-employment.


How to Reproduce This Project
1. Install Dependencies
pip install numpy pandas scikit-learn matplotlib seaborn
2. Load the Census Data
Place the CSV file (for example census_income.csv) in your working directory.
3. Run the Notebook or Python Script
The script should include the following steps:
Data loading
Encoding and preprocessing
Train–test split
Training the decision tree models
Generating evaluation metrics
Plotting the tree and feature importance chart


4. Experiment With Hyperparameters
To understand overfitting and generalization, adjust parameters such as:
max_depth
min_samples_split
min_samples_leaf
criterion (gini or entropy)
Observing how these parameters affect performance provides insight into model complexity and bias–variance tradeoffs.
Important Notes About Decision Trees
They do not require feature scaling or normalization.
They can overfit if grown without restrictions.
Setting depth limits or pruning improves generalization.
They provide clear interpretability through visual tree structures and feature importance scores.
They naturally capture nonlinear relationships and feature interactions.

