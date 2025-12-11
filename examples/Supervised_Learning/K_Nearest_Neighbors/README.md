

# K-Nearest Neighbors (KNN)

This section demonstrates the implementation of the K-Nearest Neighbors classifier using the Census Income dataset. The goal is to predict whether an individual's income is <=50K or >50K using demographic and employment attributes.

The model is implemented through a reusable pipeline defined in:

src/rice_ml/supervised_learning/k_nearest_neighbors.py

and executed in the example notebook:

examples/Supervised_Learning/K_Nearest_Neighbors/K_Nearest_Neighbors.ipynb

# Method

KNN is a non-parametric classification method that predicts the class of a new instance based on the majority vote of its k nearest neighbors in feature space. Because KNN relies on distances, our pipeline includes:

• One-hot encoding for categorical variables
• Standard scaling for numerical variables
• scikit-learn’s KNeighborsClassifier

The reusable function train_knn_model() handles preprocessing, splitting, and model training. Evaluation is done through evaluate_knn_model().

# Dataset

The dataset illustrates how the structure and variability of real-world features can significantly influence the performance of machine learning algorithms. Although the task is to predict a binary outcome (income <=50K or >50K), the underlying characteristics driving this outcome are highly subjective and diverse. Individuals differ across education, occupation, work experience, demographic background, and hours worked, and each of these features varies in strength and relevance from person to person.

Because of this, the same dataset behaves very differently when processed by different algorithms. Some algorithms extract stable patterns from broad demographic trends, while others struggle when attributes do not form clean, consistent clusters. This makes the Census Income dataset a strong example for understanding how model performance depends not only on algorithm choice but also on the complexity, imbalance, and heterogeneity of the underlying feature space.

The dataset is also heavily skewed. Most individuals earn <=50K and belong to common job categories, education ranges, and work patterns. These groups form dense clusters and are easy for distance-based or rule-based models to classify. In contrast, the >50K group is much smaller and far more diverse, spanning multiple industries and education levels. This creates dispersed, irregular feature patterns, which directly impact algorithms that rely on similarity, such as KNN.

Overall, this dataset demonstrates how the variability and imbalance within real demographic data shape model behavior, influence predictive accuracy, and highlight the strengths and limitations of different machine learning approaches.

# Results

Running the model with k = 5 produced the following performance:
    Accuracy: ~0.83
    Dataset Imbalance: ~75 percent <=50K, ~25 percent >50K

I chose to stick with k = 5, because it starts as a great starting point and we get a decent accuracy with it. More importantly, the results worked in the most loogical way possible. Further explaination below.

Class-wise performance:

Class	Precision	Recall	F1-Score	 Explanation
<=50K	~0.87	    ~0.91	~0.89	   Dense, homogeneous cluster → easy for KNN
>50K	~0.67	    ~0.58	~0.62	   Spread-out, heterogeneous patterns → harder for KNN

# Interpretation

These results match real socioeconomic structure:

• Individuals earning <=50K exhibit consistent demographic and occupational patterns (education, job types, work hours), creating tight clusters in feature space that KNN classifies reliably.

• Individuals earning >50K come from more diverse backgrounds (varied industries, education, experience, job roles). This class forms multiple scattered micro-clusters, which distance-based models struggle with. Furthermore, with the diverse nature of success in society, a lot more factors from a custom dataset would be required to accurately predict higher distincition between high income groups. 

• The imbalance between classes furt v b her reinforces KNN’s bias toward the majority class which is majorly people with an income <=50k

Overall, the results are accurate, explainable, and aligned with expectations for KNN on the nature of this dataset. 