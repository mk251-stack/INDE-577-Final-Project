

# K-Nearest Neighbors (KNN)

This section demonstrates the implementation of the K-Nearest Neighbors classifier using the Census Income dataset. The goal is to predict whether an individual's income is <=50K or >50K using demographic and employment attributes.

The model is implemented through a reusable pipeline defined in:

src/rice_ml/supervised_learning/k_nearest_neighbors.py

and executed in the example notebook:

examples/Supervised_Learning/K_Nearest_Neighbors/K_Nearest_Neighbors.ipynb

# Algorithm

KNN is a non-parametric classification method that predicts the class of a new instance based on the majority vote of its k nearest neighbors in feature space. Because KNN relies on distances, proper preprocessing is critical and therefore our pipeline includes:

• One-hot encoding for categorical variables
• Standard scaling for numerical variables
• scikit-learn’s KNeighborsClassifier

The reusable function train_knn_model handles preprocessing, train test splitting, and model training. Evaluation is performed using evaluate_knn_model, which reports accuracy, a confusion matrix, and class level metrics.

# Dataset

The dataset illustrates how the structure and variability of real-world features can significantly influence the performance of machine learning algorithms. Although the task is to predict a binary outcome (income <=50K or >50K), the underlying characteristics driving this outcome are highly subjective and diverse. Individuals differ across education, occupation, work experience, demographic background, and hours worked, and each of these features varies in strength and relevance from person to person.

Because of this, the same dataset behaves very differently when processed by different algorithms. Some algorithms extract stable patterns from broad demographic trends, while others struggle when attributes do not form clean, consistent clusters. This makes the Census Income dataset a strong example for understanding how model performance depends not only on algorithm choice but also on the complexity, imbalance, and heterogeneity of the underlying feature space.

The dataset is also heavily skewed. Most individuals earn <=50K and belong to common job categories, education ranges, and work patterns. These groups form dense clusters and are easy for distance-based or rule-based models to classify. 

In contrast, individuals earning more than 50K form a much smaller and more heterogeneous group. They span diverse industries, education levels, and career paths, resulting in scattered and irregular feature patterns. This structure directly impacts algorithms such as KNN that rely on neighborhood similarity.

Overall, this dataset demonstrates how the variability and imbalance within real demographic data shape model behavior, influence predictive accuracy, and highlight the strengths and limitations of different machine learning approaches.

# Results

Running the model with k = 5 produced the following performance:
    Accuracy: ~0.83
    Dataset Imbalance: ~75 percent <=50K, ~25 percent >50K

I chose the value k = 5 because I see it as a reasonable balance between sensitivity to local patterns and robustness to noise. This choice yields stable performance while remaining consistent with expected KNN behavior on this dataset.

Class-wise performance:

Interpretation: Dense and homogeneous clusters that are well captured by KNN

Class	Precision	Recall	F1-Score	 Explanation
<=50K	~0.87	    ~0.91	~0.89	   Dense, homogeneous cluster → easy for KNN
>50K	~0.67	    ~0.58	~0.62	   Spread-out, heterogeneous patterns → harder for KNN

# Interpretation

These results align well with the underlying socioeconomic structure of the data.

Individuals earning less than or equal to 50K tend to exhibit more consistent demographic and occupational patterns, forming tight clusters in feature space that KNN can classify reliably.

In contrast, individuals earning more than 50K come from a wide range of backgrounds, industries, and career paths. This class forms multiple scattered micro-clusters rather than a single cohesive group, which distance-based models struggle to capture effectively. Additionally, many factors influencing higher income such as professional networks, role seniority, and industry specific dynamics are not explicitly represented in the dataset.

The strong class imbalance further reinforces KNN’s bias toward the majority class, as most local neighborhoods are dominated by ≤50K observations.

Overall, the results are accurate, explainable, and consistent with expectations for a KNN classifier applied to an imbalanced and heterogeneous real-world dataset.

# Conclusion

KNN works exactly the way it should here. It performs strongly on the majority class, where patterns are dense and predictable, and predictably struggles on the higher income class, where outcomes are more diverse and harder to cluster. The overall accuracy is solid, but the confusion matrix makes it clear that distance alone is not enough to cleanly separate higher earners.

That in itself acts as a perfect reflection of the data. Income at the higher end is shaped by factors that are noisy, indirect, or simply not captured in the dataset. KNN exposes this limitation very clearly.

As a baseline, KNN is useful, interpretable, and is very straightforward in what can and cannot be done. Its behavior sets the stage for other algos which are better suited to handle heterogeneity and imbalance when the problem demands more than local similarity.