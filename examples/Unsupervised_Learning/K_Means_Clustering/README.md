# K-Means Clustering on the Census Income Dataset

## Overview

This project applies **K-Means clustering**, an **unsupervised learning** algorithm, to the **Census Income (Adult)** dataset to uncover latent socioeconomic and behavioral patterns.  
The objective is to identify groups of individuals with similar demographic, educational, and work-related characteristics **without using income labels during training**.

K-Means is used here as an **exploratory tool** to understand structure in high-dimensional socioeconomic data rather than as a predictive model.

---

## Notebook quick reference
- **Dataset:** Adult Census Income (`datasets/census_income.csv`) without using labels during clustering
- **Expected runtime:** ~6–8 minutes on a modern laptop with standard scaling
- **Key parameters to tweak:** number of clusters `k`, initialization strategy, and maximum iterations
- **Demonstrates:** centroid-based clustering, inertia/Elbow analysis, and post-hoc inspection of cluster compositions

## Project Structure

### 1. Dataset and Representation

- Dataset: `census_income.csv`
- Source: **UCI Census Income (Adult) dataset**

The dataset contains demographic, education, employment, and financial attributes describing individuals.

#### Features Used for Clustering

- **Numeric features**:
  - `age`
  - `fnlwgt`
  - `education_num`
  - `capital_gain`
  - `capital_loss`
  - `hours_per_week`

- **Categorical features (one-hot encoded)**:
  - `workclass`
  - `education`
  - `marital_status`
  - `occupation`
  - `relationship`
  - `race`
  - `sex`
  - `native_country`

The `income` variable:
- Is **excluded from clustering**
- Is converted to binary (0/1) **only for post-hoc interpretation and evaluation**

---

### 2. Preprocessing

The preprocessing pipeline includes:

- Removal of missing values
- One-hot encoding of categorical variables
- Standardization of all features using `StandardScaler`

Standardization ensures that features with larger numerical ranges do not dominate the Euclidean distance used by K-Means.

---

### 3. K-Means Clustering Algorithm

K-Means partitions data into `k` clusters by minimizing **within-cluster variance (inertia)**.

#### Algorithm Steps

1. Initialize `k` centroids
2. Assign each data point to the nearest centroid in feature space
3. Update centroids as the mean of assigned points
4. Repeat until assignments stabilize or convergence is reached

In this analysis:
- Clustering is performed in the **full standardized feature space**
- PCA is **not used for clustering**, only for visualization

---

### 4. Cluster Selection

The number of clusters is selected using the **elbow method**, which examines inertia as a function of `k`.

- A visible elbow indicates diminishing returns from increasing `k`
- Based on this analysis, **k = 3** is chosen as a balance between:
  - Cluster compactness
  - Interpretability

PCA explained-variance analysis confirms that the dataset is **high-dimensional**, reinforcing the decision to perform clustering in the original feature space rather than in reduced dimensions.

---

## Evaluation and Interpretation Strategy

Post-hoc evaluation includes:

- PCA projections of the clustered data for visualization
- Cluster-level summary statistics in the original feature space
- Comparison with the `income` variable **only for interpretation**

Metrics such as homogeneity are used cautiously, since income is not assumed to be the primary organizing factor.

---

## Key Findings

- The elbow method supports **k = 3** as a reasonable clustering choice
- K-Means uncovers distinct **socioeconomic profiles**, driven by differences in:
  - education
  - working hours
  - age
  - capital gains and losses
- One cluster exhibits a **higher average income tendency**, while others represent lower-income or mixed-income patterns
- A low homogeneity score confirms that **income is not the dominant clustering dimension**

These results suggest that K-Means captures broader behavioral structure rather than directly separating income classes.

---

## Limitations

- K-Means assumes spherical clusters and equal variance, which may not hold for socioeconomic data
- Results are sensitive to feature scaling and the choice of `k`
- High dimensionality can reduce distance interpretability
- PCA visualizations may obscure true cluster separation

Despite these limitations, K-Means provides useful insight into latent structure within complex demographic datasets.

---

## Conclusion

K-Means clustering provides a simple yet effective framework for exploring latent socioeconomic structure in the Census Income dataset. By clustering in the full standardized feature space and using income labels only for interpretation, this analysis highlights the strengths and limitations of unsupervised learning when applied to real-world demographic data.
