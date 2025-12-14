# K-Means Clustering (Census Income)

This directory contains an example implementation and analysis of **K-Means
clustering** applied to the Census Income dataset.  
The goal is to uncover **latent socioeconomic behavior patterns** using
demographic, education, work, and financial features in a fully **unsupervised**
setting.

---

## Algorithm

K-Means is an unsupervised clustering algorithm that partitions data into
`k` clusters by minimizing within-cluster variance (inertia).  
Each data point is assigned to the nearest cluster centroid in feature space,
and centroids are updated iteratively until convergence.

In this analysis:
- Clustering is performed in the **full standardized feature space**
- PCA is used **only for visualization and exploratory analysis**, not for
  determining clusters

---

## Data

- Dataset: `census_income.csv`
- Source: UCI Census Income (Adult) dataset
- Features used for clustering:
  - Numeric: `age`, `fnlwgt`, `education_num`, `capital_gain`,
    `capital_loss`, `hours_per_week`
  - Categorical (one-hot encoded): `workclass`, `education`,
    `marital_status`, `occupation`, `relationship`, `race`,
    `sex`, `native_country`

The `income` variable is:
- **Excluded from clustering**
- Converted to binary (0/1) **only for post-hoc interpretation and evaluation**

---

## Methodology

1. **Preprocessing**
   - Missing values removed
   - Categorical features one-hot encoded
   - All features standardized prior to clustering

2. **Cluster Selection**
   - The number of clusters (`k`) is chosen using the **elbow method**
     based on inertia
   - PCA explained-variance analysis confirms the data is
     **high-dimensional**, reinforcing the decision to cluster in the
     original feature space

3. **Model Fitting**
   - Final K-Means model trained with `k = 3`
   - Clustering performed on the full standardized dataset

4. **Interpretation**
   - PCA 2D projections used to visualize cluster structure and centroids
   - Cluster-level summary statistics computed in the original feature space

---

## Results & Interpretation

- The elbow method indicates **k = 3** as a reasonable trade-off between
  compactness and interpretability
- PCA visualizations show overlapping clusters in 2D, which is expected when
  projecting high-dimensional data
- Cluster summaries reveal distinct **socioeconomic profiles**, driven by
  differences in:
  - education
  - working hours
  - age
  - capital gains and losses
- One cluster exhibits a **higher average income tendency**, while the remaining
  clusters represent lower-income or mixed-income behavioral patterns

Although cluster-based income alignment reaches moderate accuracy, a low
homogeneity score confirms that **income is not the primary organizing
dimension**. Instead, K-Means captures broader behavioral structure.

---

## Key Takeaways

- K-Means successfully uncovers meaningful demographic and behavioral structure
- Income correlates with this structure but does not form clean cluster boundaries
- PCA is valuable for visualization but should not be confused with clustering
  objectives
- Unsupervised results must be interpreted carefully when ground-truth labels
  are not the dominant signal
