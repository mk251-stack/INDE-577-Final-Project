# DBSCAN Clustering

This directory contains example code and analysis for **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** applied to an energy dataset.  
The workflow is designed to handle **large, high-dimensional datasets** through dimensionality reduction, feature scaling, and memory-aware preprocessing.

---

## Algorithm

DBSCAN is an **unsupervised, density-based clustering algorithm** that groups observations based on local neighborhood density rather than centroid distance.  
Key characteristics:

- Does **not** require specifying the number of clusters in advance
- Can discover **arbitrarily shaped clusters**
- Explicitly identifies **noise and outliers**
- Sensitive to feature scale and dimensionality

Core hyperparameters:
- `eps`: neighborhood radius
- `min_samples`: minimum number of points required to form a dense region

---

## Data

- **Dataset**: `datasets/energy.csv`
- Observations represent energy production measurements across time and categories.
- Only **numeric features** are used, as DBSCAN relies on continuous distance metrics.

Preprocessing steps:
- Selection of numeric columns only
- Standardization using `StandardScaler`
- Random subsampling of **50,000 observations** to ensure computational feasibility
- Explicit memory cleanup using Python’s `gc` module

---

## Dimensionality Reduction

Because DBSCAN performs poorly in high-dimensional spaces, **Incremental PCA** is applied prior to clustering.

PCA is used to:
- Reduce dimensionality to two components
- Preserve dominant variance structure
- Enable reliable density estimation
- Support 2D visualization

Incremental PCA is chosen specifically to accommodate large datasets without exhausting memory.

---

## Clustering Procedure

1. Scale numeric features
2. Apply Incremental PCA (`n_components = 2`)
3. Rescale PCA outputs
4. Use a **k-distance plot** to guide selection of `eps`
5. Run DBSCAN with tuned hyperparameters

Final configuration:
- `eps = 0.18`
- `min_samples = 10`

---

## Results & Interpretation

- DBSCAN identifies:
  - One **dominant dense cluster** representing typical energy behavior
  - Several **smaller clusters** corresponding to less frequent patterns
  - A set of **noise points** (`label = -1`) indicating anomalous or sparse observations
- Applying DBSCAN directly to the full high-dimensional feature space yields limited structure, highlighting the importance of dimensionality reduction.
- PCA enables DBSCAN to operate effectively by mitigating the curse of dimensionality.

These results demonstrate DBSCAN’s strength in identifying **density-based structure and outliers**, rather than producing balanced or spherical clusters.

---

## Notebook Contents

The accompanying notebook includes:
- Data loading and numeric feature selection
- Feature scaling and memory optimization
- Incremental PCA reduction
- k-distance visualization for parameter tuning
- DBSCAN clustering
- Cluster labeling and visualization
- Interpretation of clusters and noise points
- Final conclusions

---

## Key Findings

- The energy dataset is dominated by **one primary usage pattern**, as evidenced by a single large, high-density cluster identified by DBSCAN.
- Direct application of DBSCAN on the raw, high-dimensional dataset resulted in poor clustering performance, highlighting the limitations of density-based methods in high-dimensional spaces.
- Applying **Incremental PCA** was essential for:
  - Reducing noise
  - Revealing meaningful density structure
  - Enabling DBSCAN to function effectively
- Hyperparameter tuning using the **k-distance plot** and eps sensitivity analysis revealed a **stable clustering region** around `eps = 0.18`.
- At this stable eps value, DBSCAN consistently identified:
  - One dominant cluster representing **normal energy usage**
  - A small number of **noise points**, corresponding to anomalous or rare behavior
- Lower eps values (e.g., `eps = 0.01`) produced fine-grained separation useful for **anomaly detection**, while higher eps values favored **robust, stable clustering**.

---

## Conclusion

This project demonstrates that DBSCAN can be a powerful unsupervised learning tool for energy consumption data **when paired with appropriate preprocessing techniques**. While the raw dataset did not exhibit strong density separation, the integration of feature scaling, subsampling, and PCA enabled DBSCAN to uncover a stable and interpretable structure.

The final results indicate that energy usage behavior is largely homogeneous, with a small number of deviations that can be effectively isolated as anomalies. As such, DBSCAN is best suited in this context for **outlier and anomaly detection** rather than broad segmentation. This workflow highlights the importance of dimensionality reduction and parameter tuning when applying density-based clustering methods to large, real-world datasets.
