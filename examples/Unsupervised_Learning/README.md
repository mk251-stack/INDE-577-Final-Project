# Unsupervised Learning

This directory contains implementations and example notebooks for **unsupervised
learning algorithms**, which aim to discover structure, patterns, or relationships
in data **without using labeled outcomes**.

Unlike supervised learning, unsupervised methods do not receive ground-truth
targets during training. Instead, they rely on the geometry, density, or connectivity
of the data to extract meaningful insights.

The modules in this folder demonstrate how different unsupervised techniques can
be applied for **clustering**, **dimensionality reduction**, and **graph-based structure discovery**, 
using real-world datasets.

---

## Algorithms Included

### 1. K-Means Clustering
**Directory:** `K_Means_Clustering/`

- Partitions data into a fixed number of clusters by minimizing within-cluster
  variance.
- Requires the number of clusters to be specified in advance.
- Demonstrated on the **Census Income dataset**.
- Includes:
  - Feature preprocessing and scaling
  - Elbow method for cluster selection
  - PCA visualization
  - Cluster interpretation using income labels *only for analysis*

---

### 2. DBSCAN (Density-Based Clustering)
**Directory:** `DBSCAN/`

- Density-based clustering algorithm that identifies core points, border points,
  and noise.
- Does **not** require the number of clusters to be specified.
- Particularly effective for detecting outliers and irregular patterns.
- Demonstrated on a large **energy consumption dataset**.
- Includes:
  - Memory-safe preprocessing
  - Incremental PCA for dimensionality reduction
  - k-distance plot for hyperparameter selection
  - Visualization of clusters and noise points

---

### 3. Community Detection (Graph-Based Clustering)
**Directory:** `Community_Detection/`

- Fully **unsupervised label propagation** applied to a similarity graph.
- Discovers communities based on local neighborhood structure rather than
  Euclidean distance alone.
- The number of communities emerges naturally from the graph.
- Demonstrated on **Fashion-MNIST** images using a k-NN similarity graph.
- Includes:
  - Graph construction
  - Unsupervised label propagation
  - PCA visualization
  - Qualitative interpretation using true labels *after training*

---

### 4. PCA – Dimensionality Reduction
**Directory:** `PCA_Dimensionality_Reduction/`

- Linear dimensionality reduction technique based on variance maximization.
- Used to project high-dimensional data into lower-dimensional spaces.
- Demonstrated as both:
  - A standalone exploratory tool, and
  - A preprocessing step for clustering algorithms
- Includes explained variance analysis and visual interpretation.

---

## Project Structure

Each algorithm directory follows a consistent structure:

- `*.ipynb` — fully documented example notebook
- `README.md` — algorithm-specific explanation and results
- Reusable implementation code lives in:
`src/rice_ml/unsupervised_learning/`
- Unit tests for all algorithms are located in:
`tests/unit/`

This separation ensures that:
- Core logic is reusable and testable
- Notebooks focus on explanation, visualization, and interpretation

---

## Relationship to Other Modules

The unsupervised learning modules complement:

- **Supervised Learning** — where labels guide prediction
- **Semi-Supervised Learning** — where limited labels guide propagation

Together, these sections illustrate how machine learning objectives and algorithms
change depending on the availability of labeled data.
