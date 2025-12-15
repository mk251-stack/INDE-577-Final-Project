# rice_ml Source Code

The `rice_ml` package contains custom machine learning algorithms and utilities
developed for the INDE 577: Data Science & Machine Learning course.

The code prioritizes **clarity and instructional value** over raw performance,
allowing students to follow complete modeling workflows and understand core
algorithmic ideas without heavy abstraction.

---

## Package Structure

- `supervised_learning/`  
  Classification and regression models, including perceptron, logistic
  regression, k-nearest neighbors, decision trees, regression trees,
  random forests, and ensemble methods.  
  Includes both from-scratch implementations and scikit-learn benchmarks.

- `unsupervised_learning/`  
  Clustering and dimensionality reduction methods such as K-Means, DBSCAN,
  PCA, and community detection.

- `semi_supervised/`  
  Semi-supervised learning algorithms, including graph-based label propagation.

- `processing/`  
  Data preprocessing utilities such as scaling, imputation, and
  postprocessing helpers for model evaluation.

- `utils/`  
  Shared helper functions used across modules.

- `visualization/`  
  Plotting and visualization utilities for model diagnostics and results.

- `data/`  
  Internal helpers for loading or handling data within the package
  (datasets themselves live in the top-level `datasets/` directory).

---

## Development

Install the package in editable mode from the repository root:

```bash
pip install -e .
```

Run unit tests from the repository root:
```
pytest
```