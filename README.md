# INDE 577 Data Science & Machine Learning – Final Project

This repository contains a teaching-focused Python package that implements
core machine learning algorithms and utilities from scratch using NumPy,
Pandas, and Matplotlib. The project emphasizes clarity, minimal abstraction,
and explicit implementations to support learning and experimentation.

The package is accompanied by example notebooks covering supervised,
unsupervised, and semi-supervised learning methods, and is intended for use
in the INDE 577 Data Science & Machine Learning course.

---

## Installation

Create and activate a virtual environment, then install the project in editable
mode with development tools:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```



## Quickstart

Train a k-Nearest Neighbors classifier on a tiny dataset:

```python
import numpy as np
from rice_ml.supervised_learning import KNNClassifier

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([0, 0, 1, 1])

clf = KNNClassifier(n_neighbors=3).fit(X, y)
predictions = clf.predict([[0.1, 0.1]])
print(predictions)
```

## Project layout

- `src/rice_ml/`: Core package source code
  - `processing/` — Scaling, imputation, and preprocessing utilities  
  - `supervised_learning/` — Supervised models (e.g., regression, k-NN, trees, ensembles)  
  - `unsupervised_learning/` — Clustering, PCA, and related methods  
  - `semi_supervised/` — Semi-supervised learning algorithms (e.g., label propagation)  
  - `visualization/` — Plotting and diagnostic helpers  
  - `utils/` — Shared helper functions
- `tests/`: Pytest-based unit tests for core functionality
- `examples/`: Jupyter notebooks demonstrating algorithms and workflows  
- `datasets/` — Small curated datasets and dataset documentation  

## Running tests

There are two supported ways to run the unit test suite.

### Option 1 — Editable install (recommended when internet access is available)

    pip install -e .[dev]
    pytest

### Option 2 — Direct source execution (offline / restricted environments)

If editable installation is not possible (for example, restricted network
access), tests can be run directly against the source tree:

    PYTHONPATH=src pytest

This method bypasses editable installation while preserving the intended
src layout and import structure.