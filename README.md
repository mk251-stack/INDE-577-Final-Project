# INDE 577 Data Science & Machine Learning Final Project

A teaching-focused Python package that implements core machine learning
building blocks (preprocessing, models, evaluation) from scratch using
NumPy, Pandas, and Matplotlib. The goal is to provide clear, dependency-light
examples for students in INDE 577.

## Installation

Create and activate a virtual environment, then install the project in editable
mode with the development tools:

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

- `src/rice_ml/`: Package source code
  - `supervised_learning/`: Models such as k-NN, linear regression, and decision trees
  - `preprocessing/`: Scaling, imputation, and preprocessing utilities
  - `post_processing/`: Analysis helpers
- `tests/`: Pytest-based unit tests
- `examples/`: Example notebooks and scripts

## Testing

Run the unit test suite with:

```bash
pytest
```
