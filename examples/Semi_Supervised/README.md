# Semi-Supervised Learning — Label Propagation

This module implements a **graph-based Label Propagation algorithm** for semi-supervised classification.  
Only a small fraction of training samples need labeled class values; the remainder of the data contribute
through the geometry of a k-nearest-neighbor (k-NN) similarity graph.

## Algorithm overview

Given:

- Feature matrix **X** for all samples (labeled and unlabeled)
- Label vector **y** where unlabeled points are marked as `-1`

the algorithm:

1. Builds a sparse **k-NN similarity graph**  
   \[
   w_{ij} = \exp(-\gamma\,||x_i-x_j||^2)
   \]

2. Normalizes rows of the similarity matrix to form a stochastic diffusion operator \(W\).

3. Initializes a label matrix **Y₀** containing one-hot vectors for labeled samples and zeros elsewhere.

4. Iteratively solves the fixed-point update

\[
F_{t+1} = \alpha W F_t + (1 - \alpha) Y_0
\]

until convergence.

Final predictions correspond to `argmax(F)` over each row.

---

## Included components

### `LabelPropagation`

Core estimator implementing sparse graph construction, iterative diffusion,
convergence checking, and both:

- **Transductive inference**: predictions over training graph nodes  
- **Inductive inference**: k-NN soft-vote approximation for unseen points

### `make_semi_supervised_labels`

Utility function that:

- Keeps a fixed number of labeled samples per class
- Assigns label `-1` to all remaining points

Used to simulate low-label regimes.

### `label_propagation_grid_search`

Performs a grid search over

- `n_neighbors`
- `alpha`
- `gamma`

Tracking convergence, runtime, and inductive test accuracy.

---

## Typical usage

```python
from rice_ml.semi_supervised import LabelPropagation
from rice_ml.semi_supervised.utils import make_semi_supervised_labels

y_semi, _, _ = make_semi_supervised_labels(y, n_labeled_per_class=10)

lp = LabelPropagation(n_neighbors=10, alpha=0.90)
lp.fit(X, y_semi)

y_graph = lp.predict()     # Transductive
y_test  = lp.predict(X_test)  # Inductive
