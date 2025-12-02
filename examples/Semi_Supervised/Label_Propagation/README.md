# Label Propagation — Fashion-MNIST Demo

This directory contains a full experimental demonstration of
graph-based **Label Propagation** applied to the Fashion-MNIST dataset.

The experiment illustrates how unlabeled data can be exploited
to propagate limited supervision using a k-NN graph.

---

## Notebook

**`Label_Propagation_final.ipynb`**

### Workflow

1. Load raw Fashion-MNIST images
2. Subsample training data for graph construction
3. Hide most labels via `make_semi_supervised_labels`
4. Train custom `LabelPropagation`
5. Evaluate:
   - Transductive performance (on graph nodes)
   - Inductive performance (on held-out test points)
6. Compare against Logistic Regression baseline
7. Study scalability as the number of labeled samples increases
8. Tune hyperparameters via grid search
9. Visualize learned structure with PCA

---

## Key results

- With **10 labels per class**:
  - Graph accuracy ≈ **0.55**
  - Test accuracy ≈ **0.52**

- Logistic Regression baseline trained on the same labeled set achieves
  ≈ **0.69** test accuracy, but does not leverage unlabeled data.

- Label Propagation performs best in extreme low-label regimes.

- Hyperparameter tuning identifies an optimal region around:
  - `n_neighbors = 10`
  - `alpha ≈ 0.90`
  - `gamma = None` or `0.01`

yielding test accuracies up to ≈ **0.66–0.67**.

---

## How to run

Activate the project virtual environment and launch:

```bash
jupyter notebook
