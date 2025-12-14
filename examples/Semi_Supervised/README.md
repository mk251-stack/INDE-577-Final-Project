# Semi-Supervised Learning

This directory contains an implementation and example notebook for a
**semi-supervised learning algorithm**, which learns from a combination of
**a small labeled dataset and a large pool of unlabeled data**.

Semi-supervised learning bridges the gap between supervised and unsupervised
approaches by exploiting the structure of unlabeled data while still being
guided by limited supervision. These methods are particularly useful in
settings where labeling data is expensive or impractical at scale.

The material in this folder demonstrates how graph-based methods can leverage
data connectivity to extend sparse labels across a dataset.

---

## Algorithm Included

### Label Propagation (Graph-Based Semi-Supervised Learning)
**Directory:** `Label_Propagation/`

- Graph-based semi-supervised algorithm that propagates labels across a
  similarity graph.
- Uses both labeled and unlabeled samples during training.
- Primarily **transductive**, with limited inductive generalization.
- Demonstrated on the **Fashion-MNIST** image dataset using a k-NN graph.
- Includes:
  - Construction of a sparse k-NN similarity graph
  - Semi-supervised label masking with a controlled number of labeled samples per class
  - Evaluation of transductive and inductive performance
  - Comparison with a supervised Logistic Regression baseline
  - Sensitivity analysis with respect to the number of labeled samples
  - Hyperparameter analysis and PCA visualization

---

## Project Structure

The semi-supervised learning section follows the same structure as the other
learning paradigms in this repository:

- `examples/Semi_Supervised/Label_Propagation/`
  - `Label_Propagation_final.ipynb` — fully documented example notebook
  - `README.md` — algorithm-specific explanation, evaluation, and conclusions
- Reusable implementation code lives in:
  `src/rice_ml/semi_supervised/`
- Unit tests for the core components are located in:
  `tests/unit/`

This separation ensures that:
- Core algorithm logic is reusable and testable
- Notebooks focus on explanation, visualization, and interpretation
- The codebase remains consistent across learning paradigms

---

## Relationship to Other Modules

The semi-supervised learning section complements:

- **Supervised Learning** — where models rely entirely on labeled data
- **Unsupervised Learning** — where no labels are used during training

Together, these sections illustrate how learning strategies evolve as the
availability of labeled data changes, and how unlabeled data can be exploited
to improve performance in low-label regimes.
