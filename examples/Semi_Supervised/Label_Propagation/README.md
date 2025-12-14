# Semi-Supervised Label Propagation on Fashion-MNIST

## Overview
This project applies a **graph-based Label Propagation algorithm** to the
**Fashion-MNIST** dataset in a **semi-supervised learning** setting, where only a
small fraction of training samples are labeled.

Label Propagation leverages both labeled and unlabeled data by constructing a
k-nearest-neighbor (k-NN) graph and diffusing label information across the graph.
The goal of this analysis is to evaluate how effectively limited supervision can
be extended to large unlabeled datasets and to compare its performance against a
purely supervised baseline.

---

## Algorithm Overview: Label Propagation

Label Propagation is a **graph-based semi-supervised algorithm** that operates as follows:

1. Construct a k-nearest-neighbor (k-NN) graph over data points  
2. Initialize labels:
   - Labeled points → one-hot encoded labels  
   - Unlabeled points → zero vectors  
3. Iteratively propagate labels across the graph using a diffusion process:

\[
F_{t+1} = \alpha W F_t + (1 - \alpha) Y_0
\]

where:
- \( W \) is the row-normalized similarity matrix
- \( Y_0 \) encodes the initial labeled points
- \( \alpha \in (0,1) \) controls the strength of diffusion

The algorithm converges to a fixed point that assigns soft labels to all nodes.

---

## Project Structure

### 1. Load and Explore Dataset
The Fashion-MNIST dataset is loaded directly from raw IDX files using a custom
data loader. Images are flattened into 784-dimensional vectors and normalized to
the range \([0, 1]\).

A random subset of the dataset is selected to keep graph construction
computationally tractable while preserving class diversity.

---

### 2. Subsampling for Graph Construction
To enable efficient k-NN graph construction:

- **6,000** training samples are used as graph nodes
- **2,000** test samples are reserved for held-out evaluation

A visual inspection of sample images confirms that the subset is representative
of all Fashion-MNIST classes.

---

### 3. Semi-Supervised Labeling Setup
A semi-supervised learning scenario is created by hiding most labels:

- A fixed number of labeled samples per class is retained
- All remaining samples are marked as unlabeled
- Labeled samples are evenly distributed across classes

This setup simulates realistic conditions where annotation is costly and limited.

---

### 4. Label Propagation Model
Label Propagation is trained on the graph-based representation of the data.

Key aspects of the model:
- A sparse k-NN graph defines neighborhood relationships
- Labels are propagated iteratively using a diffusion process
- A convergence criterion ensures numerical stability

---

### 5. Evaluation: Transductive vs. Inductive Performance
Performance is evaluated in two complementary ways:

- **Transductive evaluation**  
  Accuracy is measured on all graph nodes, including:
  - Initially labeled samples
  - Initially unlabeled samples

- **Inductive evaluation**  
  Predictions are generated for a held-out test set not used during training,
  measuring generalization to unseen data.

This distinction highlights the strengths and limitations of graph-based
semi-supervised methods.

---

### 6. Confusion Matrices and Error Analysis
Confusion matrices are computed for both:

- Graph-based (transductive) predictions
- Test-set (inductive) predictions

Class-wise precision, recall, and F1-scores reveal that:
- Distinct classes (e.g., trousers, bags, boots) are classified more accurately
- Visually similar upper-body garments are frequently confused

Correct and misclassified examples are visualized to provide qualitative insight
into model behavior.

---

### 7. Supervised Baseline: Logistic Regression
A supervised **Logistic Regression** model is trained using **only the labeled
samples** available to the semi-supervised model.

This baseline:
- Does not leverage unlabeled data
- Provides a reference point for evaluating the benefits and limitations of
  label propagation

Performance is evaluated on the same held-out test set.

---

### 8. Effect of the Number of Labeled Samples
Model performance is analyzed as the number of labeled samples per class varies:

\[
n_{\text{labeled}} \in \{2, 5, 10, 20, 50\}
\]

For each setting:
- A new semi-supervised labeling is generated
- Label Propagation and Logistic Regression are retrained
- Test accuracy is recorded and compared

This experiment illustrates how additional supervision impacts both approaches.

---

### 9. Hyperparameter Analysis
A small grid search is conducted to study the effect of:

- Number of neighbors in the k-NN graph
- Diffusion strength
- Similarity scaling parameter

Results confirm that graph locality and diffusion strength strongly influence
performance and convergence behavior.

---

### 10. PCA Visualization of Learned Structure
Principal Component Analysis (PCA) is applied to visualize the graph training data
in two dimensions.

Points are colored by:
- True class labels
- Labels inferred by propagation

This visualization provides an intuitive view of how propagated labels align with
the underlying geometry of the dataset.

---

## Dataset

This experiment uses the **Fashion-MNIST** dataset (Zalando Research).

Due to file-size constraints, the dataset is **not included** in this repository
and must be downloaded separately.

📥 **Download:**  
https://www.kaggle.com/datasets/zalando-research/fashionmnist

### Setup Instructions

#### Option 1 — Kaggle website
1. Download the dataset ZIP  
2. Extract files into `datasets/FashionMNIST/raw/`  
3. Ensure filenames match those expected by `load_fashion_mnist_raw`

#### Option 2 — Kaggle CLI
If you have the Kaggle API installed and configured:

```bash
kaggle datasets download -d zalando-research/fashionmnist
unzip fashionmnist.zip -d datasets/
```

---

## Evaluation Metrics

### Accuracy (Primary Metric)
Accuracy is reported for:
- Transductive predictions on graph nodes
- Inductive predictions on held-out test data
- Supervised baseline predictions

### Class-Level Metrics
Precision, recall, and F1-scores are used to analyze per-class performance and
identify systematic confusion patterns.

### Qualitative Evaluation
Visual inspection of predicted images and PCA embeddings supports the quantitative
results and highlights intrinsic dataset ambiguity.

---

## Key Findings
- Label Propagation achieves meaningful accuracy with **extremely limited labeled data**, demonstrating effective use of unlabeled samples.
- Transductive performance exceeds inductive performance, reflecting the graph-based nature of the algorithm.
- Logistic Regression generalizes better as labeled data increases, despite not using unlabeled data.
- Graph construction and diffusion strength are critical to performance.
- Visually similar Fashion-MNIST classes remain challenging for both methods.

---

## Conclusion
Label Propagation provides a practical and interpretable approach to
**semi-supervised learning** in low-label regimes. While it cannot fully match the
generalization performance of strong supervised models, it effectively exploits
unlabeled data and reveals the structure of the data manifold.

This analysis demonstrates the strengths, limitations, and practical trade-offs
of graph-based semi-supervised learning on real image data.