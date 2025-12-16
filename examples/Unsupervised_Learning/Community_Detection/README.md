# Community Detection via Unsupervised Label Propagation

## Overview

This project applies **Community Detection** using an **unsupervised label propagation algorithm (LPA)** to uncover latent structure in high-dimensional data. Community detection is an **unsupervised learning** task whose goal is to identify groups of data points (communities) that are more strongly connected to each other than to the rest of the dataset.

Unlike supervised or semi-supervised learning:
- **No labels are provided during training**
- **The number of communities is not specified in advance**
- Communities emerge purely from the structure of a similarity graph

The objective of this analysis is **structure discovery**, not prediction.

---

## Notebook quick reference
- **Dataset:** Fashion-MNIST image embeddings (subset loaded in-notebook)
- **Expected runtime:** ~7–9 minutes on a modern laptop with the provided subsampling
- **Key parameters to tweak:** number of nearest neighbors for the graph, maximum iterations, and convergence tolerance
- **Demonstrates:** community detection via label propagation on image data and how graph structure influences discovered clusters

## Project Structure

### 1. Dataset and Representation

The example notebook demonstrates community detection on the **Fashion-MNIST** dataset.

- Each image is flattened into a **784-dimensional feature vector**
- Pixel intensities are normalized to the range \([0, 1]\)
- Ground-truth labels are **not used during training**

Because graph-based methods scale poorly with dataset size, a **random subset** of the dataset is used. This enables efficient graph construction, iterative label propagation, and clear visualization, while still preserving meaningful structure.

---

### 2. Preprocessing

The preprocessing steps include:

- Flattening image data into feature vectors  
- Normalizing pixel values  
- Subsampling the dataset for computational efficiency  

No dimensionality reduction is applied prior to graph construction in order to preserve local similarity relationships.

---

### 3. Similarity Graph Construction (k-NN Graph)

Community detection is performed on a **graph representation** of the data:

- Each data point corresponds to a **node**
- Edges are constructed using a **k-nearest-neighbor (k-NN)** graph based on Euclidean distance
- Edge weights are computed using an **exponential decay function** of pairwise distances

The resulting weighted graph captures **local neighborhood structure**, which is critical for effective label diffusion.

---

### 4. Unsupervised Label Propagation Algorithm

The community detection algorithm is based on **Unsupervised Label Propagation (LPA)**.

#### Algorithm Steps

1. Assign a **unique label** to each node
2. Iteratively update each node’s label to the **most frequent label among its neighbors**, weighted by edge strength
3. Repeat until labels stabilize or a maximum number of iterations is reached

Nodes that converge to the same label are interpreted as belonging to the same community.

#### Key Properties

- **Fully unsupervised**: no labels are clamped or fixed
- **Emergent structure**: the number of communities is determined by the graph
- **Graph-based**: performance depends on graph quality
- **Local diffusion**: labels propagate through neighborhood interactions

This algorithm is closely related to **semi-supervised label propagation**, but differs fundamentally in that **no labeled data is provided**, and the goal is **structure discovery rather than classification**.

---

## Evaluation Strategy

Community detection does not aim to predict known class labels or optimize classification accuracy. As a result:

- **Supervised metrics** (accuracy, precision, recall) are not applicable
- **Clustering metrics** assuming fixed class structure (e.g., homogeneity, NMI) can be misleading

Instead, evaluation is **qualitative and structural**, focusing on:

- Stability and coherence of discovered communities
- Alignment with known semantic categories *only for interpretation*
- Visual inspection using **PCA projections** of the original feature space

True Fashion-MNIST labels are examined **only after training** to help interpret the semantic meaning of the discovered communities.

---

## Key Findings

- The algorithm discovers multiple communities without prior knowledge of the number of classes
- Several communities align closely with semantic Fashion-MNIST categories (e.g., trousers, bags, footwear)
- Visually similar clothing types form mixed communities, reflecting inherent ambiguity in the dataset
- The k-NN similarity graph effectively captures meaningful local relationships

---

## Limitations

- Results are sensitive to the choice of `k` in the k-NN graph
- Community structure depends strongly on feature representation and distance metric
- Graph construction and propagation scale poorly to very large datasets, requiring subsampling

Despite these limitations, the method provides valuable insight into the **organization of high-dimensional image data**.

---

## Relationship to Other Modules

This community detection example complements the **Semi-Supervised Label
Propagation** module by using the same dataset and graph construction under a
different learning paradigm.

Together, these modules illustrate how **graph-based diffusion methods** can be
used for:

- Prediction with limited supervision (semi-supervised learning), and
- Exploratory structure discovery (unsupervised learning),

depending on how label information is incorporated.

---

## Conclusion

Unsupervised label propagation provides a powerful framework for **community detection on graph-structured data**. By relying solely on neighborhood relationships, the algorithm reveals meaningful latent structure without requiring labeled data or predefined class counts. This makes it a valuable exploratory tool for understanding complex, high-dimensional datasets.
