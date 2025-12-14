# Community Detection

This directory contains example code and analysis for **Community Detection**
using an **unsupervised label propagation** algorithm.

Community detection is an unsupervised learning task whose goal is to identify
groups of data points (communities) that are more strongly connected to each
other than to the rest of the dataset. Unlike supervised or semi-supervised
learning, no labels are provided during training, and the number of communities
is not specified in advance.

---

## Algorithm

The community detection algorithm implemented here is based on
**Unsupervised Label Propagation (LPA)** applied to a similarity graph.

The core idea is as follows:

1. Each data point is represented as a node in a graph.
2. Edges connect nearby points based on a k-nearest-neighbor (k-NN) similarity
   graph.
3. Each node is initially assigned a unique label.
4. Labels are iteratively updated by adopting the most common label among a
   node’s neighbors (weighted by edge strength).
5. The process repeats until labels stabilize.

Nodes that converge to the same label are interpreted as belonging to the same
community.

### Key properties

- **Fully unsupervised**: no labels are used or fixed during training.
- **Emergent structure**: the number of communities is determined by the graph.
- **Graph-based**: results depend on the quality of the similarity graph.
- **Local diffusion**: community assignments reflect local neighborhood structure.

This algorithm is conceptually related to semi-supervised label propagation, but
differs in that *no labels are clamped* and the objective is **structure
discovery rather than classification**.

---

## Data

In the accompanying example notebook, community detection is demonstrated on the
**Fashion-MNIST** dataset.

- Each image is flattened into a 784-dimensional feature vector.
- Pixel values are normalized to the range \([0, 1]\).
- A **k-NN graph** is constructed using Euclidean distance.
- Edge weights are defined using an exponential decay of distances.

Because graph-based methods scale poorly with dataset size, a **random subset**
of the full dataset is used. This allows efficient graph construction, iterative
label propagation, and clear visualization, while still preserving meaningful
structure.

True Fashion-MNIST labels are **not used during training**. They are examined
*only after community detection* to help interpret the semantic meaning of the
discovered communities.

---

## Example Notebook

The main demonstration of this algorithm can be found in:

'examples/Unsupervised_Learning/Community_Detection/Community_Detection.ipynb'


The notebook includes:

- Graph construction using k-NN.
- Unsupervised community detection via label propagation.
- PCA visualization of discovered communities.
- Qualitative interpretation using ground-truth labels.
- A conceptual comparison with semi-supervised label propagation.

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
