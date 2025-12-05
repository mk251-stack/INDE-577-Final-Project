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

## Dataset

This experiment uses the **Fashion-MNIST** dataset, provided by Zalando Research.

Due to file-size limits, the raw dataset is **not included in this repository**
and must be downloaded separately.

📥 **Download here:**  
https://www.kaggle.com/datasets/zalando-research/fashionmnist

### Download instructions

#### Option 1 — Kaggle website
1. Open the link above.
2. Download the dataset ZIP file.
3. Extract all files into the project’s `datasets/` directory so they match
   the expected filenames used in the notebook.

#### Option 2 — Kaggle CLI
If you have the Kaggle API installed and configured:

```bash
kaggle datasets download -d zalando-research/fashionmnist
unzip fashionmnist.zip -d datasets/
