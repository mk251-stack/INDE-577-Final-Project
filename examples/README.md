# Examples

This directory contains **hands-on Jupyter notebook examples** demonstrating a wide range of machine learning algorithms implemented and analyzed as part of the INDE-577 Final Project.

The examples are organized by **learning paradigm**, with each subdirectory focusing on a specific class of methods. Every algorithm includes:
- a runnable notebook,
- clear preprocessing and modeling steps,
- visualizations and metrics,
- and an accompanying README with algorithm-specific explanations and interpretations.

---

## Directory Structure
```
examples/
├── Semi_Supervised/
│   └── Label_Propagation/
├── Supervised_Learning/
│   ├── Decision_Trees/
│   ├── Ensemble_Methods/
│   ├── K_Nearest_Neighbors/
│   ├── Linear_Regression/
│   ├── Logistic_Regression/
│   ├── Multilayer_Perceptron/
│   ├── Perceptron/
│   ├── Random_Forests/
│   └── Regression_Trees/
└── Unsupervised_Learning/
    ├── Community_Detection/
    ├── DBSCAN/
    ├── K_Means_Clustering/
    └── PCA_Dimensionality_Reduction/
```

---

## Supervised Learning

The **Supervised_Learning** folder explores classical and modern supervised models for both **classification** and **regression**.

Key themes include:
- linear vs nonlinear decision boundaries,
- bias–variance tradeoffs,
- handling high-dimensional and imbalanced data,
- interpretability versus predictive power.

Algorithms covered:
- Linear Regression  
- Logistic Regression  
- Perceptron  
- Multilayer Perceptron (custom neural network)  
- K-Nearest Neighbors  
- Decision Trees  
- Regression Trees  
- Ensemble Methods (Bagging, Boosting, Voting)  
- Random Forests  

Each notebook combines:
- custom implementations (where appropriate),
- scikit-learn benchmarks,
- detailed evaluation and interpretation.

---

## Unsupervised Learning

The **Unsupervised_Learning** folder focuses on discovering structure in unlabeled data.

Topics include:
- clustering,
- density-based methods,
- dimensionality reduction,
- graph-based community discovery.

Algorithms covered:
- K-Means Clustering  
- DBSCAN  
- Principal Component Analysis (PCA)  
- Community Detection via Label Propagation  

The notebooks emphasize:
- algorithm intuition,
- parameter sensitivity,
- visualization of learned structure,
- limitations of unsupervised methods.

---

## Semi-Supervised Learning

The **Semi_Supervised** folder demonstrates learning scenarios where **only a small fraction of data is labeled**.

Currently included:
- **Label Propagation** on graph-based data  

This section highlights:
- how unlabeled data can improve performance,
- diffusion-based learning dynamics,
- tradeoffs between supervision and structure assumptions.

---

## How to Run the Examples

1. Install dependencies from the repository root:
```bash
pip install -r requirements.txt
```
2. Navigate to any subdirectory inside `examples/`
3. Open the corresponding notebook and run cells sequentially

Each subfolder contains a dedicated README with detailed explanations of:
- the algorithm,
- the dataset used,
- preprocessing choices,
- results and interpretation.

---

## Summary

The `examples/` directory serves as the core demonstration layer of the project.
Rather than focusing solely on final performance, these notebooks emphasize model behavior, assumptions, and interpretability, providing a comprehensive and educational exploration of machine learning techniques commonly used in practice.

