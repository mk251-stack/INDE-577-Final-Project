# Supervised Learning

This directory contains implementations and example notebooks for a range of **supervised learning algorithms**, organized to demonstrate both theoretical understanding and practical application on real datasets.

Each algorithm is implemented following a consistent workflow:
- Data loading and inspection
- Preprocessing and feature engineering
- Model training using custom implementations and/or scikit-learn
- Performance evaluation with appropriate metrics
- Interpretation of results and model behavior

The goal of this section is to compare classical supervised learning methods across **classification and regression tasks**, highlighting their strengths, limitations, and appropriate use cases.

---

## Included Algorithms

### Linear Regression
- Predicts a continuous target using a linear combination of features
- Serves as a baseline regression model
- Includes analysis of coefficients, residuals, and multicollinearity

### Logistic Regression
- Binary classification using a probabilistic linear model
- Evaluated using accuracy, precision, recall, F1-score, and ROC-AUC
- Highlights differences between linear decision boundaries and probabilistic outputs

### Perceptron
- Classic linear classifier trained with misclassification updates
- Implemented from scratch and compared against scikit-learn
- Demonstrates when linear separability is sufficient in high-dimensional feature spaces

### Multilayer Perceptron (MLP)
- Single-hidden-layer neural network implemented from scratch
- Uses backpropagation and gradient descent
- Compared against scikit-learn’s `MLPClassifier`
- Highlights optimization challenges, vanishing gradients, and model collapse behavior

### Decision Trees
- Tree-based classifier using hierarchical feature splits
- Depth-limited to balance interpretability and generalization
- Includes feature importance analysis and class imbalance discussion

### Regression Trees
- Tree-based regression model for continuous targets
- Depth tuning to control bias–variance tradeoff
- Evaluated using MSE, MAE, and R²
- Includes diagnostic plots and feature importance interpretation

### Random Forests
- Ensemble method combining multiple decision trees
- Applied to the Adult Census Income dataset
- Demonstrates variance reduction, improved stability, and more reliable feature importance
- Compared conceptually against single decision trees

### K-Nearest Neighbors (KNN)
- Instance-based learning method for classification
- Uses distance-based voting in feature space
- Includes preprocessing via scaling and one-hot encoding
- Evaluated using confusion matrices and classification metrics

---

## Datasets Used

Multiple real-world datasets are used across the supervised learning modules, including:
- **Adult Census Income** (classification)
- **Energy Generation** (classification)
- **Boston Housing** (regression)
- **UCI Credit Card** (classification)

Each notebook handles preprocessing locally to preserve dataset integrity across the project.

---

## Evaluation Philosophy

Evaluation metrics are chosen based on task type:
- **Classification:** accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices
- **Regression:** mean squared error (MSE), mean absolute error (MAE), R²

Special attention is paid to:
- Class imbalance
- Overfitting vs. underfitting
- Interpretability vs. predictive power
- Model assumptions and failure modes

---

## Design Principles

- **Modularity:** Core logic is implemented in reusable Python modules under `src/rice_ml`
- **Reproducibility:** Fixed random states and consistent preprocessing
- **Transparency:** Emphasis on explaining model behavior, not just reporting metrics
- **Comparison:** Whenever possible, custom implementations are benchmarked against scikit-learn

---

## How to Run

1. Install dependencies from the repository root:
   ```bash
   pip install -r requirements.txt
    ```
2. Navigate to any subdirectory within `examples/Supervised_Learning/`
3. Open the corresponding notebook and run cells sequentially

Each subfolder contains a dedicated README with algorithm-specific details and interpretations.

---

## Summary
This supervised learning section provides a structured exploration of classical machine learning models, demonstrating how different algorithms behave under varying data conditions. By combining custom implementations, scikit-learn benchmarks, and careful analysis, this module emphasizes both conceptual understanding and practical modeling judgment.