# Perceptron

This directory contains the custom Perceptron implementation and a Jupyter notebook demonstrating training, evaluation, and analysis of the algorithm using the energy.csv dataset from U.S. Energy Production.

The implementation follows the classic perceptron update rule and is consistent with the structure used in the other supervised learning modules of this repository.

## Overview of the Algorithm

The Perceptron is one of the earliest supervised learning algorithms for binary classification.  
It iteratively adjusts a linear decision boundary by updating model weights whenever a misclassification occurs.

### Key Characteristics

- Learns a linear classifier using incremental weight updates
- Uses the sign(w · x + b) decision function
- Updates only on misclassified examples
- Supports configurable learning rate and number of epochs
- Works with numerical features and binary labels
- Converges when data is linearly separable

### Update Rule

For each training example:

prediction = sign(w · x)
if prediction != actual:
w = w + eta * actual * x
b = b + eta * actual


Where:
- `w` = weight vector  
- `b` = bias  
- `eta` = learning rate  
- `actual` ∈ {–1, 1}

## Data

The perceptron notebook uses the energy.csv dataset located in:
datasets/energy.csv


### Features Used

After preprocessing:
- YEAR
- MONTH
- STATE (One-Hot Encoded)
- TYPE OF PRODUCER (One-Hot Encoded)
- ENERGY SOURCE (One-Hot Encoded)
- GENERATION (scaled)

### Target Variable

A binary classification label is created based on the median energy generation value:
- `1` = High generation (above the median)
- `–1` = Low generation (at or below the median)

## Notebook Contents (`Perceptron.ipynb`)

The notebook demonstrates the full workflow:

- Loading and inspecting the dataset
- Cleaning and preprocessing
- One-Hot Encoding categorical features
- Feature scaling with a custom StandardScaler
- Creating the binary target label
- Splitting into train/test sets
- Training the custom Perceptron model
- Evaluating performance:
  - Accuracy
  - Classification report
  - Confusion matrix
- Comparison with scikit-learn's Perceptron (optional)

## Running the Notebook

1. Install dependencies from the repository root: `pip install -r requirements.txt`.
2. Open `examples/Supervised_Learning/Perceptron/Perceptron.ipynb` in Jupyter.
3. The notebook expects the dataset at `datasets/energy.csv` and uses the `src/` directory for the custom `rice_ml` package.

## Implementation (`perceptron.py`)

Located in:
src/rice_ml/supervised_learning/perceptron.py


Includes:
- Initialization of parameters  
- Fit method using the perceptron update rule  
- Prediction method  
- Tracking misclassifications per epoch  

## Results Summary

On the energy generation classification task, the perceptron achieves:

- Training Accuracy: ~96.13%
- Test Accuracy: ~96.22%
- Scikit-learn Perceptron benchmark test accuracy: ~92.07%

The model demonstrates strong generalization and minimal overfitting.  
The high-dimensional one-hot-encoded feature space enables the perceptron to find a separating hyperplane effectively.

## Key findings
- The perceptron typically performed as a fast baseline, but its capacity is limited when classes are not linearly separable in the feature space.
- Sensitive to learning rate and epochs; convergence behavior depends on scaling and class overlap.

## Conclusions
Useful as an educational baseline and to validate preprocessing, but generally outperformed by logistic regression or multi-layer models on complex real-world structure. Best used as a stepping stone to neural networks (MLP).