# Perceptron

This directory contains the custom Perceptron implementation and a Jupyter notebook demonstrating training, evaluation, and analysis of the algorithm using the energy.csv dataset from U.S. Energy Production.

The implementation follows the classic perceptron update rule and is consistent with the structure used in the other supervised learning modules of this repository.

## 1. Overview of the Algorithm

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

## 2. Data

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

## 3. Notebook Contents (`Perceptron.ipynb`)

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

## 4. Implementation (`perceptron.py`)

Located in:
src/rice_ml/supervised_learning/perceptron.py


Includes:
- Initialization of parameters  
- Fit method using the perceptron update rule  
- Prediction method  
- Tracking misclassifications per epoch  

## 5. Results Summary

On the energy generation classification task, the perceptron achieves:

- Training Accuracy: ~95.35%
- Test Accuracy: ~95.44%

The model demonstrates strong generalization and minimal overfitting.  
The high-dimensional one-hot-encoded feature space enables the perceptron to find a separating hyperplane effectively.
