# Multilayer Perceptron

This directory contains example code and notes for the Multilayer Perceptron (MLP) algorithm in supervised learning, using both a custom implementation and a comparison with scikit-learn's MLPClassifier.

## Notebook quick reference
- **Dataset:** U.S. Energy Production (`datasets/energy.csv`) with median-threshold binary label
- **Expected runtime:** ~10–12 minutes on a modern laptop when training both custom and scikit-learn models
- **Key parameters to tweak:** hidden_units, learning_rate, epochs, batch_size, and regularization
- **Demonstrates:** end-to-end preprocessing + scaling, stability challenges with sigmoid activations, and comparison to `sklearn.neural_network.MLPClassifier`

## Algorithm

The Multilayer Perceptron (MLP) is a feedforward neural network composed of fully connected layers.
In this project, we implement a single-hidden-layer MLP from scratch, supporting:

- Sigmoid activation functions
- One hidden layer
- Cross-entropy loss
- Backpropagation
- Gradient descent optimization
- Xavier initialization

The objective of the MLP is to learn a nonlinear decision boundary capable of separating binary classes by adjusting weights through iterative gradient-based optimization.

Key hyperparameters include:

- hidden_units: number of neurons in the hidden layer
- learning_rate: step size for gradient descent
- epochs: number of training iterations
- random_state: seed for reproducible initialization

## Data

The dataset used in the example notebook is energy.csv, containing 496,774 observations related to electrical energy generation in the United States.

Main features include:

- YEAR: calendar year
- MONTH: month of generation
- STATE: U.S. state or region
- TYPE OF PRODUCER: type of generator
- ENERGY SOURCE: fuel or energy category
- GENERATION (Megawatthours): continuous output

A binary target variable (target) is created using the median generation value:

- 1 = High generation
- 0 = Low generation

Preprocessing steps include:

- One-Hot Encoding of categorical features
- Standardization of numerical features
- Train/Test split before model training

These steps ensure the data is formatted properly for neural network optimization.

## Notebook Workflow (`Multilayer_Perceptron.ipynb`)

The notebook walks through:

- Loading and cleaning `datasets/energy.csv`
- Creating the binary target label based on the median generation value
- One-Hot Encoding categorical variables and scaling numeric features
- Training the custom single-hidden-layer MLP implementation
- Plotting the cross-entropy loss curve and confusion matrix
- Training and evaluating scikit-learn's `MLPClassifier` for comparison

## Running the Notebook

1. Install project dependencies from the repository root with `pip install -r requirements.txt`.
2. Open `examples/Supervised_Learning/Multilayer_Perceptron/Multilayer_Perceptron.ipynb` in Jupyter.
3. Ensure `datasets/energy.csv` is present and that the `src/` directory is available on the Python path for custom imports.

## Results Summary

- Custom MLP: train accuracy ≈ 0.505 and test accuracy ≈ 0.504, indicating the single-hidden-layer model failed to learn a useful decision boundary on this dataset.
- scikit-learn `MLPClassifier`: test accuracy ≈ 0.981 with balanced precision/recall across classes; the notebook also records a `Training interrupted by user` warning during fitting.

## Key findings
- In principle, MLPs improve flexibility by modeling nonlinear decision boundaries; however, the custom implementation in this notebook did not successfully realize this potential without additional tuning.
- Performance depends strongly on architecture (hidden units), learning rate, epochs, batch size, and regularization.
- Susceptible to overfitting; train/validation curves are important diagnostics.

## Conclusions
MLP is appropriate when nonlinear relationships and interactions matter, but requires careful tuning and regularization to generalize. This outcome is expected for a minimal MLP implementation without advanced optimizers, regularization, or architectural enhancements, and serves as a diagnostic baseline rather than a production-ready neural network.