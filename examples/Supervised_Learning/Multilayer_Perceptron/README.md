# Multilayer Perceptron

This directory contains example code and notes for the Multilayer Perceptron (MLP) algorithm in supervised learning, using both a custom implementation and a comparison with scikit-learn's MLPClassifier.

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
