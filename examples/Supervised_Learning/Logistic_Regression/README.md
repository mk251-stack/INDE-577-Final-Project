# Logistic Regression

INDE 577 — Supervised Learning
Custom Implementation + Scikit-Learn Benchmark

This project builds and evaluates a logistic regression classifier to predict whether an energy-generation event represents high or low electrical output using the U.S. Energy dataset (energy.csv).
The full implementation, analysis, and results are documented in:

Logistic_Regression.ipynb

## Goal of the Project

The objective is to use logistic regression to classify each observation as:

1 — High Generation: Above median megawatt-hour output

0 — Low Generation: At or below median output

This transforms the continuous variable GENERATION (Megawatthours) into a binary classification task.

## Dataset Overview

The dataset contains 496,774 rows and the following features:
- YEAR - Calendar year of energy generation
- MONTH - Month of observation
- STATE - U.S. state or region
- TYPE OF PRODUCER - Generator type (e.g., utility, IPP, industrial)
- ENERGY SOURCE - Fuel/source (coal, gas, hydro, wind, etc.)
- GENERATION (Megawatthours) - Continuous production output
- target - Derived binary label (0/1)

Additional notes:
- No missing values were found.
- After median-based labeling, classes are approximately balanced.

## Project Workflow


Steps:
- Data loading and inspection
- Target creation using median generation threshold
- One-Hot Encoding of categorical variables
- Feature scaling using a custom StandardScaler
- Exploratory Data Analysis, including:
  - Distribution plots
  - Categorical frequency charts
  - Correlation heatmap
- Model building:
  - Custom Logistic Regression (implemented from scratch)
  - Scikit-Learn LogisticRegression baseline
- Model evaluation:
  - Accuracy
  - Classification report
  - Confusion matrix
  - ROC curve and AUC
- 5-fold cross-validation
- Coefficient interpretation
- Final conclusions and recommendations

## Running the Notebook

1. Install dependencies from the project root: `pip install -r requirements.txt`.
2. Launch Jupyter and open `examples/Supervised_Learning/Logistic_Regression/Logistic_Regression.ipynb`.
3. The notebook expects the dataset at `datasets/energy.csv` and adds `src/` to the Python path for the custom implementation.


## Exploratory Data Analysis (EDA)

Key findings:
- Numeric columns include: YEAR, MONTH, GENERATION, target.
- Categorical variables show strong imbalance across states and energy sources.
- Correlation with the target is generally low, meaning the classifier relies heavily on encoded categories.
- No missing values detected.
EDA visualizations include:
- Histograms for numeric features
- Bar charts for the top 20 cat.
- Correlation heatmap

## Model Development
Custom Logistic Regression
Implemented from scratch using:
- Sigmoid function
- Gradient descent optimization
- Adjustable learning rate and maximum iterations
  
After encoding and scaling, the model is trained on a 75-dimensional feature vector.

Scikit-Learn Baseline

A standard LogisticRegression(max_iter=1000) model is used for benchmarking.

## Results
Accuracy

Both models achieved nearly identical accuracy:

Model - Accuracy
Custom Logistic Regression - 0.7948
Scikit-Learn Logistic Regression	- 0.7948
### AUC Score
Model	- AUC
Custom Logistic Regression - 0.8861
Scikit-Learn Logistic Regression - 0.8861

### Confusion Matrix
The custom model correctly classifies approximately 78–80% of each class.

### Cross-Validation
5-fold cross-validation mean accuracy: 0.7944
(Confirms stability and absence of overfitting.)

## Interpretation of Coefficients

- Positive coefficients increase the probability of high generation.
- Negative coefficients indicate association with low generation.
- Standardization allows direct comparison between coefficients.
- The most influential predictors are energy source categories and producer type categories.

## Key findings

- Logistic regression achieved strong discriminatory ability on the high vs. low generation task, with AUC ≈ 0.887 (leakage-safe pipeline).
- Because your target is built from a training-median threshold, class balance is controlled and performance is easier to interpret across classes.
- The most influential predictors typically align with domain expectations: energy source and state drive much of the separability.

## Conclusions
- Logistic Regression performs consistently and effectively on this classification task.
- The custom implementation matches Scikit-Learn performance exactly, validating correctness.
- AUC values around 0.886 indicate strong discriminative power.
- The dataset is large, balanced, and clean, making it well-suited for classification.

