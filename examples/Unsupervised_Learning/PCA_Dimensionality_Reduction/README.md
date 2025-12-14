# Dimensionality Reduction Analysis on the Energy Dataset

## Overview
This project applies **Principal Component Analysis (PCA)** to an energy dataset to reduce its dimensionality while preserving the majority of important information. PCA is an **unsupervised learning** technique that identifies new orthogonal components (principal components) that capture the most variance in the dataset.  
The goal of this analysis is to simplify a multi-feature dataset, uncover structural patterns, remove redundancy, and create lower-dimensional representations suitable for visualization and further machine-learning tasks.

---

## Project Structure

### 1. Load and Explore Dataset
The energy dataset is loaded from a CSV file and inspected to understand the variables and structure. Only numerical features are used for PCA.

### 2. Preprocessing
- Selection of numerical columns  
- Standardization using `StandardScaler`  

This ensures all features contribute equally and prevents scale-based bias in PCA.

### 3. Applying PCA
PCA is performed using **3 principal components**, based on explained-variance results.  
The reduced dataset is stored for visualization and analysis.

### 4. Component Selection (Scree Plot + Cumulative Variance)
Two key plots justify the number of components used:
- **Scree Plot**: Visualizes variance explained by each component  
- **Cumulative Variance Plot**: Shows total variance retained as components accumulate  

These confirm that reducing the dataset to 3 components is appropriate.

### 5. PCA Interpretation (Loadings + Heatmap)
PCA loadings show how each original feature contributes to the principal components.  
A heatmap visualizes these contributions and helps interpret what each component represents.

### 6. Visualization of Reduced Dimensions
To understand the dataset’s structure in lower-dimensional space:
- A **2D scatter plot** of PC1 vs. PC2
- A **3D PCA scatter plot** of PC1, PC2, PC3  

These plots reveal a planar distribution, indicating strong correlations among features.

---

## Evaluation Metrics

### Explained Variance Ratio (Primary Metric)
The explained variance ratios for the first three components are:

- **PC1:** 40.62%  
- **PC2:** 25.01%  
- **PC3:** 24.11%  

Together, these components explain:

### ➜ **89.74% of the total variance**

This indicates the reduced 3-dimensional representation preserves nearly all important information.

### Qualitative Evaluation
Because PCA is an unsupervised technique, evaluation focuses on:
- The proportion of variance retained by the selected components
- Structural patterns observed in reduced-dimensional visualizations
- Interpretability of principal component loadings

### Persisting Reduced Data (Optional)

The PCA-reduced dataset can optionally be saved to disk using the
`save_reduced_data` utility function. Persisting the reduced
representation enables reuse in downstream tasks such as clustering,
anomaly detection, or time-series analysis without recomputing PCA.


---

## Key Findings
- The first three principal components capture **~90%** of the dataset’s variance.  
- PCA reveals that most variation occurs along two dominant directions, forming a plane-like structure in 3D space.  
- Strong correlations among original energy variables make PCA highly effective.  
- Dimensionality reduction reduces complexity without sacrificing essential information.

---

## Conclusion
PCA successfully reduces the energy dataset to three components while retaining **89.74%** of the original variance. This simplifies the dataset, enhances interpretability, and prepares it for downstream tasks such as clustering or anomaly detection.  
The results confirm that PCA is an appropriate and powerful unsupervised technique for analyzing the structure of the energy dataset.

