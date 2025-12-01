# K-Means Clustering Analysis (Income Influence)

This project clusters individuals from the Census Income dataset using demographic, education, work-hours, and financial behavior features to uncover patterns that influence income groups.

## Dataset  
- Source: `datasets/census_income.csv`  
- Rows are cleaned, categorical features are encoded, and numerical features are standardized before clustering. `income` is converted to binary (0/1) only for cluster interpretation, **not** as a clustering input feature.

## Method  
1. K-Means clustering is fitted in the full scaled feature space using:  
   - `age`, `fnlwgt`, `education_num`, `capital_gain`, `capital_loss`, `hours_per_week`, and encoded demographic categories (`occupation`, `workclass`, `marital_status`, etc.)
2. The elbow method on inertia guides cluster (K) selection.
3. PCA 2D/3D is used only for visualization, not for determining K.

## Results & Insights  
- The data forms overlapping clusters in 2D PCA, indicating high-dimensional separation.
- Scaled radar plots and real feature summaries provide clearer cluster interpretation.
- Clusters reveal meaningful socioeconomic personas that influence income:
  - One cluster strongly aligns with **high-income behavior (>50K)**.
  - Remaining clusters group lifestyle/work profiles leaning lower income.

## Usage  
### Clone the repository and run the notebook:
```bash
git clone <repo-url>
cd INDE_577_Final_Project
python --version  # should show Python 3.11.1
pip install -r requirements.txt
jupyter notebook

