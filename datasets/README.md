# Datasets

This folder contains datasets used by example notebooks and unit tests
throughout the project. All datasets are included locally to ensure
reproducibility and to avoid reliance on external downloads at runtime.

---

## Census Income Dataset (`census_income.csv`)

This dataset is derived from publicly available U.S. Census data
(commonly referred to as the *Census Income* or *Adult* dataset).

The raw data is distributed as separate training and test files with
minor formatting inconsistencies. To simplify usage across notebooks
and tests, these files were merged and cleaned into a single CSV file:

- `census_income.csv`

### Reproducibility

The script `merge.py` documents the exact steps used to construct
`census_income.csv`, including:
- assigning column names,
- merging multiple source files,
- cleaning target label formatting.

Users do **not** need to run this script to use the package; it is
included for transparency and reproducibility.

---

## Other Datasets

### Fashion-MNIST (`FashionMNIST/`)
Image dataset used for unsupervised learning and community detection
examples. The dataset was obtained from Kaggle and stored locally for
consistent access.

### Boston Housing (`BostonHousing.csv`)
Tabular regression dataset used in supervised learning examples.
The dataset was obtained from Kaggle.

### Credit Card Default (`UCI_Credit_Card.csv`)
Classification dataset used for supervised learning examples.
The dataset was obtained from Kaggle.

### Energy Dataset (`energy.csv`)
Tabular dataset used in regression examples. 
The dataset was obtained from Kaggle.
