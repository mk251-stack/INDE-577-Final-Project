"""
merge.py

Utility script to construct a single Census Income (Adult) dataset CSV
from the census website.

The original dataset is distributed as separate training and test files
(`adult.data` and `adult.test`) with inconsistent formatting. This script:

- Downloads both files directly from the UCI repository
- Assigns consistent column names
- Removes formatting artifacts in the target label (trailing '.')
- Concatenates train and test into a single dataset
- Exports the result as `census_income.csv`

This script is provided for transparency and reproducibility.
The generated CSV is used by example notebooks and tests.
"""

import pandas as pd

cols = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]

train_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
test_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

train = pd.read_csv(train_url, names=cols, sep=",", skipinitialspace=True)
test = pd.read_csv(test_url, names=cols, sep=",", skipinitialspace=True, comment="|", skiprows=1)

data = pd.concat([train, test], axis=0)
data["income"] = data["income"].str.replace(".", "", regex=False)

data.to_csv("census_income.csv", index=False)
print("✅ census_income.csv created successfully!")
