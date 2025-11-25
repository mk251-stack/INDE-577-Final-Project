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
