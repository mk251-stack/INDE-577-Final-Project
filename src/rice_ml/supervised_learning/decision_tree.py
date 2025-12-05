import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def _project_root():
    current_dir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(current_dir, "..", "..", ".."))


def load_census_dataset(target_col="income"):
    root = _project_root()
    csv_path = os.path.join(root, "datasets", "census_income.csv")

    df = pd.read_csv(csv_path)

    y = df[target_col].astype(str).str.contains(">") .astype(int)
    X = df.drop(columns=[target_col,"fnlwgt"])

    cat_cols = X.select_dtypes(include=["object"]).columns
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    return X, y


def prepare_train_test(X, y, test_size=0.25, random_state=42):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def train_decision_tree_classifier(
    X_train,
    y_train,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
):
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_classifier(model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return acc, cm, report, y_pred
