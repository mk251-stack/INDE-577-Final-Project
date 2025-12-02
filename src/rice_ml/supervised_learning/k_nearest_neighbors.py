import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def build_knn_pipeline(cat_cols, num_cols, n_neighbors=5):
    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ])

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    model = Pipeline([
        ("preprocess", preprocess),
        ("knn", knn),
    ])

    return model


def train_knn_model(
    df,
    target_col,
    test_size=0.2,
    random_state=42,
    n_neighbors=5,
    cat_cols=None,
    num_cols=None,
):
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    if cat_cols is None:
        cat_cols = X.select_dtypes(include=["object", "category"]).columns
    if num_cols is None:
        num_cols = X.select_dtypes(exclude=["object", "category"]).columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = build_knn_pipeline(cat_cols, num_cols, n_neighbors=n_neighbors)
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test, cat_cols, num_cols


def evaluate_knn_model(model, X_test, y_test, print_report=True):
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds)

    if print_report:
        print("Accuracy:", acc)
        print("Confusion Matrix:\n", cm)
        print("Classification Report:\n", report)

    return {"accuracy": acc, "confusion_matrix": cm, "classification_report": report}
