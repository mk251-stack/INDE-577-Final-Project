import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


def knn_train(X_train, y_train, k=5):
    """
    Train a simple KNN classifier.
    """
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model


def knn_predict(model, X_test):
    """
    Predict labels using a trained KNN model.
    """
    return model.predict(X_test)


def knn_evaluate(model, X_test, y_test):
    """
    Print accuracy and classification report.
    """
    y_pred = knn_predict(model, X_test)
    acc = accuracy_score(y_test, y_pred)

    print("KNN Accuracy:", acc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return acc
