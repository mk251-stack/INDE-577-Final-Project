# src/rice_ml/visualization/plots.py

from __future__ import annotations

from typing import Optional, Dict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_digits_grid(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    class_names: Optional[Dict[int, str]] = None,
    title: str = "",
    n_rows: int = 4,
    n_cols: int = 8,
) -> None:
    """
    Plot a grid of grayscale images with labels (and optionally predictions).

    X is expected to be of shape (n_samples, 28, 28).
    """
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.5 * n_cols, 1.5 * n_rows))
    axes = axes.flatten()

    for i in range(n_rows * n_cols):
        ax = axes[i]
        if i < len(X):
            ax.imshow(X[i], cmap="gray")
            ax.axis("off")
            true_label = y_true[i]
            true_name = (
                class_names.get(int(true_label), str(true_label))
                if class_names is not None
                else str(true_label)
            )
            if y_pred is None:
                ax.set_title(f"{true_name}", fontsize=8)
            else:
                pred_label = y_pred[i]
                pred_name = (
                    class_names.get(int(pred_label), str(pred_label))
                    if class_names is not None
                    else str(pred_label)
                )
                ax.set_title(f"T:{true_name}\nP:{pred_name}", fontsize=8)
        else:
            ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: np.ndarray,
    class_names: Optional[Dict[int, str]] = None,
    title: str = "",
) -> None:
    """
    Simple confusion matrix plot with optional class-name mapping.
    """
    tick_labels = [
        class_names.get(int(c), str(c)) if class_names is not None else str(c)
        for c in classes
    ]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(tick_labels)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, cm[i, j], ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    plt.show()


def plot_pca_2d(
    X: np.ndarray,
    y: np.ndarray,
    class_names: Optional[Dict[int, str]] = None,
    title: str = "PCA projection",
    random_state: int = 42,
) -> None:
    """
    Project data to 2D using PCA and color points by class label.
    """
    pca = PCA(n_components=2, random_state=random_state)
    X_2d = pca.fit_transform(X)

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=y,
        s=10,
        alpha=0.7,
        cmap="tab10",
    )
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)

    if class_names is not None:
        # Build a simple legend
        unique_labels = np.unique(y)
        handles = []
        labels = []
        for c in unique_labels:
            mask = y == c
            if not np.any(mask):
                continue
            handles.append(
                plt.Line2D(
                    [], [], marker="o", linestyle="", markersize=6,
                    color=scatter.cmap(scatter.norm(c))
                )
            )
            labels.append(class_names.get(int(c), str(c)))
        plt.legend(handles, labels, title="Classes", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()
