# %% [markdown]
# # **Semi-Supervised Learning with Label Propagation on Fashion-MNIST**
#
# This notebook demonstrates a full semi-supervised learning workflow using:
# - The **Fashion-MNIST** dataset loaded from `torchvision.datasets`
# - A custom implementation of **Label Propagation** from `rice_ml.semi_supervised`
# - A realistic scenario where only a *small fraction* of labels are known
#
# The notebook is designed to be educational and complete. It explains:
# - Why semi-supervised learning is useful
# - How label propagation works mathematically
# - How we prepare and scale image data
# - How performance changes as we vary the number of labeled examples
#
# This notebook will become part of the *Semi-Supervised Learning* section of your project.

# %% [markdown]
# ## **1. Imports and Setup**

# %%
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from rice_ml.semi_supervised.label_propagation import LabelPropagation

# For splitting labeled/unlabeled data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# %% [markdown]
# ## **2. Load Fashion-MNIST from TorchVision**
#
# Fashion-MNIST consists of **70,000 clothing images**, each 28×28 pixels:
# - 60,000 train images
# - 10,000 test images
#
# Each image belongs to one of 10 classes:
# T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
#
# For label propagation:
# - We flatten the 28×28 images into vectors of length 784
# - We use only a subset of labels and treat the rest as unknown (-1)

# %%
transform = transforms.Compose([transforms.ToTensor()])

train_data = datasets.FashionMNIST(
    root="datasets/", train=True, download=True, transform=transform
)
test_data = datasets.FashionMNIST(
    root="datasets/", train=False, download=True, transform=transform
)

X_train = train_data.data.numpy().reshape(len(train_data), -1) / 255.0
y_train = train_data.targets.numpy()

X_test = test_data.data.numpy().reshape(len(test_data), -1) / 255.0
y_test = test_data.targets.numpy()

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# %% [markdown]
# ## **3. Convert Training Labels into Labeled + Unlabeled**
#
# Semi-supervised learning assumes:
# - Only **a small portion** of the training labels are known
# - The rest are unlabeled
#
# We simulate this by:
# - Taking 5% of training labels as known
# - Setting the remaining 95% to **-1** (unlabeled)

# %%
labeled_fraction = 0.05  # 5% labeled

X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(
    X_train, y_train, train_size=labeled_fraction, stratify=y_train, random_state=42
)

# Create the combined dataset:
X_full = np.vstack([X_labeled, X_unlabeled])

y_full = np.concatenate([
    y_labeled,
    -1 * np.ones_like(y_unlabeled)  # unlabeled = -1
])

print("Total samples:", len(X_full))
print("Labeled:", len(X_labeled))
print("Unlabeled:", np.sum(y_full == -1))


# %% [markdown]
# ## **4. Fit Label Propagation on Partially Labeled Data**
#
# The algorithm:
#
# 1. Construct a similarity matrix from the input features  
# 2. Normalize similarities to create a propagation matrix  
# 3. Iteratively propagate labels across the graph  
# 4. Stop when convergence is reached  
#
# Your implementation uses:
# - k-nearest neighbors for similarity
# - A clamping factor `alpha` to control smoothing
#
# We now run the model.

# %%
model = LabelPropagation(k_neighbors=10, alpha=0.8, max_iter=50, tol=1e-4)
model.fit(X_full, y_full)

# Predict on the *unlabeled portion of training data*
y_full_pred = model.predict(X_full)

train_accuracy = accuracy_score(y_train, y_full_pred[: len(y_train)])  # compare only true train labels
train_accuracy


# %% [markdown]
# ## **5. Evaluate on the True Test Set**
#
# Even though label propagation never sees the true test labels, it creates a classifier indirectly.
#
# We evaluate by:
# 1. Taking each test image  
# 2. Finding its nearest neighbor in the propagated-labeled train set  
# 3. Assigning the test label as that neighbor's label  
#
# (This is built into your model's `predict()` function.)

# %%
y_test_pred = model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test Accuracy:", test_accuracy)


# %% [markdown]
# ## **6. Classification Report**

# %%
print(classification_report(y_test, y_test_pred))


# %% [markdown]
# ## **7. Visualize Propagated Labels**
#
# Let's look at 10 random predicted labels on the test set:

# %%
idx = np.random.choice(len(X_test), 10, replace=False)

plt.figure(figsize=(12, 3))
for i, j in enumerate(idx):
    plt.subplot(1, 10, i + 1)
    plt.imshow(X_test[j].reshape(28, 28), cmap="gray")
    plt.title(int(y_test_pred[j]))
    plt.axis("off")
plt.show()


# %% [markdown]
# ## **8. Experiment — Effect of Labeled Percentage**
#
# A key benefit of semi-supervised learning is high performance even with very few labeled samples.
#
# Let’s test:
#
# - 1% labeled
# - 5% labeled
# - 10% labeled
#
# and see how test accuracy changes.

# %%
fractions = [0.01, 0.05, 0.10]
results = {}

for frac in fractions:
    X_lab, X_unlab, y_lab, y_unlab = train_test_split(
        X_train, y_train, train_size=frac, stratify=y_train, random_state=42
    )

    X_all = np.vstack([X_lab, X_unlab])
    y_all = np.concatenate([y_lab, -1 * np.ones_like(y_unlab)])

    model = LabelPropagation(k_neighbors=10, alpha=0.8)
    model.fit(X_all, y_all)
    y_test_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_test_pred)
    results[frac] = acc
    print(f"Labeled fraction {frac*100:.0f}% → Test Accuracy = {acc:.4f}")


# %% [markdown]
# ## **9. Plot Accuracy vs. Labeled Fraction**

# %%
plt.figure(figsize=(7, 4))
plt.plot([f * 100 for f in results.keys()], list(results.values()), marker="o")
plt.xlabel("Percentage of labeled data")
plt.ylabel("Test accuracy")
plt.title("Label Propagation Performance vs. Labeled Data Amount")
plt.grid(True)
plt.show()

# %% [markdown]
# # **Notebook Complete**
#
# After you run this notebook:
# 1. Send me screenshots of the results (accuracy numbers, plots).  
# 2. I will write **final polished explanations** for each result section.  
# 3. Then we'll convert this into a fully polished `.ipynb` for your GitHub repo.
#
# ---
#
# ✔ Ready for you to run.
#  
# When you're done, upload screenshots and I will write the final narrative.
