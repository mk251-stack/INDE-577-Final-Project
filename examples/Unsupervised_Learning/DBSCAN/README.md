# 📘 DBSCAN Clustering on Energy Dataset (with PCA Optimization)

This project applies **DBSCAN clustering** to an energy consumption dataset using a workflow optimized for **large datasets**.  
Because DBSCAN performs poorly on high-dimensional data, this project integrates **Incremental PCA**, **feature scaling**, and **memory-safe subsampling** to reliably extract meaningful clusters without running into memory errors.

---

## ⭐ Project Goals
- Identify natural consumption patterns within the energy dataset.
- Detect anomalies or irregular energy behaviors using DBSCAN.
- Overcome memory limitations and computational bottlenecks using PCA and subsampling.
- Visualize clusters clearly in a reduced 2D PCA space.

---

## 🔧 Methods Used

### **1. Data Preprocessing**
- Selected only numeric features (DBSCAN requires continuous values).
- Standardized all variables using `StandardScaler`.
- Subsampled **50,000 rows** to prevent RAM overflow.
- Performed manual memory cleanup using Python's `gc` module.

### **2. Dimensionality Reduction (PCA)**
Implemented **Incremental PCA** (`n_components=2`) to:
- Handle large datasets efficiently.
- Reduce dimensionality to reveal density-based structure.
- Enable 2D visualization of clusters.

### **3. DBSCAN Clustering**
- Ran DBSCAN on PCA-transformed and rescaled data.
- Used a **k-distance plot** to guide selection of `eps`.
- Found the most stable clustering at  
  **eps = 0.18**, **min_samples = 10**
- DBSCAN produced:
  - **3 major clusters**
  - Several small clusters
  - A small number of noise points (`cluster = -1`)

---

## 📊 Key Findings

- DBSCAN on the raw dataset produced only one cluster → not meaningful.
- After PCA + scaling, DBSCAN successfully identified **distinct density regions**.
- The algorithm revealed:
  - One **dominant cluster** representing typical energy behavior.
  - Several **smaller, unique clusters** representing less common patterns.
  - **Noise points** marking outlier or anomalous observations.
- PCA was essential for enabling DBSCAN to detect meaningful structure.

---

## 📁 Notebook Features
The included notebook contains:
- Full preprocessing pipeline  
- PCA reduction (Incremental PCA)  
- Memory-optimized workflow  
- k-distance visualization  
- Hyperparameter tuning (`eps`)  
- Final DBSCAN clustering  
- Cluster visualization  
- Summary statistics  
- Final conclusion section  

---

## ✅ Conclusion
DBSCAN, when applied directly to the high-dimensional energy dataset, was unable to form meaningful clusters.  
However, by integrating **PCA**, **standard scaling**, and **careful hyperparameter tuning**, DBSCAN was able to reveal several distinct patterns of energy usage and identify outliers.

This project demonstrates that **DBSCAN becomes a powerful clustering tool once dimensionality and scale are properly controlled**, especially on large real-world datasets.

---

## 📬 Contact / Notes
If you'd like to extend this project with:
- K-Means comparison  
- Silhouette scoring  
- Auto-tuned eps selection  
- Saving cluster outputs to CSV  

Just let me know!

