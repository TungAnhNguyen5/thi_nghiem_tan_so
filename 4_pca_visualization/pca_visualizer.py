import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# CREATE SYNTHETIC UAV FEATURES

np.random.seed(42)

samples_per_class = 100

# Drone A features
drone_a = np.random.normal(
    loc=0.0,
    scale=1.0,
    size=(samples_per_class, 10)
)

# Drone B features
drone_b = np.random.normal(
    loc=4.0,
    scale=1.2,
    size=(samples_per_class, 10)
)

# Drone C features
drone_c = np.random.normal(
    loc=-3.0,
    scale=0.8,
    size=(samples_per_class, 10)
)

# Combine all samples
X = np.vstack((drone_a, drone_b, drone_c))

# Labels for coloring
labels = (
    ["Drone A"] * samples_per_class +
    ["Drone B"] * samples_per_class +
    ["Drone C"] * samples_per_class
)

print("Feature matrix shape:", X.shape)


# STEP 2 — STANDARDIZE FEATURES

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Data standardized.")


# STEP 3 — APPLY PCA

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("PCA output shape:", X_pca.shape)

print("\nExplained variance ratio:")
print(pca.explained_variance_ratio_)

# STEP 4 — VISUALIZE

plt.figure(figsize=(10, 7))

labels_array = np.array(labels)
unique_labels = np.unique(labels_array)

for drone in unique_labels:

    idx = labels_array == drone

    plt.scatter(
        X_pca[idx, 0],
        X_pca[idx, 1],
        label=drone,
        alpha=0.75,
        s=60
    )

plt.xlabel("Principal Component 1", fontsize=12)
plt.ylabel("Principal Component 2", fontsize=12)

plt.title("PCA Visualization of UAV RF Features", fontsize=14)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig("pca_clusters.png", dpi=300)
plt.show()