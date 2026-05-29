from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def project_for_plot(features: np.ndarray) -> np.ndarray:
    if features.shape[0] < 2:
        return np.zeros((features.shape[0], 2), dtype=np.float64)

    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(features)


def plot_clusters(
    embedding: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    plt.close("all")
    fig, ax = plt.subplots(figsize=(10, 7))

    unique_labels = np.unique(labels)
    noise_label = -1
    color_labels = [label for label in unique_labels if label != noise_label]
    color_map = plt.get_cmap("tab10", max(len(color_labels), 1))

    for color_index, label in enumerate(color_labels):
        mask = labels == label
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=55,
            alpha=0.8,
            color=color_map(color_index),
            label=f"Cluster {label}",
        )

    if np.any(labels == noise_label):
        mask = labels == noise_label
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=45,
            alpha=0.7,
            color="0.45",
            marker="x",
            label="Noise",
        )

    ax.set_title(title)
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
