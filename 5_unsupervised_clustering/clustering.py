from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

try:
    import hdbscan
except ImportError:  # pragma: no cover - optional dependency
    hdbscan = None


def fit_clusterer(
    features: np.ndarray,
    algorithm: str,
    cluster_count: int | None,
    random_state: int,
    dbscan_eps: float,
    min_samples: int,
) -> np.ndarray:
    if algorithm == "kmeans":
        if cluster_count is None:
            raise ValueError("cluster_count is required for kmeans")
        clusterer = KMeans(
            n_clusters=cluster_count,
            random_state=random_state,
            n_init=10,
        )
        return clusterer.fit_predict(features)

    if algorithm == "gmm":
        if cluster_count is None:
            raise ValueError("cluster_count is required for gmm")
        clusterer = GaussianMixture(
            n_components=cluster_count,
            random_state=random_state,
        )
        return clusterer.fit_predict(features)

    if algorithm == "dbscan":
        clusterer = DBSCAN(eps=dbscan_eps, min_samples=min_samples)
        return clusterer.fit_predict(features)

    if algorithm == "hdbscan":
        if hdbscan is None:
            raise RuntimeError(
                "hdbscan is not installed; install it separately to use "
                "this algorithm"
            )
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_samples)
        return clusterer.fit_predict(features)

    raise ValueError(f"Unsupported clustering algorithm: {algorithm}")


def choose_cluster_count(
    features: np.ndarray,
    algorithm: str,
    random_state: int,
    minimum_k: int = 2,
    maximum_k: int = 8,
) -> tuple[int | None, float | None]:
    if features.shape[0] < minimum_k + 1:
        return None, None

    best_k: int | None = None
    best_score: float | None = None
    upper_bound = min(maximum_k, features.shape[0] - 1)

    for cluster_count in range(minimum_k, upper_bound + 1):
        if algorithm == "kmeans":
            clusterer = KMeans(
                n_clusters=cluster_count,
                random_state=random_state,
                n_init=10,
            )
            labels = clusterer.fit_predict(features)
        else:
            clusterer = GaussianMixture(
                n_components=cluster_count,
                random_state=random_state,
            )
            labels = clusterer.fit_predict(features)

        if len(np.unique(labels)) < 2:
            continue

        score = float(silhouette_score(features, labels))
        if best_score is None or score > best_score:
            best_k = cluster_count
            best_score = score

    return best_k, best_score


def scale_features(features: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(features)


def format_cluster_summary(labels: np.ndarray) -> str:
    unique, counts = np.unique(labels, return_counts=True)
    parts = [
        f"{int(label)}:{int(count)}"
        for label, count in zip(unique, counts)
    ]
    return ", ".join(parts)
