from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "5_unsupervised_clustering"
        / "main.py"
    )
    spec = importlib.util.spec_from_file_location(
        "unsupervised_clustering_main", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_tone(
    duration_s: float,
    fs: int,
    carrier_hz: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    t = np.arange(0, duration_s, 1 / fs)
    return (amplitude *
            np.exp(1j * 2 * np.pi * carrier_hz * t)
            ).astype(np.complex64)


def test_window_features_shape() -> None:
    module = load_module()
    fs = 1_000_000
    rng = np.random.default_rng(123)
    noise = 0.05 * (
        rng.normal(0, 1, int(0.02 * fs))
        + 1j * rng.normal(0, 1, int(0.02 * fs))
    )
    iq = generate_tone(0.02, fs, 120_000) + noise

    features, frame_indices, frame_times = module.build_feature_matrix(
        iq.astype(np.complex64),
        fs=fs,
        window_size=1024,
        hop_size=512,
    )

    assert features.shape[1] == len(module.FEATURE_NAMES)
    assert features.shape[0] == frame_indices.shape[0] == frame_times.shape[0]
    assert features.shape[0] > 0


def test_kmeans_auto_selects_clusters_for_two_tones() -> None:
    module = load_module()
    fs = 1_000_000
    tone_a = generate_tone(0.012, fs, 90_000, amplitude=1.0)
    tone_b = generate_tone(0.012, fs, 290_000, amplitude=1.0)
    iq = np.concatenate([tone_a, tone_b]).astype(np.complex64)

    features, _, _ = module.build_feature_matrix(
        iq,
        fs=fs,
        window_size=1024,
        hop_size=512,
    )
    scaled_features = module.scale_features(features)
    cluster_count, silhouette = module.choose_cluster_count(
        scaled_features,
        algorithm="kmeans",
        random_state=42,
    )

    assert cluster_count is not None
    assert silhouette is not None
    labels = module._fit_clusterer(
        scaled_features,
        algorithm="kmeans",
        cluster_count=cluster_count,
        random_state=42,
        dbscan_eps=0.9,
        min_samples=4,
    )

    assert len(np.unique(labels)) >= 2
