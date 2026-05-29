from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

try:
    import hdbscan
except ImportError:  # pragma: no cover - optional dependency
    hdbscan = None


FEATURE_NAMES = [
    "mean_power",
    "rms_amplitude",
    "crest_factor",
    "spectral_centroid_hz",
    "spectral_spread_hz",
    "occupied_bandwidth_hz",
    "spectral_entropy",
    "spectral_flatness",
    "peak_frequency_hz",
    "phase_diff_std",
]
def load_iq(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 2 != 0:
        raise ValueError(f"IQ file {path} has an odd number of float32 values")

    iq = raw[0::2] + 1j * raw[1::2]
    return iq.astype(np.complex64, copy=False)


def resolve_input_paths(input_argument: str) -> list[Path]:
    input_path = Path(input_argument)
    if input_path.is_dir():
        return sorted(input_path.glob("*.dat"))

    if input_path.suffix.lower() == ".dat":
        return [input_path]

    return sorted(Path.cwd().glob("*.dat"))


def window_iq(iq: np.ndarray, window_size: int, hop_size: int) -> list[np.ndarray]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if hop_size <= 0:
        raise ValueError("hop_size must be positive")
    if iq.size < window_size:
        raise ValueError("IQ capture is too short for the selected window size")

    return [iq[start : start + window_size] for start in range(0, iq.size - window_size + 1, hop_size)]


def occupied_bandwidth(iq: np.ndarray, fs: float, occupancy: float = 0.99) -> float:
    if not 0 < occupancy < 1:
        raise ValueError("occupancy must be between 0 and 1")
    if iq.size < 2:
        raise ValueError("IQ capture is too short to estimate bandwidth")

    window = np.hanning(iq.size).astype(np.float64)
    spectrum = np.fft.fftshift(np.fft.fft(iq * window))
    power = np.abs(spectrum) ** 2
    power_sum = float(power.sum())
    if power_sum == 0:
        return 0.0

    normalized_power = power / power_sum
    cumulative_power = np.cumsum(normalized_power)
    lower = (1 - occupancy) / 2
    upper = 1 - lower
    start = int(np.searchsorted(cumulative_power, lower, side="left"))
    end = int(np.searchsorted(cumulative_power, upper, side="left"))
    bandwidth_bins = max(end - start, 0)
    return float(bandwidth_bins * fs / iq.size)


def extract_features(frame: np.ndarray, fs: float) -> np.ndarray:
    if frame.size < 8:
        raise ValueError("IQ window is too short for feature extraction")

    magnitude = np.abs(frame).astype(np.float64, copy=False)
    power = magnitude**2
    mean_power = float(np.mean(power))
    rms_amplitude = float(np.sqrt(mean_power))
    crest_factor = float(np.max(magnitude) / rms_amplitude) if rms_amplitude > 0 else 0.0

    window = np.hanning(frame.size).astype(np.float64)
    spectrum = np.fft.fftshift(np.fft.fft(frame * window))
    spectral_power = np.abs(spectrum) ** 2
    power_sum = float(spectral_power.sum())
    frequencies = np.fft.fftshift(np.fft.fftfreq(frame.size, d=1 / fs))

    if power_sum > 0:
        probabilities = spectral_power / power_sum
        spectral_centroid_hz = float(np.sum(frequencies * probabilities))
        spectral_spread_hz = float(
            np.sqrt(np.sum(((frequencies - spectral_centroid_hz) ** 2) * probabilities))
        )
        peak_frequency_hz = float(frequencies[int(np.argmax(spectral_power))])
        probability_nz = probabilities[probabilities > 0]
        spectral_entropy = float(-np.sum(probability_nz * np.log2(probability_nz)) / np.log2(frame.size))
        spectral_flatness = float(np.exp(np.mean(np.log(spectral_power + 1e-12))) / np.mean(spectral_power))
    else:
        spectral_centroid_hz = 0.0
        spectral_spread_hz = 0.0
        peak_frequency_hz = 0.0
        spectral_entropy = 0.0
        spectral_flatness = 0.0

    phase_diff = np.angle(frame[1:] * np.conj(frame[:-1]))
    phase_diff_std = float(np.std(phase_diff)) if phase_diff.size else 0.0
    bandwidth_hz = occupied_bandwidth(frame, fs=fs)

    return np.array(
        [
            mean_power,
            rms_amplitude,
            crest_factor,
            spectral_centroid_hz,
            spectral_spread_hz,
            bandwidth_hz,
            spectral_entropy,
            spectral_flatness,
            peak_frequency_hz,
            phase_diff_std,
        ],
        dtype=np.float64,
    )


def build_feature_matrix(
    iq: np.ndarray,
    fs: float,
    window_size: int,
    hop_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = window_iq(iq, window_size=window_size, hop_size=hop_size)
    feature_rows = [extract_features(frame, fs=fs) for frame in frames]
    features = np.vstack(feature_rows)
    frame_indices = np.arange(len(frames), dtype=np.int32)
    frame_times = (frame_indices * hop_size + window_size / 2) / fs
    return features, frame_indices, frame_times


def _fit_clusterer(
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
        clusterer = KMeans(n_clusters=cluster_count, random_state=random_state, n_init=10)
        return clusterer.fit_predict(features)

    if algorithm == "gmm":
        if cluster_count is None:
            raise ValueError("cluster_count is required for gmm")
        clusterer = GaussianMixture(n_components=cluster_count, random_state=random_state)
        return clusterer.fit_predict(features)

    if algorithm == "dbscan":
        clusterer = DBSCAN(eps=dbscan_eps, min_samples=min_samples)
        return clusterer.fit_predict(features)

    if algorithm == "hdbscan":
        if hdbscan is None:
            raise RuntimeError("hdbscan is not installed; install it separately to use this algorithm")
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
            clusterer = KMeans(n_clusters=cluster_count, random_state=random_state, n_init=10)
            labels = clusterer.fit_predict(features)
        else:
            clusterer = GaussianMixture(n_components=cluster_count, random_state=random_state)
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


def write_window_csv(
    output_path: Path,
    input_name: str,
    frame_indices: np.ndarray,
    frame_times: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["input_file", "window_index", "window_time_s", *FEATURE_NAMES, "cluster_label"])
        for window_index, window_time, row, label in zip(frame_indices, frame_times, features, labels):
            writer.writerow(
                [
                    input_name,
                    int(window_index),
                    f"{float(window_time):.6f}",
                    *[f"{float(value):.6f}" for value in row],
                    int(label),
                ]
            )


def format_cluster_summary(labels: np.ndarray) -> str:
    unique, counts = np.unique(labels, return_counts=True)
    parts = [f"{int(label)}:{int(count)}" for label, count in zip(unique, counts)]
    return ", ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Window IQ data, extract features, and cluster windows without labels"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=".",
        help="Path to an IQ .dat file or directory; defaults to all .dat files in the current folder",
    )
    parser.add_argument("--fs", type=float, default=1_000_000, help="Sample rate in Hz")
    parser.add_argument("--window-size", type=int, default=4096, help="Window size in samples")
    parser.add_argument("--hop-size", type=int, default=2048, help="Hop size between windows in samples")
    parser.add_argument(
        "--algorithm",
        choices=("kmeans", "gmm", "dbscan", "hdbscan"),
        default="kmeans",
        help="Clustering algorithm to use",
    )
    parser.add_argument(
        "--cluster-count",
        default="auto",
        help='Number of clusters for kmeans/gmm, or "auto" to select the best value with silhouette score',
    )
    parser.add_argument("--dbscan-eps", type=float, default=0.9, help="DBSCAN epsilon in scaled feature space")
    parser.add_argument("--min-samples", type=int, default=4, help="Minimum samples for DBSCAN/HDBSCAN")
    parser.add_argument("--output-dir", default="output", help="Directory for generated CSV and PNG files")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for clustering")
    args = parser.parse_args()

    input_paths = resolve_input_paths(args.input)
    if not input_paths:
        raise FileNotFoundError("No .dat files were found to process")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_path in input_paths:
        iq = load_iq(input_path)
        features, frame_indices, frame_times = build_feature_matrix(
            iq,
            fs=args.fs,
            window_size=args.window_size,
            hop_size=args.hop_size,
        )
        scaled_features = scale_features(features)

        cluster_count: int | None
        if args.algorithm in {"kmeans", "gmm"} and args.cluster_count == "auto":
            cluster_count, silhouette = choose_cluster_count(
                scaled_features,
                algorithm=args.algorithm,
                random_state=args.random_state,
            )
            if cluster_count is None:
                cluster_count = 2
            print(
                f"{input_path.name}: selected {cluster_count} clusters for {args.algorithm}"
                + (
                    f" (silhouette={silhouette:.3f})"
                    if silhouette is not None
                    else ""
                )
            )
        elif args.algorithm in {"kmeans", "gmm"}:
            cluster_count = int(args.cluster_count)
        else:
            cluster_count = None

        labels = _fit_clusterer(
            scaled_features,
            algorithm=args.algorithm,
            cluster_count=cluster_count,
            random_state=args.random_state,
            dbscan_eps=args.dbscan_eps,
            min_samples=args.min_samples,
        )

        summary = format_cluster_summary(labels)
        print(f"{input_path.name}: {args.algorithm} labels -> {summary}")

        csv_output = output_dir / f"{input_path.stem}_{args.algorithm}_windows.csv"
        write_window_csv(
            csv_output,
            input_name=input_path.name,
            frame_indices=frame_indices,
            frame_times=frame_times,
            features=features,
            labels=labels,
        )

        embedding = project_for_plot(scaled_features)
        plot_output = output_dir / f"{input_path.stem}_{args.algorithm}_clusters.png"
        plot_clusters(
            embedding,
            labels,
            plot_output,
            title=f"{input_path.stem} - {args.algorithm.upper()} clustering",
        )

        print(f"Wrote {csv_output}")
        print(f"Wrote {plot_output}")


if __name__ == "__main__":
    main()