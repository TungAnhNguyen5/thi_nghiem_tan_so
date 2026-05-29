from __future__ import annotations

import argparse
import sys
from pathlib import Path

# NOTE: tests import this file directly by path using importlib.
# Ensure the folder containing this script is on sys.path so sibling-module imports work.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import numpy as np  # noqa: E402

from clustering import (  # noqa: E402
    choose_cluster_count,
    format_cluster_summary,
    fit_clusterer,
    scale_features,
)
from features import FEATURE_NAMES, build_feature_matrix  # noqa: E402
from io_utils import load_iq, resolve_input_paths, write_window_csv  # noqa: E402
from viz import plot_clusters, project_for_plot  # noqa: E402

# Backwards-compatible alias for tests/previous code.
_fit_clusterer = fit_clusterer


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
                + (f" (silhouette={silhouette:.3f})" if silhouette is not None else "")
            )
        elif args.algorithm in {"kmeans", "gmm"}:
            cluster_count = int(args.cluster_count)
        else:
            cluster_count = None

        labels = fit_clusterer(
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
