from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from features import FEATURE_NAMES


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
        writer.writerow(
            [
                "input_file",
                "window_index",
                "window_time_s",
                *FEATURE_NAMES,
                "cluster_label",
            ]
        )
        for window_index, window_time, row, label in zip(
            frame_indices,
            frame_times,
            features,
            labels,
        ):
            writer.writerow(
                [
                    input_name,
                    int(window_index),
                    f"{float(window_time):.6f}",
                    *[f"{float(value):.6f}" for value in row],
                    int(label),
                ]
            )
