from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def load_iq(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 2 != 0:
        raise ValueError(
            f"IQ file {path} has an odd number of float32 values"
        )

    iq = raw[0::2] + 1j * raw[1::2]
    return iq.astype(np.complex64, copy=False)


def signal_power(iq: np.ndarray) -> float:
    if iq.size == 0:
        raise ValueError("IQ capture is empty")
    return float(np.mean(np.abs(iq) ** 2))


def occupied_bandwidth(
    iq: np.ndarray,
    fs: float,
    occupancy: float = 0.99,
) -> float:
    if not 0 < occupancy < 1:
        raise ValueError("occupancy must be between 0 and 1")
    if iq.size < 2:
        raise ValueError("IQ capture is too short to estimate bandwidth")

    window = np.hanning(iq.size).astype(np.float32)
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


def spectral_entropy(iq: np.ndarray) -> float:
    if iq.size == 0:
        raise ValueError("IQ capture is empty")

    window = np.hanning(iq.size).astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft(iq * window))
    power = np.abs(spectrum) ** 2
    power_sum = float(power.sum())
    if power_sum == 0:
        return 0.0

    probabilities = power / power_sum
    probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return float(entropy / np.log2(iq.size))


def resolve_input_paths(input_argument: str) -> list[Path]:
    input_path = Path(input_argument)
    if input_path.is_dir():
        return sorted(input_path.glob("*.dat"))

    if input_path.suffix.lower() == ".dat":
        return [input_path]

    return sorted(Path.cwd().glob("*.dat"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract power, bandwidth, and entropy from IQ data"
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=".",
        help=(
            "Path to an IQ .dat file or directory; defaults to all .dat files "
            "in the current folder"
        ),
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=1_000_000,
        help="Sample rate in Hz",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for generated CSV files",
    )
    args = parser.parse_args()

    input_paths = resolve_input_paths(args.input)
    if not input_paths:
        raise FileNotFoundError("No .dat files were found to process")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_path in input_paths:
        iq = load_iq(input_path)
        power = signal_power(iq)
        bandwidth_hz = occupied_bandwidth(iq, fs=args.fs)
        entropy = spectral_entropy(iq)

        output_path = output_dir / f"{input_path.stem}.csv"
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["file", "power", "bandwidth_hz", "entropy"])
            writer.writerow(
                [
                    input_path.name,
                    f"{power:.6f}",
                    f"{bandwidth_hz:.2f}",
                    f"{entropy:.6f}",
                ]
            )

        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()