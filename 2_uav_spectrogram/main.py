from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_iq(path: Path) -> np.ndarray:
	raw = np.fromfile(path, dtype=np.float32)
	if raw.size % 2 != 0:
		raise ValueError(f"IQ file {path} has an odd number of float32 values")

	iq = raw[0::2] + 1j * raw[1::2]
	return iq.astype(np.complex64, copy=False)


def compute_spectrogram(iq: np.ndarray, fs: int, nfft: int = 1024, hop: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	if iq.size < nfft:
		raise ValueError("IQ capture is too short for the selected FFT size")

	window = np.hanning(nfft).astype(np.float32)
	frame_count = 1 + (iq.size - nfft) // hop

	spec = np.empty((nfft, frame_count), dtype=np.float32)

	for index in range(frame_count):
		start = index * hop
		frame = iq[start : start + nfft]
		spectrum = np.fft.fftshift(np.fft.fft(frame * window, n=nfft))
		spec[:, index] = 20 * np.log10(np.abs(spectrum) + 1e-12)

	frequencies = np.fft.fftshift(np.fft.fftfreq(nfft, d=1 / fs)) / 1e3
	times = (np.arange(frame_count) * hop + nfft / 2) / fs
	return spec, frequencies, times


def plot_spectrogram(spec: np.ndarray, frequencies: np.ndarray, times: np.ndarray, output_path: Path) -> None:
	plt.close("all")
	fig, ax = plt.subplots(figsize=(12, 6))

	image = ax.imshow(
		spec,
		origin="lower",
		aspect="auto",
		extent=[times[0], times[-1], frequencies[0], frequencies[-1]],
		cmap="magma",
	)

	ax.set_title("UAV Spectrogram")
	ax.set_xlabel("Time (s)")
	ax.set_ylabel("Frequency (kHz)")
	fig.colorbar(image, ax=ax, label="Magnitude (dB)")
	fig.tight_layout()
	fig.savefig(output_path, dpi=300, bbox_inches="tight")
	plt.show()


def resolve_input_paths(input_argument: str) -> list[Path]:
	input_path = Path(input_argument)
	if input_path.is_dir():
		return sorted(input_path.glob("*.dat"))

	if input_path.suffix.lower() == ".dat":
		return [input_path]

	return sorted(Path.cwd().glob("*.dat"))


def main() -> None:
	parser = argparse.ArgumentParser(description="Render a UAV spectrogram from interleaved float32 IQ data")
	parser.add_argument("input", nargs="?", default=".", help="Path to an IQ .dat file or directory; defaults to all .dat files in the current folder")
	parser.add_argument("--fs", type=int, default=1_000_000, help="Sample rate in Hz")
	parser.add_argument("--output-dir", default="output", help="Directory for generated PNG files")
	args = parser.parse_args()

	input_paths = resolve_input_paths(args.input)
	if not input_paths:
		raise FileNotFoundError("No .dat files were found to process")

	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	for input_path in input_paths:
		output_path = output_dir / f"{input_path.stem}.png"
		iq = load_iq(input_path)
		spec, frequencies, times = compute_spectrogram(iq, fs=args.fs)
		plot_spectrogram(spec, frequencies, times, output_path)


if __name__ == "__main__":
	main()
