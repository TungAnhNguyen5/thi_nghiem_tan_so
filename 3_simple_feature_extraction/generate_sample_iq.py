from __future__ import annotations

from pathlib import Path

import numpy as np


def burst_signal(
	t: np.ndarray,
	fs: int,
	start: float,
	end: float,
	carrier_hz: float,
	drift_hz: float,
	amplitude: float,
) -> np.ndarray:
	window = np.where((t >= start) & (t <= end), 1.0, 0.0)
	window *= np.hanning(t.size)
	instantaneous_frequency = carrier_hz + drift_hz * np.sin(2 * np.pi * 1.5 * t)
	phase = 2 * np.pi * np.cumsum(instantaneous_frequency) / fs
	return amplitude * window * np.exp(1j * phase)


def main() -> None:
	fs = 1_000_000
	duration = 0.25
	t = np.arange(0, duration, 1 / fs)

	# A compact baseband example that looks more like overlapping drone-like emitters:
	# short bursts, small frequency offsets, mild drift, and background noise.
	signal = (
		burst_signal(t, fs, 0.02, 0.09, 110_000, 2_000, 0.95)
		+ burst_signal(t, fs, 0.06, 0.16, 245_000, -3_000, 0.80)
		+ burst_signal(t, fs, 0.12, 0.22, 390_000, 1_500, 0.70)
	)

	carrier_leak = 0.12 * np.exp(1j * 2 * np.pi * 12_000 * t)
	noise = 0.18 * (
		np.random.normal(0, 1, t.size) + 1j * np.random.normal(0, 1, t.size)
	)
	iq = (signal + carrier_leak + noise).astype(np.complex64)

	interleaved = np.empty(iq.size * 2, dtype=np.float32)
	interleaved[0::2] = np.real(iq)
	interleaved[1::2] = np.imag(iq)

	output_path = Path(__file__).with_name("drone_sample_iq.dat")
	interleaved.tofile(output_path)
	print(f"Wrote {output_path}")


if __name__ == "__main__":
	main()