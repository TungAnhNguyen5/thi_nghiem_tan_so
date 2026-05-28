import numpy as np
import matplotlib.pyplot as plt

# Parameters
fs = 2000          # Sampling frequency (Hz)
duration = 1.0     # seconds

# Baseband proxy for multiple drones: each source has a burst window,
# a small frequency offset, and a little frequency drift.
drone_sources = [
	{"center_hz": 120, "amplitude": 1.0, "start": 0.10, "end": 0.45, "drift_hz": 18},
	{"center_hz": 260, "amplitude": 0.85, "start": 0.22, "end": 0.72, "drift_hz": -12},
	{"center_hz": 410, "amplitude": 0.75, "start": 0.50, "end": 0.90, "drift_hz": 22},
	{"center_hz": 610, "amplitude": 0.65, "start": 0.12, "end": 0.35, "drift_hz": 8},
]

# Time vector
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# Generate a bursty multi-source signal
signal = np.zeros_like(t)
for source in drone_sources:
	burst = np.where((t >= source["start"]) & (t <= source["end"]), 1.0, 0.0)
	window = np.hanning(burst.size)
	burst *= window
	center = source["center_hz"]
	drift = source["drift_hz"]
	instantaneous_frequency = center + drift * np.sin(2 * np.pi * 2 * t)
	phase = 2 * np.pi * np.cumsum(instantaneous_frequency) / fs
	signal += source["amplitude"] * burst * np.sin(phase)

noise = 0.20 * np.random.normal(0, 1, len(signal))
noisy_signal = signal + noise

# FFT
N = len(noisy_signal)
fft_result = np.fft.fft(noisy_signal)

# Frequency bins
freqs = np.fft.fftfreq(N, 1/fs)

# Magnitude
magnitude = np.abs(fft_result)

# Keep positive frequencies
half = N // 2
freqs = freqs[:half]
magnitude = magnitude[:half]

# Plot
# Close any existing figures so each run replaces the previous graph window.
plt.close("all")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
output_file = "fft_visualizer.png"

# Time domain
axes[0].plot(t[:200], noisy_signal[:200])
axes[0].set_title("Noisy Time Domain Signal")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Amplitude")

# Frequency domain
axes[1].plot(freqs, magnitude)
axes[1].set_title("FFT Spectrum")
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Magnitude")

fig.tight_layout()
fig.savefig(output_file, dpi=300, bbox_inches="tight")
plt.show()