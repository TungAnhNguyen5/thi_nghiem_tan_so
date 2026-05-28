import numpy as np
import matplotlib.pyplot as plt

# Parameters
fs = 2000          # Sampling frequency (Hz)
f = 100             # Signal frequency (Hz)
duration = 1.0     # seconds

# Time vector
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# Generate sine wave
signal = np.sin(2 * np.pi * f * t)
noise = 0.3 * np.random.normal(0, 1, len(signal))
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