import numpy as np
import matplotlib.pyplot as plt

# Parameters
fs = 1000          # Sampling frequency (Hz)
f = 50             # Signal frequency (Hz)
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
plt.figure(figsize=(12,5))
output_file = "fft_visualizer.png"

# Time domain
plt.subplot(1,2,1)
plt.plot(t[:200], noisy_signal[:200])
plt.title("Noisy Time Domain Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

# Frequency domain
plt.subplot(1,2,2)
plt.plot(freqs, magnitude)
plt.title("FFT Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")

plt.tight_layout()
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.show()