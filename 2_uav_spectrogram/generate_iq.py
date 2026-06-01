import numpy as np

fs = 1_000_000
duration = 3

t = np.arange(0, duration, 1/fs)

signal = np.zeros_like(t, dtype=np.complex64)

hop_freqs = [100e3, 200e3, -150e3, 50e3]

segment = len(t) // len(hop_freqs)

for i, f in enumerate(hop_freqs):
    start = i * segment
    end = (i + 1) * segment

    tt = t[start:end]

    signal[start:end] = np.exp(2j * np.pi * f * tt)

interleaved = np.empty(signal.size * 2, dtype=np.float32)
interleaved[0::2] = np.real(signal)
interleaved[1::2] = np.imag(signal)

interleaved.tofile("hopping_iq.dat")

print("Generated hopping_iq.dat")
