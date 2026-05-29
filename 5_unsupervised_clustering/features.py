from __future__ import annotations

import numpy as np

from windowing import window_iq


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
        spectral_spread_hz = float(np.sqrt(np.sum(((frequencies - spectral_centroid_hz) ** 2) * probabilities)))
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
