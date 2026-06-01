from __future__ import annotations

import numpy as np


def simulate_doppler(
    fs: int,
    duration: float,
    f0: float,
    v: float,
    closest_approach: float,
    snr_db: float = 30.0,
    drone_mode: bool = False,
    fm_depth: float = 40.0,
    fm_rate: float = 5.0,
    chirp_interval: float = 0.25,
    chirp_width: float = 0.02,
    chirp_span: float = 400.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a complex baseband signal from a source moving in 2D.

    The source moves along the x-axis at constant speed `v` with
    y=closest_approach. The observer is at the origin. Doppler-shifted
    frequency is computed from radial velocity using f_obs = f0 * c / (c -
    v_r).

    Returns:
        iq: complex64 IQ samples
        f_obs: instantaneous observed frequency in Hz (float64)
    """
    c = 3e8
    n = int(np.floor(fs * duration))
    t = np.arange(n) / fs

    # Place the source so it passes nearest to the origin near the middle
    # of the record
    x0 = -v * duration / 2.0

    # Position over time for moving object
    x = x0 + v * t
    y = np.full_like(x, closest_approach)

    # Distance and radial velocity
    r = np.sqrt(x**2 + y**2)
    v_r = (x * v) / r

    # observed frequency (classical Doppler for c >> v)
    f_obs = f0 * (c / (c - v_r))

    if not drone_mode:
        phase = 2 * np.pi * np.cumsum(f_obs) / fs
        sig = np.exp(1j * phase)
    else:
        # base carrier with FM around f_obs
        fm = fm_depth * np.sin(2 * np.pi * fm_rate * t)
        inst_freq = f_obs + fm
        phase = 2 * np.pi * np.cumsum(inst_freq) / fs
        carrier = np.exp(1j * phase)

        # pulsed chirps: repeating chirps aligned on chirp_interval
        chirp_sig = np.zeros_like(t, dtype=np.complex128)
        pulse_times = np.arange(0, duration, chirp_interval)
        for pt in pulse_times:
            start_idx = int(pt * fs)
            end_idx = min(start_idx + int(chirp_width * fs), t.size)
            if end_idx <= start_idx:
                continue
            tt = t[start_idx:end_idx] - pt
            chirp_inst = f_obs[start_idx:end_idx] + np.linspace(
                -chirp_span / 2,
                chirp_span / 2,
                tt.size,
            )
            chirp_phase = 2 * np.pi * np.cumsum(chirp_inst) / fs
            window = np.hanning(tt.size)
            chirp_sig[start_idx:end_idx] += window * np.exp(1j * chirp_phase)

        # rotor-related amplitude modulation (low-frequency envelope)
        rotor_env = 1.0 + 0.6 * np.sin(2 * np.pi * (fm_rate / 2) * t)

        sig = (0.6 * carrier + 0.3 * chirp_sig) * rotor_env

    # Add Gaussian noise according to SNR in dB (power relative to signal
    # power)
    sig_power = np.mean(np.abs(sig) ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (
        np.random.normal(size=sig.shape)
        + 1j * np.random.normal(size=sig.shape)
    )
    iq = (sig + noise).astype(np.complex64)
    return iq, f_obs


def build_scene_components(
    args,
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create stationary, moving, blades, combined IQ and f_obs (Hz)."""
    stationary = args.stationary_amp * np.exp(1j * 2 * np.pi * args.f0 * t)

    iq_move, f_obs = simulate_doppler(
        fs=args.fs,
        duration=args.duration,
        f0=args.f0,
        v=args.v,
        closest_approach=args.d0,
        snr_db=200.0,  # near-noiseless moving tone; noise is added later
        drone_mode=args.drone,
        fm_depth=args.fm_depth,
        fm_rate=args.fm_rate,
        chirp_interval=args.chirp_interval,
        chirp_width=args.chirp_width,
        chirp_span=args.chirp_span,
    )
    moving = args.moving_amp * iq_move

    blade_mod = args.blade_mod_hz * np.sin(
        2 * np.pi * args.blade_rot_hz * t
    )
    phase_blades = 2 * np.pi * np.cumsum(args.f0 + blade_mod) / args.fs
    blades = args.blades_amp * np.exp(1j * phase_blades)

    combined = stationary + moving + blades
    return stationary, moving, blades, combined, f_obs


def add_overall_noise(iq: np.ndarray, snr_db: float) -> np.ndarray:
    sig_power = np.mean(np.abs(iq) ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (
        np.random.normal(size=iq.shape) + 1j * np.random.normal(size=iq.shape)
    )
    return (iq + noise).astype(np.complex64)


def obs_freq_trace_khz(
    f_obs_hz: np.ndarray,
    times: np.ndarray,
    fs: int,
) -> np.ndarray:
    obs_f_khz = np.empty(times.shape[0], dtype=np.float64)
    for i, ft in enumerate(times):
        idx = int(ft * fs)
        idx = max(0, min(idx, len(f_obs_hz) - 1))
        obs_f_khz[i] = f_obs_hz[idx] / 1e3
    return obs_f_khz
