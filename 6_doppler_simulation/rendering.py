from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def compute_spectrogram(
    iq: np.ndarray,
    fs: int,
    nfft: int = 2048,
    hop: int = 256,
):
    if iq.size < nfft:
        raise ValueError("IQ capture is too short for the selected FFT size")

    window = np.hanning(nfft).astype(np.float32)
    frame_count = 1 + (iq.size - nfft) // hop

    spec = np.empty((nfft, frame_count), dtype=np.float32)

    for index in range(frame_count):
        start = index * hop
        frame = iq[start:start + nfft]
        spectrum = np.fft.fftshift(np.fft.fft(frame * window, n=nfft))
        spec[:, index] = 20 * np.log10(np.abs(spectrum) + 1e-12)

    frequencies = np.fft.fftshift(np.fft.fftfreq(nfft, d=1 / fs)) / 1e3
    times = (np.arange(frame_count) * hop + nfft / 2) / fs
    return spec, frequencies, times


def plot_spectrogram(
    spec,
    freqs,
    times,
    output_path: Path,
    title: str = "Doppler Spectrogram",
):
    plt.close("all")
    fig, ax = plt.subplots(figsize=(12, 6))

    img = ax.imshow(
        spec,
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        cmap="magma",
    )

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    fig.colorbar(img, ax=ax, label="Magnitude (dB)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")


def _save_animation_frames(
    frames: list[np.ndarray], output_path: Path, fps: int
) -> None:
    """Save RGB frames to GIF or MP4 based on the output suffix."""
    try:
        import imageio
    except Exception:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "imageio is required to create animations; install it with "
            "`pip install imageio pillow imageio-ffmpeg`"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        imageio.mimsave(str(output_path), frames, fps=fps)
        return
    if suffix == ".mp4":
        try:
            writer = imageio.get_writer(
                str(output_path),
                fps=fps,
                codec="libx264",
                quality=8,
                macro_block_size=None,
            )
        except TypeError:
            writer = imageio.get_writer(
                str(output_path),
                fps=fps,
                codec="libx264",
                quality=8,
            )
        with writer:
            for frame in frames:
                writer.append_data(frame)
        return

    raise ValueError(
        f"Unsupported animation output format: {suffix}. Use .gif or .mp4"
    )


def _render_spectrogram_frames(
    spec: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    overlay: dict | None,
) -> list[np.ndarray]:
    """Render frames for a spectrogram with an optional overlay.

    overlay dict supports:
      - mode: "trace"
      - trace_times: np.ndarray
      - trace_freq_khz: np.ndarray
      - title: str
      - vmin/vmax: float
    """
    plt.close("all")
    fig, ax = plt.subplots(figsize=(12, 6))

    vmin = np.percentile(spec, 2)
    vmax = np.percentile(spec, 98)
    if overlay and "vmin" in overlay:
        vmin = overlay["vmin"]
    if overlay and "vmax" in overlay:
        vmax = overlay["vmax"]

    img = ax.imshow(
        spec,
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title((overlay or {}).get("title", "Animated Doppler Spectrogram"))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    fig.colorbar(img, ax=ax, label="Magnitude (dB)")

    frames: list[np.ndarray] = []
    n_frames = spec.shape[1]

    trace_times = None
    trace_freq_khz = None
    if overlay and overlay.get("mode") == "trace":
        trace_times = overlay.get("trace_times")
        trace_freq_khz = overlay.get("trace_freq_khz")
        if trace_times is None or trace_freq_khz is None:
            raise ValueError(
                "trace overlay requires trace_times and trace_freq_khz"
            )
        if trace_freq_khz.shape[0] != n_frames:
            trace_freq_khz = np.interp(
                np.linspace(0, trace_freq_khz.shape[0] - 1, n_frames),
                np.arange(trace_freq_khz.shape[0]),
                trace_freq_khz,
            )

    for idx in range(n_frames):
        tpos = times[idx]
        line = ax.axvline(tpos, color="white", linewidth=1.2)
        trace_line = None
        if trace_times is not None and trace_freq_khz is not None:
            (trace_line,) = ax.plot(
                trace_times[: idx + 1],
                trace_freq_khz[: idx + 1],
                color="cyan",
                linewidth=1.5,
            )
        fig.canvas.draw()

        w, h = fig.canvas.get_width_height()
        try:
            buf = fig.canvas.tostring_rgb()
            data = np.frombuffer(buf, dtype=np.uint8)
            frame = data.reshape((h, w, 3))
        except Exception:
            buf = fig.canvas.tostring_argb()
            data = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
            frame = data[:, :, 1:4]

        frames.append(frame)
        line.remove()
        if trace_line is not None:
            trace_line.remove()

    return frames


def animate_spectrogram(
    spec,
    freqs,
    times,
    output_path: Path,
    fps: int = 20,
) -> None:
    frames = _render_spectrogram_frames(spec, freqs, times, overlay=None)
    _save_animation_frames(frames, output_path, fps=fps)


def animate_spectrogram_overlay(
    spec,
    freqs,
    times,
    output_path: Path,
    obs_freq_khz: np.ndarray,
    fps: int = 20,
) -> None:
    frames = _render_spectrogram_frames(
        spec,
        freqs,
        times,
        overlay={
            "mode": "trace",
            "trace_times": times,
            "trace_freq_khz": obs_freq_khz,
            "title": "Animated Doppler Spectrogram",
        },
    )
    _save_animation_frames(frames, output_path, fps=fps)


def animate_spectrogram_compare(
    specs,
    freqs,
    times,
    output_path: Path,
    fps: int = 20,
    titles=None,
) -> None:
    """Create side-by-side comparison animation for a list of spectrograms."""
    try:
        import imageio  # noqa: F401
    except Exception:  # pragma: no cover
        raise RuntimeError(
            "imageio is required to create animations; install it with "
            "`pip install imageio pillow imageio-ffmpeg`"
        )

    n = len(specs)
    plt.close("all")
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]

    all_data = np.hstack([s.flatten() for s in specs])
    vmin = np.percentile(all_data, 2)
    vmax = np.percentile(all_data, 98)

    for ax, spec in zip(axes, specs):
        ax.imshow(
            spec,
            origin="lower",
            aspect="auto",
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
        )

    for i, ax in enumerate(axes):
        ax.set_xlabel("Time (s)")
        if i == 0:
            ax.set_ylabel("Frequency (kHz)")
        if titles and i < len(titles):
            ax.set_title(titles[i])

    frames: list[np.ndarray] = []
    n_frames = specs[0].shape[1]
    for idx in range(n_frames):
        tpos = times[idx]
        lines = [ax.axvline(tpos, color="white", linewidth=1.2) for ax in axes]
        fig.canvas.draw()

        w, h = fig.canvas.get_width_height()
        try:
            buf = fig.canvas.tostring_rgb()
            data = np.frombuffer(buf, dtype=np.uint8)
            frame = data.reshape((h, w, 3))
        except Exception:
            buf = fig.canvas.tostring_argb()
            data = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
            frame = data[:, :, 1:4]

        frames.append(frame)
        for line in lines:
            line.remove()

    _save_animation_frames(frames, output_path, fps=fps)
