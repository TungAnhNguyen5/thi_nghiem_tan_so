from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from scene import (
    add_overall_noise,
    build_scene_components,
    obs_freq_trace_khz,
)
from rendering import (
    animate_spectrogram_compare,
    animate_spectrogram_overlay,
    compute_spectrogram,
    plot_spectrogram,
)


def _resolve_under_script_dir(path_str: str) -> Path:
    """Resolve a path so relative paths are under this script's folder.

    This keeps outputs deterministic regardless of the current working
    directory.
    Absolute paths are left unchanged.
    """
    path = Path(path_str)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate Doppler from a moving sine source and plot spectrogram"
        )
    )
    parser.add_argument(
        "--fs", type=int, default=20000, help="Sample rate (Hz)"
    )
    parser.add_argument(
        "--duration", type=float, default=6.0, help="Duration (s)"
    )
    parser.add_argument(
        "--f0", type=float, default=1000.0, help="Carrier frequency (Hz)"
    )
    parser.add_argument(
        "--v", type=float, default=30.0, help="Source speed (m/s)"
    )
    parser.add_argument(
        "--d0",
        type=float,
        default=50.0,
        help="Closest approach distance (m)",
    )
    parser.add_argument("--snr-db", type=float, default=30.0, help="SNR in dB")
    parser.add_argument(
        "--output",
        default="output/doppler_spectrogram.png",
        help="Output PNG path",
    )

    parser.add_argument(
        "--stationary-amp",
        type=float,
        default=0.8,
        help="Amplitude of stationary transmitter",
    )
    parser.add_argument(
        "--moving-amp",
        type=float,
        default=1.0,
        help="Amplitude of moving object",
    )
    parser.add_argument(
        "--blades-amp",
        type=float,
        default=0.5,
        help="Amplitude of rotating blades micro-Doppler",
    )
    parser.add_argument(
        "--blade-rot-hz",
        type=float,
        default=20.0,
        help="Rotation rate of blades (Hz)",
    )
    parser.add_argument(
        "--blade-mod-hz",
        type=float,
        default=40.0,
        help="Micro-Doppler modulation amplitude (Hz)",
    )

    parser.add_argument(
        "--save-iq",
        default=None,
        help="Optional path to save interleaved float32 IQ .dat file",
    )
    parser.add_argument(
        "--save-png",
        action="store_true",
        help="Also save a PNG snapshot of the combined spectrogram",
    )

    parser.add_argument(
        "--animate",
        choices=("gif", "mp4", "none"),
        default="none",
        help=(
            "Create an animated GIF/MP4 of the spectrogram "
            "(gif/mp4) or none"
        ),
    )
    parser.add_argument(
        "--animate-fps",
        type=int,
        default=20,
        help="Frames per second for animation",
    )
    parser.add_argument(
        "--animate-mode",
        choices=("overlay", "compare"),
        default="overlay",
        help=(
            "Animation mode: overlay moving trace (overlay) or side-by-"
            "side comparison of components (compare)"
        ),
    )

    parser.add_argument(
        "--drone",
        action="store_true",
        help=(
            "Use a more realistic drone-like moving emitter "
            "(FM carrier, chirps, rotor envelope)"
        ),
    )
    parser.add_argument(
        "--fm-depth",
        type=float,
        default=40.0,
        help="FM depth for drone carrier (Hz)",
    )
    parser.add_argument(
        "--fm-rate",
        type=float,
        default=5.0,
        help="FM modulation rate for drone carrier (Hz)",
    )
    parser.add_argument(
        "--chirp-interval",
        type=float,
        default=0.25,
        help="Interval between chirp pulses (s)",
    )
    parser.add_argument(
        "--chirp-width",
        type=float,
        default=0.02,
        help="Duration of each chirp pulse (s)",
    )
    parser.add_argument(
        "--chirp-span",
        type=float,
        default=400.0,
        help="Frequency span of each chirp (Hz)",
    )

    args = parser.parse_args()

    n = int(np.floor(args.fs * args.duration))
    t = np.arange(n) / args.fs

    stationary, moving, blades, combined, f_obs = build_scene_components(
        args,
        t,
    )
    iq = add_overall_noise(combined, args.snr_db)

    out_path = _resolve_under_script_dir(args.output)

    if args.save_iq:
        interleaved = np.empty(iq.size * 2, dtype=np.float32)
        interleaved[0::2] = np.real(iq)
        interleaved[1::2] = np.imag(iq)
        iq_path = _resolve_under_script_dir(args.save_iq)
        iq_path.parent.mkdir(parents=True, exist_ok=True)
        iq_path.write_bytes(interleaved.tobytes())

    spec_comb, freqs, times = compute_spectrogram(
        iq,
        fs=args.fs,
        nfft=2048,
        hop=256,
    )

    if args.save_png:
        png_path = out_path.with_suffix(".png")
        plot_spectrogram(
            spec_comb,
            freqs,
            times,
            png_path,
            title=f"Doppler Scene: v={args.v} m/s, d0={args.d0} m",
        )
        print(f"Wrote {png_path}")

    if args.animate in ("gif", "mp4"):
        anim_path = out_path.with_suffix("." + args.animate)
        if args.animate_mode == "overlay":
            obs_f_khz = obs_freq_trace_khz(f_obs, times, args.fs)
            animate_spectrogram_overlay(
                spec_comb,
                freqs,
                times,
                anim_path,
                obs_f_khz,
                fps=args.animate_fps,
            )
            print(f"Wrote {anim_path}")
        else:
            spec_moving, _, _ = compute_spectrogram(
                moving.astype(np.complex64),
                fs=args.fs,
                nfft=2048,
                hop=256,
            )
            spec_blades, _, _ = compute_spectrogram(
                blades.astype(np.complex64),
                fs=args.fs,
                nfft=2048,
                hop=256,
            )
            animate_spectrogram_compare(
                [spec_comb, spec_moving, spec_blades],
                freqs,
                times,
                anim_path,
                fps=args.animate_fps,
                titles=["Combined", "Moving component", "Blades"],
            )
            print(f"Wrote {anim_path}")
        return

    plot_spectrogram(
        spec_comb,
        freqs,
        times,
        out_path,
        title=f"Doppler Scene: v={args.v} m/s, d0={args.d0} m",
    )


if __name__ == "__main__":
    main()
