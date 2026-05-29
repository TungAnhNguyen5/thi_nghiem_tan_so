from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def run_script(script_path: Path, args: list[str] | None = None) -> None:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd += args
    completed = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Script {script_path} failed: returncode={completed.returncode}\nstdout={completed.stdout.decode()}\nstderr={completed.stderr.decode()}"
        )


def test_fft_visualizer_creates_png(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "1_fft_visualizer" / "fft_visualizer.py"
    out = Path("fft_visualizer.png")
    if out.exists():
        out.unlink()
    run_script(script)
    assert out.exists() and out.stat().st_size > 0
    out.unlink()


def test_spectrogram_creates_png(tmp_path: Path) -> None:
    base = Path(__file__).resolve().parents[1]
    gen = base / "2_uav_spectrogram" / "generate_iq.py"
    script = base / "2_uav_spectrogram" / "main.py"
    # generate a small dat file
    run_script(gen)
    dat = Path("hopping_iq.dat")
    assert dat.exists()

    outdir = tmp_path / "spec_out"
    outdir.mkdir()
    run_script(script, args=[str(dat), "--fs", "100000", "--output-dir", str(outdir)])
    pngs = list(outdir.glob("*.png"))
    assert pngs, "No png produced by spectrogram script"
    # cleanup
    dat.unlink()


def test_pca_visualizer_creates_png(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "4_pca_visualization" / "pca_visualizer.py"
    out = Path("pca_clusters.png")
    if out.exists():
        out.unlink()
    run_script(script)
    assert out.exists() and out.stat().st_size > 0
    out.unlink()
