from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def run_script(script_path: Path, args: list[str] | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd += args
    return subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def test_doppler_scene_creates_png(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "6_doppler_simulation" / "doppler_simulator.py"
    out = tmp_path / "doppler_test.png"
    if out.exists():
        out.unlink()

    completed = run_script(
        script,
        args=[
            "--fs",
            "8000",
            "--duration",
            "1",
            "--f0",
            "1200",
            "--v",
            "12",
            "--d0",
            "25",
            "--snr-db",
            "30",
            "--stationary-amp",
            "0.6",
            "--moving-amp",
            "1.0",
            "--blades-amp",
            "0.4",
            "--blade-rot-hz",
            "15",
            "--blade-mod-hz",
            "60",
            "--output",
            str(out),
        ],
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"Doppler script failed:\nstdout={completed.stdout.decode()}\nstderr={completed.stderr.decode()}"
        )

    assert out.exists() and out.stat().st_size > 0


def test_doppler_scene_saves_iq(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "6_doppler_simulation" / "doppler_simulator.py"
    out_dat = tmp_path / "doppler_test.iq.dat"
    out_png = tmp_path / "doppler_test2.png"

    completed = run_script(
        script,
        args=[
            "--fs",
            "8000",
            "--duration",
            "0.8",
            "--f0",
            "1500",
            "--v",
            "8",
            "--d0",
            "20",
            "--snr-db",
            "20",
            "--save-iq",
            str(out_dat),
            "--output",
            str(out_png),
        ],
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"Doppler script failed when saving IQ:\nstdout={completed.stdout.decode()}\nstderr={completed.stderr.decode()}"
        )

    assert out_png.exists() and out_png.stat().st_size > 0
    assert out_dat.exists() and out_dat.stat().st_size > 0
