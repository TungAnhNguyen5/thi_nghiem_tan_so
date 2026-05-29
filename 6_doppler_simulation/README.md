# Doppler Simulation

This mini project simulates a moving sine-wave source (2D motion) and visualizes the Doppler frequency drift over time using a spectrogram.

Run the simulator:

```bash
cd 6_doppler_simulation
python doppler_simulator.py
```

Options include sample rate, duration, carrier frequency, source speed, and closest approach distance.

Outputs:

- PNG snapshot (default output path): `output/doppler_spectrogram.png`
- Animation: use `--animate gif` or `--animate mp4` (recommended)

Note: relative paths passed via `--output` (and `--save-iq`) are resolved under `6_doppler_simulation/`, so outputs always land in this mini-project folder regardless of where you run the command.

Examples:

```bash
cd 6_doppler_simulation

# PNG snapshot
python doppler_simulator.py --save-png --output output/doppler_scene.png

# MP4 animation (overlay or compare)
python doppler_simulator.py --drone --animate mp4 --animate-mode overlay --output output/doppler_scene.png
python doppler_simulator.py --drone --animate mp4 --animate-mode compare --output output/doppler_scene.png

# Tune the drone-like emitter
python doppler_simulator.py --drone \
	--fm-depth 80 --fm-rate 10 \
	--chirp-interval 0.15 --chirp-width 0.03 --chirp-span 800 \
	--animate mp4 --animate-mode overlay \
	--output output/doppler_drone.png
```

## Code layout

The implementation is split into small modules:

- `doppler_simulator.py` — CLI entrypoint (argument parsing + orchestration)
- `scene.py` — signal synthesis (Doppler motion, drone-like mode, scene components)
- `rendering.py` — spectrogram computation + PNG/GIF/MP4 rendering
