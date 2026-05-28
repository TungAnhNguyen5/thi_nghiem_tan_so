# `thi_nghiem_tan_so`

Small signal-processing practice workspace with three mini projects:

- `fft_visualizer/` - generates a noisy sine-wave FFT plot
- `uav_spectrogram/` - renders spectrograms from interleaved float32 IQ `.dat` files
- `simple_feature_extraction/` - extracts signal power, bandwidth, and entropy from IQ `.dat` files

## Requirements

Use the repo virtual environment:

```bash
source /workspaces/codespaces-blank/thi_nghiem_tan_so/.venv/bin/activate
```

## Run the projects

FFT visualizer:

```bash
cd fft_visualizer
python fft_visualizer.py
```

UAV spectrogram:

```bash
cd uav_spectrogram
python main.py
```

Feature extraction:

```bash
cd simple_feature_extraction
python main.py ../uav_spectrogram
```

## Output

- `fft_visualizer/fft_visualizer.png`
- `uav_spectrogram/output/*.png`
- `simple_feature_extraction/output/*.csv`

## Notes

- IQ files are stored as interleaved float32 values: real, imag, real, imag, ...
- `simple_feature_extraction` writes one CSV per input file using the same stem name.